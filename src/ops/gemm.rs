use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::{Error, OpsOnIncompatibleTypeError};
use crate::graph::{Graph, Op};

use super::{Compile, ShaderTemplate};

#[derive(Debug, Serialize, Clone)]
pub struct GemmOp {
    alpha: Option<f32>,
    beta: Option<f32>,
    trans_a: Option<i64>,
    trans_b: Option<i64>,
    pub activation: Option<String>,
}

impl GemmOp {
    pub fn new(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i64>,
        trans_b: Option<i64>,
    ) -> Self {
        Self {
            alpha,
            beta,
            trans_a,
            trans_b,
            activation: None,
        }
    }
}

impl Compile for &GemmOp {
    fn compile(
        &self,
        op: &Op,
        shader_templ: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError> {
        shader_templ.push_attr("use_bias", &(op.inputs.len() > 2));

        let alpha = self.alpha.unwrap_or(1.0);
        let beta = self.beta.unwrap_or(1.0);
        shader_templ.push_attr("alpha", &alpha);
        shader_templ.push_attr("beta", &beta);

        let trans_a = self.trans_a.unwrap_or(0);
        let trans_b = self.trans_b.unwrap_or(0);
        shader_templ.push_attr("trans_a", &trans_a);
        shader_templ.push_attr("trans_b", &trans_b);

        let t_a = &graph.tensor_map[&op.inputs[0]];
        let t_b = &graph.tensor_map[&op.inputs[1]];
        let a_type = t_a.type_glsl();
        let b_type = t_b.type_glsl();
        if a_type != b_type {
            return Err(OpsOnIncompatibleTypeError {
                left: a_type,
                right: b_type,
            });
        }
        shader_templ.push_attr("in_type", &a_type);
        shader_templ.push_attr("out_type", &b_type);

        let m = if trans_a == 0 {
            t_a.shape()[0]
        } else {
            t_a.shape()[1]
        };
        let (k, n) = if trans_b == 0 {
            (t_b.shape()[0], t_b.shape()[1])
        } else {
            (t_b.shape()[1], t_b.shape()[0])
        };

        shader_templ.push_attr("m", &m);
        shader_templ.push_attr("k", &k);
        shader_templ.push_attr("n", &n);

        if op.inputs.len() > 2 {
            if let Some(bias) = &graph.tensor_map.get(&op.inputs[2]) {
                if bias.type_glsl() != b_type {
                    return Err(Error(
                        "Bias type must be identical with input and weight tensors".into(),
                    ));
                }
                shader_templ.push_attr("bias_type", &bias.type_glsl());
                if bias.shape().len() == 2 {
                    shader_templ.push_attr("bias_h", &bias.shape()[0]);
                    shader_templ.push_attr("bias_w", &bias.shape()[1]);
                } else if bias.shape().len() == 1 {
                    shader_templ.push_attr("bias_h", &1);
                    shader_templ.push_attr("bias_w", &bias.shape()[0]);
                } else {
                    return Err(Error("Cannot handle bias with rank more than 2".into()));
                }
            }
        }

        shader_templ.push_attr("activation", &self.activation);

        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let out = &graph.tensor_map[&op.outputs[0]];
        let (m, n) = (out.shape()[0], out.shape()[1]);
        let local_size_x = 16;
        let local_size_y = 16;

        // Number of workgroups in each dimension
        let num_workgroups_x = (n + local_size_x - 1) / local_size_x;
        let num_workgroups_y = (m + local_size_y - 1) / local_size_y;

        [num_workgroups_x as u32, num_workgroups_y as u32, 1]
    }

    fn activable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    #[test]
    fn simple_gemm() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        graph
            .new_tensor_f32("Y", Some(vec![1.0, 1.0]), vec![2, 1])
            .unwrap();
        graph.new_tensor_f32("output", None, vec![2, 1]).unwrap();
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                "my_gemm",
                OpType::Gemm {
                    attr: super::GemmOp::new(Some(1.0), Some(1.0), Some(0), Some(0)),
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("output") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![3.0, 7.0]))
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }

    #[test]
    fn gemm_2x2() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        graph
            .new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2])
            .unwrap();
        graph.new_tensor_f32("output", None, vec![2, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                "my_gemm",
                OpType::Gemm {
                    attr: super::GemmOp::new(Some(1.0), Some(1.0), Some(0), Some(0)),
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("output") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![5.0, 5.0, 11.0, 11.0]));
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }

    #[test]
    fn gemm_5x2() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                vec![5, 2],
            )
            .unwrap();
        graph
            .new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2])
            .unwrap();
        graph.new_tensor_f32("output", None, vec![5, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                "my_gemm",
                OpType::Gemm {
                    attr: super::GemmOp::new(Some(1.0), Some(1.0), Some(0), Some(0)),
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("output") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some(vec![
                        5.0, 5.0, 11.0, 11.0, 17.0, 17.0, 23.0, 23.0, 29.0, 29.0
                    ])
                );
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }

    #[test]
    fn gemm_bias_no_broadcast() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        graph
            .new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2])
            .unwrap();
        graph
            .new_tensor_f32("bias", Some(vec![2.0, 2.0, 3.0, 3.0]), vec![2, 2])
            .unwrap();
        graph.new_tensor_f32("output", None, vec![2, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "Y", "bias"],
                vec!["output"],
                "my_gemm",
                OpType::Gemm {
                    attr: super::GemmOp::new(Some(1.0), Some(1.0), Some(0), Some(0)),
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("output") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![7.0, 7.0, 14.0, 14.0]));
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }

    #[test]
    fn gemm_bias_broadcast() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        graph
            .new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2])
            .unwrap();
        graph
            .new_tensor_f32("bias", Some(vec![2.0, 3.0]), vec![2, 1])
            .unwrap();
        graph.new_tensor_f32("output", None, vec![2, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "Y", "bias"],
                vec!["output"],
                "my_gemm",
                OpType::Gemm {
                    attr: super::GemmOp::new(Some(1.0), Some(1.0), Some(0), Some(0)),
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("output") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![7.0, 7.0, 14.0, 14.0]));
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }
    #[test]
    fn gemm_3x3_trans_b() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("X", Some((1..=9).map(|v| v as f32).collect()), vec![3, 3])
            .unwrap();
        graph
            .new_tensor_f32("Y", Some((1..=9).map(|v| v as f32).collect()), vec![3, 3])
            .unwrap();
        graph.new_tensor_f32("output", None, vec![3, 3]).unwrap();
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                "my_gemm",
                OpType::Gemm {
                    attr: super::GemmOp::new(Some(1.0), Some(1.0), Some(0), Some(1)),
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("output") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some(vec![
                        14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0
                    ])
                );
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }
}
