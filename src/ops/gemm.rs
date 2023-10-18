use crate::{Graph, Op};

use super::Compile;

pub struct GemmOp {
    alpha: f32,
    beta: f32,
    trans_a: i32,
    trans_b: i32,
}

impl GemmOp {
    pub fn new(alpha: f32, beta: f32, trans_a: i32, trans_b: i32) -> Self {
        Self {
            alpha,
            beta,
            trans_a,
            trans_b,
        }
    }
}

impl Compile for GemmOp {
    fn compile(&self, op: &Op, shader_template: &str, graph: &Graph) -> Result<String, String> {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        tera.add_raw_template("Gemm", shader_template)
            .map_err(|e| e.to_string())?;

        context.insert("use_bias", &(op.inputs.len() > 2));
        context.insert("alpha", &self.alpha);
        context.insert("beta", &self.beta);
        context.insert("trans_a", &self.trans_a);
        context.insert("trans_b", &self.trans_b);

        let t_a = &graph.tensor_map[&op.inputs[0]];
        let t_b = &graph.tensor_map[&op.inputs[1]];

        context.insert("m", &t_a.shape()[0]);
        context.insert("k", &t_a.shape()[1]);
        context.insert("n", &t_b.shape()[1]);

        if op.inputs.len() > 2 {
            if let Some(bias) = &graph.tensor_map.get(&op.inputs[2]) {
                if bias.shape().len() == 2 {
                    context.insert("bias_h", &bias.shape()[0]);
                    context.insert("bias_w", &bias.shape()[1]);
                } else if bias.shape().len() == 1 {
                    context.insert("bias_h", &1);
                    context.insert("bias_w", &bias.shape()[0]);
                } else {
                    return Err("Cannot handle bias with rank more than 2".into());
                }
            }
        }

        let rendered = tera
            .render(&op.op_type.to_string(), &context)
            .map_err(|e| e.to_string());
        rendered
    }
}

#[cfg(test)]
mod tests {
    use crate::{Graph, Tensor};

    #[test]
    fn simple_gemm() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
        graph.new_tensor_f32("Y", Some(vec![1.0, 1.0]), vec![2, 1]);
        graph.new_tensor_f32("output", None, vec![2, 1]);
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                vec![],
                "my_gemm",
                crate::OpType::Gemm {
                    alpha: 1.,
                    beta: 1.,
                    trans_a: 0,
                    trans_b: 0,
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
        graph.new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
        graph.new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2]);
        graph.new_tensor_f32("output", None, vec![2, 2]);
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                vec![],
                "my_gemm",
                crate::OpType::Gemm {
                    alpha: 1.,
                    beta: 1.,
                    trans_a: 0,
                    trans_b: 0,
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
        graph.new_tensor_f32(
            "X",
            Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            vec![5, 2],
        );
        graph.new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2]);
        graph.new_tensor_f32("output", None, vec![5, 2]);
        graph
            .new_op(
                vec!["X", "Y"],
                vec!["output"],
                vec![],
                "my_gemm",
                crate::OpType::Gemm {
                    alpha: 1.,
                    beta: 1.,
                    trans_a: 0,
                    trans_b: 0,
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
        graph.new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
        graph.new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2]);
        graph.new_tensor_f32("bias", Some(vec![2.0, 2.0, 3.0, 3.0]), vec![2, 2]);
        graph.new_tensor_f32("output", None, vec![2, 2]);
        graph
            .new_op(
                vec!["X", "Y", "bias"],
                vec!["output"],
                vec![],
                "my_gemm",
                crate::OpType::Gemm {
                    alpha: 1.,
                    beta: 1.,
                    trans_a: 0,
                    trans_b: 0,
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
        graph.new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
        graph.new_tensor_f32("Y", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![2, 2]);
        graph.new_tensor_f32("bias", Some(vec![2.0, 3.0]), vec![2, 1]);
        graph.new_tensor_f32("output", None, vec![2, 2]);
        graph
            .new_op(
                vec!["X", "Y", "bias"],
                vec!["output"],
                vec![],
                "my_gemm",
                crate::OpType::Gemm {
                    alpha: 1.,
                    beta: 1.,
                    trans_a: 0,
                    trans_b: 0,
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
}
