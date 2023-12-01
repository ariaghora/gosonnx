use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::Error;
use crate::{
    graph::{Graph, Op},
    ops::to_csv_str,
};

use super::{Compile, ShaderTemplate};

#[derive(Debug, Serialize, Clone)]
pub struct FlattenOp {
    axis: i64,
}

impl FlattenOp {
    pub fn new(axis: i64) -> Self {
        Self { axis }
    }
}

impl Compile for &FlattenOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), GosonnxError> {
        if self.axis < 0 {
            return Err(Error("Cannot handle negative axis yet".into()));
        }

        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        shader_templ.push_attr("X_type", &x.type_glsl());
        shader_templ.push_attr("Y_type", &y.type_glsl());
        shader_templ.push_attr("in_dim", &to_csv_str(&x.shape()));
        shader_templ.push_attr("out_dim", &to_csv_str(&y.shape()));
        shader_templ.push_attr("in_ndim", &x.shape().len());
        shader_templ.push_attr("out_ndim", &y.shape().len());
        shader_templ.push_attr("axis", &self.axis);

        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();
        let local_size_x = 32;
        let local_size_y = 8;

        // Number of workgroups in each dimension
        let num_workgroups_x = (output_dims[1] + local_size_x - 1) / local_size_x;
        let num_workgroups_y = (output_dims[0] + local_size_y - 1) / local_size_y;
        let num_workgroups_z = 1; // Since the output tensor is 2D, the z-dimension is always 1.

        [
            num_workgroups_x as u32,
            num_workgroups_y as u32,
            num_workgroups_z as u32,
        ]
    }

    fn activable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::ops::activation_op::ActivationOp;
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    use super::FlattenOp;

    #[test]
    fn simple_flatten() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0,
                    -6.0, -7.0, -8.0, -9.0,
                ]),
                vec![1, 2, 3, 3],
            )
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 18]).unwrap();
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "flatten",
                OpType::Flatten {
                    attr: FlattenOp::new(1),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0,
                    -6.0, -7.0, -8.0, -9.0
                ])
            );
        }
    }

    #[test]
    fn simple_flatten_relu() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0,
                    -6.0, -7.0, -8.0, -9.0,
                ]),
                vec![1, 2, 3, 3],
            )
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 18]).unwrap();
        graph.new_tensor_f32("Z", None, vec![1, 18]).unwrap();
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "flatten",
                OpType::Flatten {
                    attr: FlattenOp::new(1),
                },
            )
            .unwrap();
        graph
            .new_op(
                vec!["Y"],
                vec!["Z"],
                "relu",
                OpType::Relu {
                    attr: ActivationOp::new(vec![]),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Z").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0
                ])
            );
        }
    }
}
