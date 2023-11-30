use serde::Serialize;

use crate::errors::GosonnxError;
use crate::{
    graph::{Graph, Op},
    ops::to_csv_str,
};

use super::{Compile, ShaderTemplate};

#[derive(Debug, Serialize, Clone)]
pub struct MaxPoolOp {
    ceil_mode: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl MaxPoolOp {
    pub fn new(ceil_mode: i64, kernel_shape: Vec<i64>, pads: Vec<i64>, strides: Vec<i64>) -> Self {
        Self {
            ceil_mode,
            kernel_shape,
            pads,
            strides,
        }
    }
}

impl Compile for &MaxPoolOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), GosonnxError> {
        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        shader_templ.push_attr("X_type", &x.type_glsl());
        shader_templ.push_attr("X_type", &x.type_glsl());
        shader_templ.push_attr("Y_type", &y.type_glsl());
        shader_templ.push_attr("in_dim", &to_csv_str(&x.shape()));
        shader_templ.push_attr("out_dim", &to_csv_str(&y.shape()));

        shader_templ.push_attr("ceil_mode", &self.ceil_mode.to_string());
        shader_templ.push_attr("kernel_shape", &to_csv_str(&self.kernel_shape));
        shader_templ.push_attr("pads", &to_csv_str(&self.pads));
        shader_templ.push_attr("strides", &to_csv_str(&self.strides));

        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();

        // Local sizes are hardware-dependent; these are just example values.
        let local_size_x = 16;
        let local_size_y = 4;
        let local_size_z = 4;

        // Compute number of workgroups needed for each dimension based on the output tensor shape.
        // Ceil to account for any remaining threads.
        let workgroup_size_x = ((output_dims[3] + local_size_x - 1) / local_size_x) as u32;
        let workgroup_size_y = ((output_dims[2] + local_size_y - 1) / local_size_y) as u32;
        let workgroup_size_z =
            ((output_dims[0] * output_dims[1] + local_size_z - 1) / local_size_z) as u32;

        [workgroup_size_x, workgroup_size_y, workgroup_size_z]
    }

    fn activable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    use super::MaxPoolOp;

    #[test]
    fn simple_pool() {
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
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]).unwrap();
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "my_maxpool",
                OpType::MaxPool {
                    attr: MaxPoolOp::new(0, vec![2, 2], vec![0, 0, 0, 0], vec![1, 1]),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![5.0, 6.0, 8.0, 9.0, -1.0, -2.0, -4.0, -5.0])
            );
        }
    }
}
