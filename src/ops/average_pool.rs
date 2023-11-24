use crate::errors::GosonnxError;
use crate::errors::GosonnxError::{AttributeNotFound, InvalidInputDimension};
use serde::Serialize;

use crate::graph::{Graph, Op};

use super::{to_csv_str, Compile, ShaderTemplate};

#[derive(Debug, Serialize)]
pub struct AveragePoolOp {
    auto_pad: Option<String>,
    ceil_mode: Option<i64>,
    dilations: Option<Vec<i64>>,
    kernel_shape: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
}

impl AveragePoolOp {
    pub fn new(
        auto_pad: Option<String>,
        ceil_mode: Option<i64>,
        dilations: Option<Vec<i64>>,
        kernel_shape: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
    ) -> Self {
        Self {
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            pads,
            strides,
        }
    }
}
impl Compile for &AveragePoolOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), GosonnxError> {
        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        if x.shape().len() != 4 {
            return Err(InvalidInputDimension {
                expected: 4,
                found: x.shape().len(),
            });
        }

        shader_templ.push_attr("X_type", &x.type_glsl());
        shader_templ.push_attr("Y_type", &y.type_glsl());
        shader_templ.push_attr("in_dim", &to_csv_str(&x.shape()));
        shader_templ.push_attr("out_dim", &to_csv_str(&y.shape()));

        let auto_pad = &self.auto_pad.clone().unwrap_or("NOTSET".to_string());
        let ceil_mode = &self.ceil_mode.unwrap_or(0);
        let dilations = &self.dilations.clone().unwrap_or(vec![1; x.shape().len()]);
        let kernel_shape = if let Some(kernel_shape) = &self.kernel_shape {
            kernel_shape
        } else {
            return Err(AttributeNotFound("kernel_shape".to_string()));
        };

        let pads = &self.pads.clone().unwrap_or(vec![0, 0, 0, 0]);
        let strides = &self.strides.clone().unwrap_or(vec![1, 1]);

        shader_templ.push_attr("auto_pad", &auto_pad);
        shader_templ.push_attr("ceil_mode", &ceil_mode);
        shader_templ.push_attr("dilations", &to_csv_str(dilations));
        shader_templ.push_attr("kernel_shape", &to_csv_str(kernel_shape));
        shader_templ.push_attr("pads", &to_csv_str(pads));
        shader_templ.push_attr("strides", &to_csv_str(strides));

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
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    use super::AveragePoolOp;

    #[test]
    fn simple_global_average_pool() {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "X",
            Some((1..=18).map(|v| v as f32).collect()),
            vec![1, 2, 3, 3],
        );

        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "avg_pool",
                OpType::AveragePool {
                    attr: AveragePoolOp::new(
                        None,
                        Some(0),
                        Some(vec![1, 1]),
                        Some(vec![2, 2]),
                        Some(vec![0, 0, 0, 0]),
                        Some(vec![1, 1]),
                    ),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(values, &Some(vec![3., 4., 6., 7., 12., 13., 15., 16.]));
        }
    }
}
