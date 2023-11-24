use serde::Serialize;

use crate::errors::GosonnxError;
use crate::graph::{Graph, Op};

use super::{to_csv_str, Compile, ShaderTemplate};

#[derive(Debug, Serialize)]
pub struct ConvOp {
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvOp {
    pub fn new(
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
    ) -> Self {
        Self {
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        }
    }
}

impl Compile for &ConvOp {
    fn compile(
        &self,
        op: &Op,
        shader_template: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError> {
        let x = &graph.tensor_map[&op.inputs[0]];
        let w = &graph.tensor_map[&op.inputs[1]];
        let y = &graph.tensor_map[&op.outputs[0]];

        shader_template.push_attr("X_type", &x.type_glsl());
        shader_template.push_attr("W_type", &w.type_glsl());
        // Output type is assumed to be identical with input type
        shader_template.push_attr("Y_type", &x.type_glsl());

        shader_template.push_attr("in_dim", &to_csv_str(&x.shape()));
        shader_template.push_attr("weight_dim", &to_csv_str(&w.shape()));
        shader_template.push_attr("out_dim", &to_csv_str(&y.shape()));

        shader_template.push_attr("dilations", &to_csv_str(&self.dilations));
        shader_template.push_attr("group", &self.group);
        shader_template.push_attr("kernel_shape", &to_csv_str(&self.kernel_shape));
        shader_template.push_attr("pads", &to_csv_str(&self.pads));
        shader_template.push_attr("strides", &to_csv_str(&self.strides));
        shader_template.push_attr("output_channels", &w.shape()[0]);

        if op.inputs.len() > 2 {
            shader_template.push_attr("use_bias", &true);
            let b = &graph.tensor_map[&op.inputs[2]];
            shader_template.push_attr("B_type", &b.type_glsl());
        }

        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();
        let local_size_x_y = 16;

        let workgroup_size_x = ((output_dims[3] as f64) / (local_size_x_y as f64)).ceil() as u32; // width
        let workgroup_size_y = ((output_dims[2] as f64) / (local_size_x_y as f64)).ceil() as u32; // height
        let workgroup_size_z = output_dims[0] as u32 * output_dims[1] as u32; // batch * channels

        [workgroup_size_x, workgroup_size_y, workgroup_size_z]
    }
}

#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    use super::ConvOp;

    #[test]
    fn conv_and_bias() {
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
        graph
            .new_tensor_f32(
                "W",
                Some(vec![
                    0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0,
                    0.0,
                ]),
                vec![2, 2, 2, 2],
            )
            .unwrap();
        graph
            .new_tensor_f32("b", Some(vec![1.0, -1.0]), vec![2])
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "W", "b"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    attr: ConvOp::new(vec![1, 1], 1, vec![2, 2], vec![0, 0, 0, 0], vec![1, 1]),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(values, &Some(vec![3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]));
        }
    }

    #[test]
    fn conv_without_bias() {
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
        graph
            .new_tensor_f32(
                "W",
                Some(vec![
                    0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0,
                    0.0,
                ]),
                vec![2, 2, 2, 2],
            )
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "W"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    attr: ConvOp::new(vec![1, 1], 1, vec![2, 2], vec![0, 0, 0, 0], vec![1, 1]),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(values, &Some(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]));
        }
    }

    #[test]
    fn conv_larger_bias() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some(vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                ]),
                vec![1, 3, 3, 3],
            )
            .unwrap();
        graph
            .new_tensor_f32(
                "W",
                Some(vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                ]),
                vec![2, 3, 2, 2],
            )
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]).unwrap();
        graph
            .new_op(
                vec!["X", "W"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    attr: ConvOp::new(vec![1, 1], 1, vec![2, 2], vec![0, 0, 0, 0], vec![1, 1]),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1035.0, 1101.0, 1233.0, 1299.0, 2619.0, 2829.0, 3249.0, 3459.0
                ])
            );
        }
    }

    #[test]
    fn conv_and_bias_grouped() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "X",
            Some((1..=64).map(|v| v as f32).collect()),
            vec![1, 4, 4, 4],
        )?;
        graph.new_tensor_f32(
            "W",
            Some((0..4 * 2 * 3 * 3).map(|v| v as f32).collect()),
            vec![4, 2, 3, 3],
        )?;
        graph.new_tensor_f32("b", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![4])?;
        graph.new_tensor_f32("Y", None, vec![1, 4, 2, 2])?;
        graph.new_op(
            vec!["X", "W", "b"],
            vec!["Y"],
            "my_conv",
            OpType::Conv {
                attr: ConvOp::new(vec![1, 1], 2, vec![3, 3], vec![0, 0, 0, 0], vec![1, 1]),
            },
        )?;
        graph.run()?;
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    2947.0, 3100.0, 3559.0, 3712.0, 7483.0, 7960.0, 9391.0, 9868.0, 37652.0,
                    38453.0, 40856.0, 41657.0, 52556.0, 53681.0, 57056.0, 58181.0
                ])
            );
        }
        Ok(())
    }
}
