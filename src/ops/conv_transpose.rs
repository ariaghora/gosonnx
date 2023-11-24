use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::Error;
use crate::graph::{Graph, Op};

use super::{to_csv_str, Compile, ShaderTemplate};

#[derive(Debug, Serialize)]
pub struct ConvTransposeOp {
    dilations: Option<Vec<i64>>,
    group: Option<i64>,
    kernel_shape: Option<Vec<i64>>,
    output_padding: Option<Vec<i64>>,
    output_shape: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
}

impl ConvTransposeOp {
    pub fn new(
        dilations: Option<Vec<i64>>,
        group: Option<i64>,
        kernel_shape: Option<Vec<i64>>,
        output_padding: Option<Vec<i64>>,
        output_shape: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
    ) -> Self {
        Self {
            dilations,
            group,
            kernel_shape,
            output_padding,
            output_shape,
            pads,
            strides,
        }
    }
}

impl Compile for &ConvTransposeOp {
    fn compile(
        &self,
        op: &Op,
        shader_template: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError> {
        if let Some(g) = self.group {
            if g > 1 {
                return Err(Error("Only group=1 is supported".to_string()));
            }
        }

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

        shader_template.push_attr(
            "dilations",
            &to_csv_str(&self.dilations.as_ref().unwrap_or(&vec![1, 1])),
        );
        shader_template.push_attr("group", &self.group);
        shader_template.push_attr(
            "kernel_shape",
            &to_csv_str(&self.kernel_shape.as_ref().unwrap()),
        );
        shader_template.push_attr(
            "output_padding",
            &to_csv_str(&self.output_padding.as_ref().unwrap_or(&vec![0, 0])),
        );
        shader_template.push_attr(
            "pads",
            &to_csv_str(&self.pads.as_ref().unwrap_or(&vec![0, 0, 0, 0])),
        );
        shader_template.push_attr(
            "strides",
            &to_csv_str(&self.strides.as_ref().unwrap_or(&vec![1, 1])),
        );
        shader_template.push_attr("output_channels", &w.shape()[1]);

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
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
        utils::vec_close,
    };

    use super::ConvTransposeOp;

    #[test]
    fn conv_and_bias() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("X", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 1, 2, 2])
            .unwrap();
        graph
            .new_tensor_f32("W", Some(vec![0.1, 0.2, 0.3, 0.4]), vec![1, 1, 2, 2])
            .unwrap();
        graph.new_tensor_f32("b", Some(vec![0.5]), vec![1]).unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 1, 3, 3]).unwrap();
        graph
            .new_op(
                vec!["X", "W", "b"],
                vec!["Y"],
                "my_conv",
                OpType::ConvTranspose {
                    attr: ConvTransposeOp::new(
                        Some(vec![1, 1]),
                        Some(1),
                        Some(vec![2, 2]),
                        None,
                        None,
                        Some(vec![0, 0, 0, 0]),
                        Some(vec![1, 1]),
                    ),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert!(vec_close(
                values.as_ref().unwrap().clone(),
                vec![0.6000, 0.9000, 0.9000, 1.1000, 2.5000, 2.1000, 1.4000, 2.9000, 2.1000]
            ));
        }
    }

    #[test]
    fn conv_and_bias_larger_input() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some(vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                    18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
                    34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
                    50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
                    66., 67., 68., 69., 70., 71., 72., 73., 74.,
                ]),
                vec![1, 3, 5, 5],
            )
            .unwrap();
        graph
            .new_tensor_f32(
                "W",
                Some(vec![
                    0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000,
                    1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000,
                    2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000, 2.7000, 2.8000, 2.9000,
                    3.0000, 3.1000, 3.2000, 3.3000, 3.4000, 3.5000, 3.6000, 3.7000, 3.8000, 3.9000,
                    4.0000, 4.1000, 4.2000, 4.3000, 4.4000, 4.5000, 4.6000, 4.7000, 4.8000, 4.9000,
                    5.0000, 5.1000, 5.2000, 5.3000,
                ]),
                vec![3, 2, 3, 3],
            )
            .unwrap();
        graph
            .new_tensor_f32("b", Some(vec![0.5, 0.5]), vec![2])
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 2, 7, 7]).unwrap();
        graph
            .new_op(
                vec!["X", "W", "b"],
                vec!["Y"],
                "my_conv",
                OpType::ConvTranspose {
                    attr: ConvTransposeOp::new(
                        Some(vec![1, 1]),
                        Some(1),
                        Some(vec![3, 3]),
                        None,
                        None,
                        Some(vec![0, 0, 0, 0]),
                        Some(vec![1, 1]),
                    ),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert!(vec_close(
                values.as_ref().unwrap().clone(),
                vec![
                    225.5000, 463.4000, 714.5000, 731.6000, 748.7000, 513.8000, 264.5000, 500.0000,
                    1027.7000, 1584.2000, 1621.1000, 1658.0000, 1136.8999, 584.6000, 828.5000,
                    1702.3999, 2623.0999, 2682.5000, 2741.9001, 1878.7999, 965.3000, 923.0000,
                    1895.8999, 2920.0999, 2979.5000, 3038.9001, 2081.3000, 1068.8000, 1017.5000,
                    2089.3999, 3217.0999, 3276.4998, 3335.9001, 2283.8000, 1172.3000, 752.0000,
                    1542.5000, 2372.6001, 2414.8999, 2457.2002, 1680.5000, 861.8000, 414.5000,
                    849.2000, 1304.9000, 1327.3999, 1349.9000, 922.4000, 472.7000, 293.0000,
                    601.1000, 925.1000, 950.2999, 975.5000, 667.7000, 342.8000, 648.5000,
                    1330.1000, 2045.8999, 2099.0000, 2152.1001, 1471.7000, 754.7000, 1071.5000,
                    2196.5000, 3376.3999, 3460.0999, 3543.8003, 2421.5000, 1240.7000, 1206.5000,
                    2471.0000, 3794.8999, 3878.5999, 3962.2998, 2705.0000, 1384.7000, 1341.5000,
                    2745.5000, 4213.3999, 4297.0996, 4380.7998, 2988.5000, 1528.7000, 981.5000,
                    2006.9000, 3077.3000, 3135.7998, 3194.3003, 2177.3000, 1112.9000, 536.0000,
                    1094.9000, 1677.5000, 1708.1000, 1738.7000, 1184.3000, 605.0000
                ]
            ));
        }
    }
}
