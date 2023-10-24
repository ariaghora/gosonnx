use crate::graph::{Graph, Op};

use super::{to_csv_str, Compile};

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
    pub fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();
        let local_size_x_y = 16;

        // Number of threads needed based on output shape
        let num_threads = output_dims[0] * output_dims[2] * output_dims[3];

        // Number of threads per workgroup
        let threads_per_workgroup = local_size_x_y * local_size_x_y;

        // Number of workgroups needed
        let num_workgroups =
            (num_threads + threads_per_workgroup as i64 - 1) / threads_per_workgroup as i64;

        // Distribute workgroups evenly across two dimensions
        let workgroup_size_x = (num_workgroups as f64).sqrt().ceil() as i64;
        let workgroup_size_y = (num_workgroups + workgroup_size_x - 1) / workgroup_size_x;

        [workgroup_size_x as u32, workgroup_size_y as u32, 1]
    }
}

impl Compile for ConvOp {
    fn compile(&self, op: &Op, shader_template: &str, graph: &Graph) -> Result<String, String> {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template("Conv", shader_template)
            .map_err(|e| e.to_string())?;

        let x = &graph.tensor_map[&op.inputs[0]];
        let w = &graph.tensor_map[&op.inputs[1]];
        let y = &graph.tensor_map[&op.outputs[0]];

        context.insert("X_type", &x.type_glsl());
        context.insert("W_type", &w.type_glsl());
        // Output type is assumed to be identical with input type
        context.insert("Y_type", &x.type_glsl());

        context.insert("in_dim", &to_csv_str(&x.shape()));
        context.insert("weight_dim", &to_csv_str(&w.shape()));
        context.insert("out_dim", &to_csv_str(&y.shape()));

        context.insert("dilations", &to_csv_str(&self.dilations));
        context.insert("group", &self.group);
        context.insert("kernel_shape", &to_csv_str(&self.kernel_shape));
        context.insert("pads", &to_csv_str(&self.pads));
        context.insert("strides", &to_csv_str(&self.strides));
        context.insert("output_channels", &w.shape()[0]);

        if op.inputs.len() > 2 {
            context.insert("use_bias", &true);
            let b = &graph.tensor_map[&op.inputs[2]];
            context.insert("B_type", &b.type_glsl());
        }

        let compiled = tera.render("Conv", &context).map_err(|e| e.to_string())?;
        Ok(compiled)
    }
}

#[cfg(test)]
mod test {
    use crate::graph::{Graph, OpType, Tensor};

    #[test]
    fn conv_and_bias() {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "X",
            Some(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
                -7.0, -8.0, -9.0,
            ]),
            vec![1, 2, 3, 3],
        );
        graph.new_tensor_f32(
            "W",
            Some(vec![
                0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0,
            ]),
            vec![2, 2, 2, 2],
        );
        graph.new_tensor_f32("b", Some(vec![1.0, -1.0]), vec![2]);
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]);
        graph
            .new_op(
                vec!["X", "W", "b"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    dilations: vec![1, 1],
                    group: 1,
                    kernel_shape: vec![2, 2],
                    pads: vec![0, 0, 0, 0],
                    strides: vec![1, 1],
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
        graph.new_tensor_f32(
            "X",
            Some(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
                -7.0, -8.0, -9.0,
            ]),
            vec![1, 2, 3, 3],
        );
        graph.new_tensor_f32(
            "W",
            Some(vec![
                0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0,
            ]),
            vec![2, 2, 2, 2],
        );
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]);
        graph
            .new_op(
                vec!["X", "W"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    dilations: vec![1, 1],
                    group: 1,
                    kernel_shape: vec![2, 2],
                    pads: vec![0, 0, 0, 0],
                    strides: vec![1, 1],
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
        graph.new_tensor_f32(
            "X",
            Some(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
            ]),
            vec![1, 3, 3, 3],
        );
        graph.new_tensor_f32(
            "W",
            Some(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ]),
            vec![2, 3, 2, 2],
        );
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]);
        graph
            .new_op(
                vec!["X", "W"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    dilations: vec![1, 1],
                    group: 1,
                    kernel_shape: vec![2, 2],
                    pads: vec![0, 0, 0, 0],
                    strides: vec![1, 1],
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
}
