use crate::graph::{self, Graph, Op};

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
        let local_size_x = 16;
        let local_size_y = 16;
        let local_size_z = 1;

        // Number of workgroups in each dimension
        let num_workgroups_x = (output_dims[0] * output_dims[1] + local_size_x - 1) / local_size_x;
        let num_workgroups_y = (output_dims[2] + local_size_y - 1) / local_size_y;
        let num_workgroups_z = (output_dims[3] + local_size_z - 1) / local_size_z;

        [
            num_workgroups_x as u32,
            num_workgroups_y as u32,
            num_workgroups_z as u32,
        ]
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
}
