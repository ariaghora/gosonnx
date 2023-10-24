use crate::{
    graph::{Graph, Op},
    ops::to_csv_str,
};

use super::Compile;

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

    pub fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();
        let local_size_x_y = 16;

        // Calculate total number of threads needed in the output
        let num_threads = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];

        // Threads per workgroup
        let threads_per_workgroup = local_size_x_y * local_size_x_y;

        // Calculate total number of workgroups needed
        let num_workgroups = (num_threads + threads_per_workgroup - 1) / threads_per_workgroup;

        // Distribute workgroups evenly across two dimensions
        let workgroup_size_x = (num_workgroups as f64).sqrt().ceil() as i64;
        let workgroup_size_y = (num_workgroups + workgroup_size_x - 1) / workgroup_size_x;

        [workgroup_size_x as u32, workgroup_size_y as u32, 1]
    }
}

impl Compile for MaxPoolOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_source: &str,
        graph: &crate::graph::Graph,
    ) -> Result<String, String> {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        tera.add_raw_template("MaxPool", shader_source)
            .map_err(|e| e.to_string())?;

        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        context.insert("X_type", &x.type_glsl());
        context.insert("Y_type", &y.type_glsl());
        context.insert("in_dim", &to_csv_str(&x.shape()));
        context.insert("out_dim", &to_csv_str(&y.shape()));

        context.insert("ceil_mode", &self.ceil_mode);
        context.insert("kernel_shape", &to_csv_str(&self.kernel_shape));
        context.insert("pads", &to_csv_str(&self.pads));
        context.insert("strides", &to_csv_str(&self.strides));

        let compiled = tera
            .render("MaxPool", &mut context)
            .map_err(|e| e.to_string())?;

        Ok(compiled)
    }
}

#[cfg(test)]
mod test {
    use crate::graph::{Graph, OpType, Tensor};

    #[test]
    fn simple_pool() {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "X",
            Some(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
                -7.0, -8.0, -9.0,
            ]),
            vec![1, 2, 3, 3],
        );
        graph.new_tensor_f32("Y", None, vec![1, 2, 1, 1]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "my_maxpool",
                OpType::MaxPool {
                    ceil_mode: 0,
                    kernel_shape: vec![2, 2],
                    pads: vec![0, 0, 0, 0],
                    strides: vec![1, 1],
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(values, &Some(vec![5.0, -1.0]));
        }
    }
}
