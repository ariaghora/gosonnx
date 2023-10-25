use serde::Serialize;

use crate::{
    gpu::SHADER_DIR,
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::Compile;

#[derive(Debug, Serialize)]
pub struct ReluOp {}

impl ReluOp {}

impl Compile for ReluOp {
    fn compile(&self, _op: &Op, _shader_source: &str, _graph: &Graph) -> Result<String, String> {
        let base_shader_source = SHADER_DIR
            .get_file("_unary_elementwise.glsl")
            .unwrap()
            .contents_utf8()
            .unwrap();
        let unary_shader_source = SHADER_DIR
            .get_file("Relu.glsl")
            .unwrap()
            .contents_utf8()
            .unwrap();
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        context.insert("input_type", "float");
        context.insert("output_type", "float");

        tera.add_raw_templates(vec![
            ("_unary_elementwise", base_shader_source),
            ("Relu", unary_shader_source),
        ])
        .map_err(|e| e.to_string())?;

        let compiled = tera
            .render("Relu", &mut context)
            .map_err(|e| e.to_string())?;
        Ok(compiled)
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let local_size_x = 256;
        let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
        let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use crate::graph::{Graph, OpType, Tensor};

    #[test]
    fn simple_relu() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![0.5, -1.0, 2.0]), vec![1, 3]);
        graph.new_tensor_f32("Y", None, vec![1, 3]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "my_relu_1",
                OpType::Relu {
                    attr: super::ReluOp {},
                },
            )
            .unwrap();

        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, shape } = result {
                assert_eq!(values, &Some(vec![0.5, 0.0, 2.0]));
                assert_eq!(shape, &vec![1, 3]);
            } else {
                panic!("Output should be Tensor::F32")
            }
        } else {
            panic!("Output Y not found")
        }

        Ok(())
    }
}
