use serde::Serialize;

use crate::{
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::{compile_unary, Compile};

#[derive(Debug, Serialize)]
pub struct ReluOp;

impl Compile for &ReluOp {
    fn compile(&self, op: &Op, shader_source: &str, graph: &Graph) -> Result<String, String> {
        compile_unary(op, shader_source, graph)
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

    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

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
