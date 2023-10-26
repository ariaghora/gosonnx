use serde::Serialize;

use crate::{
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::{compile_unary, Compile};

#[derive(Debug, Serialize)]
pub struct ClipOp {
    min: Option<f32>,
    max: Option<f32>,
}

impl ClipOp {
    pub fn new(min: Option<f32>, max: Option<f32>) -> Self {
        Self { min, max }
    }
}

impl Compile for &ClipOp {
    fn compile(&self, op: &Op, shader_source: &str, graph: &Graph) -> Result<String, String> {
        let mut attrs = vec![];
        if let Some(min) = self.min {
            attrs.push(("use_min", "1".to_string()));
            attrs.push(("min_val", min.to_string()));
        } else {
            attrs.push(("min_val", "".to_string()));
        }

        if let Some(max) = self.max {
            attrs.push(("use_max", "1".to_string()));
            attrs.push(("max_val", max.to_string()));
        } else {
            attrs.push(("min_val", "".to_string()));
        }

        let compiled = compile_unary(op, Some(attrs), shader_source, graph);
        compiled
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
    fn simple_clip() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![-5., 3., 5.]), vec![1, 3]);
        graph.new_tensor_f32("Y", None, vec![1, 3]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "my_clip",
                OpType::Clip {
                    attr: super::ClipOp::new(Some(-3.), Some(3.)),
                },
            )
            .unwrap();

        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, shape } = result {
                assert_eq!(values, &Some(vec![-3., 3.0, 3.0]));
                assert_eq!(shape, &vec![1, 3]);
            } else {
                panic!("Output should be Tensor::F32")
            }
        } else {
            panic!("Output Y not found")
        }

        Ok(())
    }

    #[test]
    fn simple_clip_no_min() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![-5., 3., 5.]), vec![1, 3]);
        graph.new_tensor_f32("Y", None, vec![1, 3]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "my_clip",
                OpType::Clip {
                    attr: super::ClipOp::new(None, Some(3.)),
                },
            )
            .unwrap();

        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, shape } = result {
                assert_eq!(values, &Some(vec![-5., 3.0, 3.0]));
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
