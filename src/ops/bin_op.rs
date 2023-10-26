use super::{compile_binary, Compile};
use crate::utils::tensor_len;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct BinOpElementwise;

impl Compile for &BinOpElementwise {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_source: &str,
        graph: &crate::graph::Graph,
    ) -> Result<String, String> {
        compile_binary(op, shader_source, graph)
    }

    fn compute_workgroup_size(
        &self,
        op: &crate::graph::Op,
        graph: &crate::graph::Graph,
    ) -> [u32; 3] {
        let local_size_x = 256;
        let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
        let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    #[test]
    fn add_no_bcast() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some((0..9).map(|v| v as f32).collect()), vec![3, 3]);
        graph.new_tensor_f32("B", Some((0..9).map(|v| v as f32).collect()), vec![3, 3]);
        graph.new_tensor_f32("Y", None, vec![3, 3]);
        graph
            .new_op(
                vec!["A", "B"],
                vec!["Y"],
                "add",
                OpType::Add {
                    attr: super::BinOpElementwise {},
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some((0..9).map(|v| (v + v) as f32).collect::<Vec<f32>>())
                )
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }

    #[test]
    fn mul_no_bcast() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some((0..9).map(|v| v as f32).collect()), vec![3, 3]);
        graph.new_tensor_f32("B", Some((0..9).map(|v| v as f32).collect()), vec![3, 3]);
        graph.new_tensor_f32("Y", None, vec![3, 3]);
        graph
            .new_op(
                vec!["A", "B"],
                vec!["Y"],
                "add",
                OpType::Mul {
                    attr: super::BinOpElementwise {},
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some((0..9).map(|v| (v * v) as f32).collect::<Vec<f32>>())
                )
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }

    #[test]
    fn div_no_bcast() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some((1..10).map(|v| v as f32).collect()), vec![3, 3]);
        graph.new_tensor_f32("B", Some((1..10).map(|_| 2.0).collect()), vec![3, 3]);
        graph.new_tensor_f32("Y", None, vec![3, 3]);
        graph
            .new_op(
                vec!["A", "B"],
                vec!["Y"],
                "add",
                OpType::Div {
                    attr: super::BinOpElementwise {},
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some((1..10).map(|v| (v as f32) / 2.0).collect::<Vec<f32>>())
                )
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }
}
