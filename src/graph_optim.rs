use crate::errors::GosonnxError;
use crate::gpu::topo;
use crate::graph::{Graph, Op};
use std::collections::HashMap;

pub struct Optimizer {}

impl Optimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn optimize(&self, graph: &mut Graph) -> Result<(), GosonnxError> {
        let sorted = topo(&graph.op_map);
        let mut new_op_map: HashMap<String, Op> = HashMap::new();

        let mut head: i32 = 0;
        loop {
            // check subsequent op
            if head < sorted.len() as i32 {
                let curr = &graph.op_map[&sorted[head as usize]];
                let next = &graph.op_map[&sorted[head as usize + 1]];

                match (
                    curr.op_type.to_string().as_str(),
                    next.op_type.to_string().as_str(),
                ) {
                    ("Gemm", "Relu") => {
                        // TODO: fix linkage and set curr's activation
                        let mut fused = curr.clone();

                        head += 2;
                    }
                    _ => head += 1,
                }
            }
            if head >= sorted.len() as i32 - 1 {
                break;
            }
        }
        graph.op_map = new_op_map;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::graph::{Graph, Tensor};
    use crate::graph_optim::Optimizer;
    use crate::ops::gemm::GemmOp;
    use crate::ops::un_op::UnOpElementwise;
    use crate::ops::OpType;

    #[test]
    fn test_gemm_relu_opt() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![-1.0, -1.0, 1.0, 1.0]), vec![2, 2])?;
        graph.new_tensor_f32("W", Some(vec![1.0, 1.0]), vec![2, 1])?;
        graph.new_tensor_f32("gemm_out", None, vec![2, 1])?;
        graph.new_tensor_f32("relu_out", None, vec![2, 1])?;
        graph.new_op(
            vec!["X", "W"],
            vec!["gemm_out"],
            "gemm",
            OpType::Gemm {
                attr: GemmOp::new(None, None, None, None),
            },
        )?;
        graph.new_op(
            vec!["gemm_out"],
            vec!["relu_out"],
            "relu",
            OpType::Relu {
                attr: UnOpElementwise::new(vec![]),
            },
        )?;
        Optimizer::new().optimize(&mut graph)?;
        graph.run()?;

        let out = graph.get_output("relu_out");
        if let Some(Tensor::F32 { values, .. }) = out {
            assert_eq!(values, &Some(vec![0.0, 2.0]));

            // relu is merged with gemm, so graph's op_map should be of length 1
            assert_eq!(graph.op_map.len(), 1);
        } else {
            panic!("Must be f32, found {:?}", out);
        }

        Ok(())
    }
}
