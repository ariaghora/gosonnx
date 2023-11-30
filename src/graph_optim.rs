use crate::errors::GosonnxError;
use crate::errors::GosonnxError::Error;
use crate::gpu::topo;
use crate::graph::{Graph, Op};
use crate::ops::OpType;
use protobuf::reflect::ProtobufValue;
use std::collections::HashMap;

pub struct Optimizer {}

impl Optimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn optimize(&self, graph: Graph) -> Result<Graph, GosonnxError> {
        let mut graph = graph;
        graph.compile()?;

        let sorted = topo(&graph.op_map);

        // Activation function fusion
        for s in sorted {
            let curr = match graph.op_map.get(&s) {
                None => return Err(Error("Failed to optimize".to_string())),
                Some(op) => op,
            };

            match curr.op_type {
                OpType::Relu { .. } => {
                    let extra_attrs = vec![("activation".to_string(), "Relu".to_string())];
                    // Remove current activation node and correct the connection of previous
                    // node
                    let curr_next = curr.nexts.clone();
                    let curr_outputs = curr.outputs.clone();
                    let curr_name = curr.op_name.clone();
                    let prev = graph.op_map.get_mut(&curr.prevs[0].clone()).unwrap();
                    prev.nexts = curr_next;
                    prev.outputs = curr_outputs;
                    prev.extra_attr = Some(extra_attrs);

                    graph.op_map.remove(&curr_name);
                }
                _ => {}
            }
        }

        Ok(graph)
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

        let mut graph = Optimizer::new().optimize(graph)?;
        graph.run()?;

        // relu is merged with gemm, so graph's op_map should be of length 1
        assert_eq!(graph.op_map.len(), 1);

        let out = graph.get_output("relu_out");
        if let Some(Tensor::F32 { values, .. }) = out {
            assert_eq!(values, &Some(vec![0.0, 2.0]));
        } else {
            panic!("Must be f32, found {:?}", out);
        }

        Ok(())
    }
}
