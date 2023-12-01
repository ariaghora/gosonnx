use crate::errors::GosonnxError;
use crate::errors::GosonnxError::Error;
use crate::gpu::topo;
use crate::graph::Graph;
use crate::ops::OpType;

pub struct Optimizer {}

impl Optimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn optimize(&mut self, graph: Graph) -> Result<Graph, GosonnxError> {
        let mut graph = graph;
        graph.build_connections()?;

        let sorted = topo(&graph.op_map);

        // Activation function fusion
        for s in sorted {
            let curr = match graph.op_map.get(&s) {
                None => return Err(Error("Failed to optimize".to_string())),
                Some(op) => op,
            };

            // Skip fusing activation node if current node has no preceding node (i.e., current node
            // is likely not an activation node.
            let curr_name = curr.op_name.clone();
            if curr.prevs.len() < 1 {
                continue;
            }

            // Skip fusing activation node if previous node is not activable
            let prev_name = curr.prevs[0].clone();
            if let Some(prev) = graph.op_map.get(&prev_name) {
                if !prev.activable() {
                    continue;
                }
            }

            match &curr.op_type {
                OpType::Relu { .. } | OpType::Sigmoid { .. } => {
                    let extra_attrs = vec![("activation".to_string(), curr.op_type.to_string())];

                    let prev = graph.op_map.get_mut(&prev_name).unwrap();
                    prev.extra_attr = Some(extra_attrs);

                    graph.remove_node_and_connect_neighbors(&curr_name);
                }
                OpType::HardSigmoid { attr } => {
                    let mut extra_attrs =
                        vec![("activation".to_string(), curr.op_type.to_string())];
                    extra_attrs.append(&mut attr.attrs.clone());

                    let prev = graph.op_map.get_mut(&prev_name).unwrap();
                    prev.extra_attr = Some(extra_attrs);

                    graph.remove_node_and_connect_neighbors(&curr_name);
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
    use crate::ops::activation_op::ActivationOp;
    use crate::ops::conv::ConvOp;
    use crate::ops::conv_transpose::ConvTransposeOp;
    use crate::ops::gemm::GemmOp;
    use crate::ops::OpType;
    use crate::utils::vec_close;

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
                attr: ActivationOp::new(vec![]),
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

    #[test]
    fn test_gemm_hard_sigmoid_opt() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![-1.0, -1.0, 1.0, 1.0]), vec![2, 2])?;
        graph.new_tensor_f32("W", Some(vec![1.0, 1.0]), vec![2, 1])?;
        graph.new_tensor_f32("gemm_out", None, vec![2, 1])?;
        graph.new_tensor_f32("hs_out", None, vec![2, 1])?;
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
            vec!["hs_out"],
            "hs",
            OpType::HardSigmoid {
                attr: ActivationOp::new(vec![
                    attribute!("HardSigmoid_alpha", 0.5),
                    attribute!("HardSigmoid_beta", 0.6),
                ]),
            },
        )?;

        let mut graph = Optimizer::new().optimize(graph)?;
        graph.run()?;

        // relu is merged with gemm, so graph's op_map should be of length 1
        assert_eq!(graph.op_map.len(), 1);

        let out = graph.get_output("hs_out");
        if let Some(Tensor::F32 { values, .. }) = out {
            assert_eq!(values, &Some(vec![0.0, 1.0]));
        } else {
            panic!("Must be f32, found {:?}", out);
        }

        Ok(())
    }

    #[test]
    fn test_conv_and_bias_sigmoid_opt() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0,
                    -6.0, -7.0, -8.0, -9.0,
                ]),
                vec![1, 2, 3, 3],
            )
            .unwrap();
        graph
            .new_tensor_f32(
                "W",
                Some(vec![
                    0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0,
                    0.0,
                ]),
                vec![2, 2, 2, 2],
            )
            .unwrap();
        graph
            .new_tensor_f32("b", Some(vec![1.0, -1.0]), vec![2])
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]).unwrap();
        graph
            .new_tensor_f32("Y_sig", None, vec![1, 2, 2, 2])
            .unwrap();
        graph
            .new_op(
                vec!["X", "W", "b"],
                vec!["Y"],
                "my_conv",
                OpType::Conv {
                    attr: ConvOp::new(vec![1, 1], 1, vec![2, 2], vec![0, 0, 0, 0], vec![1, 1]),
                },
            )
            .unwrap();
        graph
            .new_op(
                vec!["Y"],
                vec!["Y_sig"],
                "sig",
                OpType::Sigmoid {
                    attr: ActivationOp::new(vec![]),
                },
            )
            .unwrap();

        let mut graph = Optimizer::new().optimize(graph).unwrap();
        graph.run().unwrap();

        assert_eq!(graph.op_map.len(), 1);

        let out = graph.get_output("Y_sig").unwrap();
        if let Tensor::F32 { values, .. } = out {
            let expected: Vec<f32> = vec![3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
                .iter()
                .map(|v| 1.0 / (1.0 + (-*v as f32).exp()))
                .collect();
            assert_eq!(values, &Some(expected));
        }
    }

    #[test]
    fn test_conv_and_bias_sig_opt() {
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
            .new_tensor_f32("Y_sig", None, vec![1, 1, 3, 3])
            .unwrap();
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
        graph
            .new_op(
                vec!["Y"],
                vec!["Y_sig"],
                "sig",
                OpType::Sigmoid {
                    attr: ActivationOp::new(vec![]),
                },
            )
            .unwrap();

        let mut graph = Optimizer::new().optimize(graph).unwrap();
        graph.run().unwrap();

        assert_eq!(graph.op_map.len(), 1);

        let out = graph.get_output("Y_sig").unwrap();
        if let Tensor::F32 { values, .. } = out {
            let expected: Vec<f32> = vec![
                0.6000, 0.9000, 0.9000, 1.1000, 2.5000, 2.1000, 1.4000, 2.9000, 2.1000,
            ]
            .iter()
            .map(|v| 1.0 / (1.0 + (-*v as f32).exp()))
            .collect();
            assert!(vec_close(values.as_ref().unwrap().clone(), expected));
        }
    }
}
