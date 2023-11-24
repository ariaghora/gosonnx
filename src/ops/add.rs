#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::{
        graph::{Graph, Tensor},
        ops::{bin_op::BinOpElementwise, OpType},
    };

    #[test]
    fn add_no_bcast() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some((0..9).map(|v| v as f32).collect()), vec![3, 3])?;
        graph.new_tensor_f32("B", Some((0..9).map(|v| v as f32).collect()), vec![3, 3])?;
        graph.new_tensor_f32("Y", None, vec![3, 3])?;
        graph
            .new_op(
                vec!["A", "B"],
                vec!["Y"],
                "add",
                OpType::Add {
                    attr: BinOpElementwise {},
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
        Ok(())
    }

    #[test]
    fn add_bcast_scalar() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some((0..9).map(|v| v as f32).collect()), vec![3, 3])?;
        graph.new_tensor_f32("B", Some(vec![10.0]), vec![1])?;
        graph.new_tensor_f32("Y", None, vec![3, 3])?;
        graph
            .new_op(
                vec!["A", "B"],
                vec!["Y"],
                "add",
                OpType::Add {
                    attr: BinOpElementwise {},
                },
            )
            .unwrap();
        graph.run()?;
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some((0..9).map(|v| ((v as f32) + 10.0)).collect::<Vec<f32>>())
                )
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }

        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some(vec![5.0]), vec![1])?;
        graph.new_tensor_f32("B", Some((0..9).map(|v| v as f32).collect()), vec![3, 3])?;
        graph.new_tensor_f32("Y", None, vec![3, 3])?;
        graph.new_op(
            vec!["A", "B"],
            vec!["Y"],
            "add",
            OpType::Add {
                attr: BinOpElementwise {},
            },
        )?;
        graph.run().unwrap();
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some((0..9).map(|v| ((v as f32) + 5.0)).collect::<Vec<f32>>())
                )
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
        Ok(())
    }

    #[test]
    fn add_bcast_tensor() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some(vec![1., 1., 2., 2.]), vec![2, 2])?;
        graph.new_tensor_f32("B", Some(vec![10.0, 20.]), vec![2, 1])?;
        graph.new_tensor_f32("Y", None, vec![2, 2])?;
        graph.new_op(
            vec!["A", "B"],
            vec!["Y"],
            "add",
            OpType::Add {
                attr: BinOpElementwise {},
            },
        )?;
        graph.run()?;
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![11., 11., 22., 22.]))
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }

        // More channels
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "A",
            Some(vec![1., 1., 2., 2., 1., 1., 2., 2.]),
            vec![2, 2, 2],
        )?;
        graph.new_tensor_f32("B", Some(vec![10.0, 20.]), vec![2, 1, 1])?;
        graph.new_tensor_f32("Y", None, vec![2, 2, 2])?;
        graph.new_op(
            vec!["A", "B"],
            vec!["Y"],
            "add",
            OpType::Add {
                attr: BinOpElementwise {},
            },
        )?;
        graph.run()?;
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![11., 11., 12., 12., 21., 21., 22., 22.]))
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
        Ok(())
    }

    #[test]
    fn add_bcast_tensor_bidireactional() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("A", Some(vec![1., 2., 3.]), vec![3, 1])?;
        graph.new_tensor_f32("B", Some(vec![1., 2., 3.]), vec![1, 3])?;
        graph.new_tensor_f32("Y", None, vec![3, 3])?;
        graph.new_op(
            vec!["A", "B"],
            vec!["Y"],
            "add",
            OpType::Add {
                attr: BinOpElementwise {},
            },
        )?;
        graph.run()?;
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(values, &Some(vec![2., 3., 4., 3., 4., 5., 4., 5., 6.]))
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
        Ok(())
    }
}
