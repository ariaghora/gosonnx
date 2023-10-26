#[cfg(test)]
mod test {
    use std::error::Error;

    use crate::{
        attribute,
        graph::{Graph, Tensor},
        ops::{un_op::UnOpElementwise, OpType},
    };

    #[test]
    fn test_hard_sigmoid() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        let in_data = vec![-1., 0., 2., 4.];
        graph.new_tensor_f32("X", Some(in_data.clone()), vec![1, 4]);
        graph.new_tensor_f32("Y", None, vec![1, 4]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "hard_sigmoid",
                OpType::HardSigmoid {
                    attr: UnOpElementwise {
                        attrs: vec![attribute!("alpha", 0.5), attribute!("beta", 0.6)],
                    },
                },
            )
            .unwrap();

        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = result {
                assert_eq!(
                    values,
                    &Some(
                        in_data
                            .iter()
                            .map(|f| (f * 0.5 + 0.6).min(1.).max(0.))
                            .collect()
                    )
                );
            } else {
                panic!("Output should be Tensor::F32")
            }
        } else {
            panic!("Output Y not found")
        }

        Ok(())
    }
}
