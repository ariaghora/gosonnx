#[cfg(test)]
mod test {
    use std::error::Error;

    use uuid::ClockSequence;

    use crate::{
        graph::{Graph, Tensor},
        ops::{un_op::UnOpElementwise, OpType},
        utils::vec_close,
    };

    #[test]
    fn test_relu() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        let in_data = vec![0.5, -1.0, 2.0];
        graph.new_tensor_f32("X", Some(in_data.clone()), vec![1, 3]);
        graph.new_tensor_f32("Y", None, vec![1, 3]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "sigmoid",
                OpType::Sigmoid {
                    attr: UnOpElementwise { attrs: vec![] },
                },
            )
            .unwrap();

        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, shape } = result {
                assert!(vec_close(
                    values.as_ref().unwrap().clone(),
                    in_data.iter().map(|f| 1. / (1. + (-f).exp())).collect()
                ));

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
