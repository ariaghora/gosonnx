#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::{
        graph::{Graph, Tensor},
        ops::{un_op::UnOpElementwise, OpType},
    };

    #[test]
    fn simple_relu() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![0.5, -1.0, 2.0]), vec![1, 3])?;
        graph.new_tensor_f32("Y", None, vec![1, 3])?;
        graph.new_op(
            vec!["X"],
            vec!["Y"],
            "my_relu_1",
            OpType::Relu {
                attr: UnOpElementwise::new(vec![]),
            },
        )?;

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
