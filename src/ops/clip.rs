use serde::Serialize;

use crate::{
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::{compile_unary, Compile};

#[cfg(test)]
mod test {
    use std::error::Error;

    use crate::{
        attribute,
        graph::{Graph, Tensor},
        ops::{un_op::UnOpElementwise, OpType},
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
                    attr: UnOpElementwise::new(vec![
                        attribute!("min_val", -3),
                        attribute!("max_val", 3),
                    ]),
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
                    attr: UnOpElementwise::new(vec![attribute!("max_val", 3)]),
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
