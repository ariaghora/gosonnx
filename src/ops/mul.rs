#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::{bin_op::BinOpElementwise, OpType},
    };

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
                    attr: BinOpElementwise {},
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
}
