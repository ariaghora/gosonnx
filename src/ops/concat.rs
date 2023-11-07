use serde::Serialize;

use crate::{graph::Tensor, utils::tensor_len};

use super::{bin_op::shape_to_strides, to_csv_str, Compile};

#[derive(Debug, Serialize)]
pub struct ConcatOp {
    axis: i64,
}

impl ConcatOp {
    pub fn new(axis: i64) -> Self {
        Self { axis }
    }
}

#[derive(Serialize)]
struct InputInfo {
    dtype: String,
    n_dim: usize,
    shape_csv: String,
    strides_csv: String,
}

impl Compile for &ConcatOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut super::ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), String> {
        let inputs: Vec<&Tensor> = op.inputs.iter().map(|v| &graph.tensor_map[v]).collect();
        let output = &graph.tensor_map[&op.outputs[0]];

        let input_info_arr: Vec<InputInfo> = inputs
            .iter()
            .map(|v| InputInfo {
                dtype: v.type_glsl(),
                n_dim: v.shape().len(),
                shape_csv: to_csv_str(&v.shape()),
                strides_csv: to_csv_str(&shape_to_strides(&v.shape())),
            })
            .collect();
        shader_templ.push_attr("input_info_arr", &input_info_arr);
        shader_templ.push_attr("output_n_dim", &output.shape().len());
        shader_templ.push_attr("output_shape_csv", &to_csv_str(&output.shape()));
        shader_templ.push_attr(
            "output_strides_csv",
            &to_csv_str(&shape_to_strides(&output.shape())),
        );

        shader_templ.push_attr("concat_axis", &self.axis);
        shader_templ.push_attr("n_inputs", &inputs.len());

        shader_templ.push_attr("output_n_elements", &tensor_len(&output).unwrap());

        // calculate output stride
        let mut output_stride = 1;
        for sz in &output.shape()[self.axis as usize + 1..] {
            output_stride *= *sz;
        }
        shader_templ.push_attr("output_stride", &output_stride);
        shader_templ.push_attr("output_dtype", &output.type_glsl());
        shader_templ.push_attr("output_binding_no", &inputs.len());

        let concat_axis_size = inputs
            .iter()
            .fold(0, |acc, t| acc + t.shape()[self.axis as usize]);
        shader_templ.push_attr("concat_axis_size", &concat_axis_size);

        Ok(())
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

    use super::ConcatOp;

    #[test]
    fn test_concat_0() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("t1", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("t2", Some(vec![5.0, 6.0, 7.0, 8.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("t3", Some(vec![-1.0, -2.0, -3.0, -4.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("y", None, vec![3, 1, 2, 2]);
        graph
            .new_op(
                vec!["t1", "t2", "t3"],
                vec!["y"],
                "concat",
                OpType::Concat {
                    attr: ConcatOp { axis: 0 },
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0,
                ])
            );
        }
    }

    #[test]
    fn test_concat_2() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("t1", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("t2", Some(vec![5.0, 6.0, 7.0, 8.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("t3", Some(vec![-1.0, -2.0, -3.0, -4.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("y", None, vec![1, 1, 6, 2]);
        graph
            .new_op(
                vec!["t1", "t2", "t3"],
                vec!["y"],
                "concat",
                OpType::Concat {
                    attr: ConcatOp { axis: 2 },
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0,
                ])
            );
        }
    }

    #[test]
    fn test_concat_3() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("t1", Some(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("t2", Some(vec![5.0, 6.0, 7.0, 8.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("t3", Some(vec![-1.0, -2.0, -3.0, -4.0]), vec![1, 1, 2, 2]);
        graph.new_tensor_f32("y", None, vec![1, 1, 2, 6]);
        graph
            .new_op(
                vec!["t1", "t2", "t3"],
                vec!["y"],
                "concat",
                OpType::Concat {
                    attr: ConcatOp { axis: 3 },
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1.0, 2.0, 5.0, 6.0, -1.0, -2.0, 3.0, 4.0, 7.0, 8.0, -3.0, -4.0
                ])
            );
        }
    }

    #[test]
    fn test_concat_2_2x1x2() {
        let mut graph = Graph::new();
        graph.new_tensor_f32("t1", Some(vec![1.0, 1.0, 1.0, 1.0]), vec![2, 1, 2]);
        graph.new_tensor_f32("t2", Some(vec![2.0, 2.0, 2.0, 2.0]), vec![2, 1, 2]);
        graph.new_tensor_f32("t3", Some(vec![3.0, 3.0, 3.0, 3.0]), vec![2, 1, 2]);
        graph.new_tensor_f32("y", None, vec![2, 1, 6]);
        graph
            .new_op(
                vec!["t1", "t2", "t3"],
                vec!["y"],
                "concat",
                OpType::Concat {
                    attr: ConcatOp { axis: 2 },
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0
                ])
            );
        }
    }
}