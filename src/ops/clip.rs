use crate::errors::GosonnxError;
use crate::errors::GosonnxError::InvalidInputNo;
use crate::graph::{Graph, Op};
use crate::ops::{Compile, ShaderTemplate};
use crate::utils::tensor_len;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ClipOp {}

impl ClipOp {
    pub fn new() -> Self {
        Self {}
    }
}

impl Compile for &ClipOp {
    fn compile(
        &self,
        op: &Op,
        shader_templ: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError> {
        // TODO: handle dynamic inputs
        if op.inputs.len() != 3 {
            return Err(InvalidInputNo {
                expected: 3,
                found: op.inputs.len(),
            });
        }

        let input = &graph.tensor_map[&op.inputs[0]];
        let min_val = &graph.tensor_map[&op.inputs[1]];
        let max_val = &graph.tensor_map[&op.inputs[2]];
        let output = &graph.tensor_map[&op.outputs[0]];
        shader_templ.push_attr("input_type", &input.type_glsl());
        shader_templ.push_attr("min_val_type", &min_val.type_glsl());
        shader_templ.push_attr("max_val_type", &max_val.type_glsl());
        shader_templ.push_attr("output_type", &output.type_glsl());
        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let local_size_x = 256;
        let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
        let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}

#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::ops::clip::ClipOp;
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    #[test]
    fn simple_clip() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![-5., 3., 5.]), vec![1, 3])?;
        graph.new_tensor_f32("min", Some(vec![-3.0]), vec![])?;
        graph.new_tensor_f32("max", Some(vec![3.0]), vec![])?;
        graph.new_tensor_f32("Y", None, vec![1, 3])?;
        graph.new_op(
            vec!["X", "min", "max"],
            vec!["Y"],
            "my_clip",
            OpType::Clip { attr: ClipOp {} },
        )?;

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
}
