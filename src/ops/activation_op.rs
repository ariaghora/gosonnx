use std::fmt::Debug;

use serde::{Serialize, Serializer};

use crate::errors::GosonnxError;
use crate::{
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::{Compile, ShaderTemplate};

#[derive(Clone)]
pub struct ActivationOp {
    pub attrs: Vec<(String, String)>,
}

impl Serialize for ActivationOp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("null")
    }
}

impl Debug for ActivationOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MyStruct {{ data: {} }}", "null")
    }
}

impl ActivationOp {
    pub fn new(attrs: Vec<(String, String)>) -> Self {
        Self { attrs }
    }
}

impl Compile for &ActivationOp {
    fn compile(
        &self,
        op: &Op,
        shader_templ: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError> {
        for (k, v) in self.attrs.iter() {
            shader_templ.push_attr(k, v);
        }

        let input = &graph.tensor_map[&op.inputs[0]];
        let output = &graph.tensor_map[&op.outputs[0]];
        shader_templ.push_attr("input_type", &input.type_glsl());
        shader_templ.push_attr("output_type", &output.type_glsl());

        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let local_size_x = 256;
        let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
        let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }

    fn activable(&self) -> bool {
        // normally after activation function (this node) there won't be any activation function
        // anymore, so this node's activable should return false
        false
    }
}
