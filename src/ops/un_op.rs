use std::fmt::Debug;

use serde::{Serialize, Serializer};

use crate::{
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::{Compile, ShaderTemplate};

pub struct UnOpElementwise {
    pub attrs: Vec<(String, Box<dyn Debug>)>,
}

impl Serialize for UnOpElementwise {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("null")
    }
}

impl Debug for UnOpElementwise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MyStruct {{ data: {} }}", "null")
    }
}

impl UnOpElementwise {
    pub fn new(attrs: Vec<(String, Box<dyn Debug>)>) -> Self {
        Self { attrs }
    }
}

impl Compile for &UnOpElementwise {
    fn compile(
        &self,
        op: &Op,
        shader_templ: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), String> {
        for (k, v) in self.attrs.iter() {
            shader_templ.push_attr(k, &format!("{:?}", v));
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
}
