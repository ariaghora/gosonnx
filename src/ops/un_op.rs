use std::fmt::Debug;

use serde::Serialize;

use crate::{
    graph::{Graph, Op},
    utils::tensor_len,
};

use super::{compile_unary, Compile};

#[derive(Debug, Serialize)]
pub struct UnOpElementwise {
    pub attrs: Vec<(String, String)>,
}

impl UnOpElementwise {
    pub fn new(attrs: Vec<(String, String)>) -> Self {
        Self { attrs }
    }
}

impl Compile for &UnOpElementwise {
    fn compile(&self, op: &Op, shader_source: &str, graph: &Graph) -> Result<String, String> {
        compile_unary(op, None, shader_source, graph)
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let local_size_x = 256;
        let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
        let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}
