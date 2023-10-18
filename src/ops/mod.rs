use crate::{Graph, Op};

pub mod gemm;

pub trait Compile {
    fn compile(&self, op: &Op, shader_source: &str, graph: &Graph) -> Result<String, String>;
}
