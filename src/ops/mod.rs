use crate::graph::{Graph, Op};

pub mod conv;
pub mod flatten;
pub mod gemm;
pub mod maxpool;
pub mod relu;

pub trait Compile {
    fn compile(&self, op: &Op, shader_source: &str, graph: &Graph) -> Result<String, String>;
    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3];
}

pub fn to_csv_str<T: ToString>(vals: &Vec<T>) -> String {
    let res: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    res.join(",")
}
