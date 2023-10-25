pub mod conv;
pub mod flatten;
pub mod gemm;
pub mod maxpool;
pub mod relu;

use self::{conv::ConvOp, flatten::FlattenOp, gemm::GemmOp, maxpool::MaxPoolOp, relu::ReluOp};
use crate::{
    gpu::SHADER_DIR,
    graph::{Graph, Op},
    onnx::onnx::NodeProto,
    utils::{get_attr_f, get_attr_i, get_attr_ints},
};
use serde::Serialize;
use std::fmt;

pub trait Compile {
    fn compile(&self, op: &Op, shader_source: &str, graph: &Graph) -> Result<String, String>;
    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3];
}

#[derive(Debug, Serialize)]
pub enum OpType {
    Conv { attr: ConvOp },
    Flatten { attr: FlattenOp },
    Gemm { attr: GemmOp },
    MaxPool { attr: MaxPoolOp },
    Relu { attr: ReluOp },
    Unknown,
}

impl<'gr, 'gpu> OpType {
    pub fn compile(
        &self,
        shader_source: &str,
        op: &'gr Op,
        graph: &'gr Graph,
    ) -> Result<(String, [u32; 3]), String> {
        let (compiled, wg) = match self {
            OpType::Conv { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Flatten { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Gemm { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::MaxPool { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Relu { attr } => self._compile(attr, shader_source, op, graph)?,
            _ => return Err(format!("Op `{:?}` is unsupported yet", op.op_type)),
        };
        Ok((compiled, wg))
    }

    fn _compile<Compilable: Compile>(
        &self,
        attr: Compilable,
        shader_source: &str,
        op: &'gr Op,
        graph: &'gr Graph,
    ) -> Result<(String, [u32; 3]), String> {
        let compiled = attr.compile(op, shader_source, graph)?;
        let wg = attr.compute_workgroup_size(op, graph);
        Ok((compiled, wg))
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Conv { .. } => write!(f, "Conv"),
            OpType::Flatten { .. } => write!(f, "Flatten"),
            OpType::Gemm { .. } => write!(f, "Gemm"),
            OpType::MaxPool { .. } => write!(f, "MaxPool"),
            OpType::Relu { .. } => write!(f, "Relu"),
            OpType::Unknown => write!(f, "Unknown"),
        }
    }
}

impl OpType {
    pub fn from_node_proto(node_proto: &NodeProto) -> Result<Self, String> {
        match node_proto.get_op_type() {
            "Conv" => Ok(Self::Conv {
                attr: ConvOp::new(
                    get_attr_ints(node_proto, "dilations").unwrap(),
                    get_attr_i(node_proto, "group").unwrap(),
                    get_attr_ints(node_proto, "kernel_shape").unwrap(),
                    get_attr_ints(node_proto, "pads").unwrap(),
                    get_attr_ints(node_proto, "strides").unwrap(),
                ),
            }),
            "Gemm" => Ok(Self::Gemm {
                attr: GemmOp::new(
                    get_attr_f(node_proto, "alpha").unwrap(),
                    get_attr_f(node_proto, "beta").unwrap(),
                    get_attr_i(node_proto, "transA").unwrap(),
                    get_attr_i(node_proto, "transB").unwrap(),
                ),
            }),
            "Flatten" => Ok(Self::Flatten {
                attr: FlattenOp::new(get_attr_i(node_proto, "axis").unwrap()),
            }),
            "MaxPool" => Ok(Self::MaxPool {
                attr: MaxPoolOp::new(
                    get_attr_i(node_proto, "ceil_mode").unwrap(),
                    get_attr_ints(node_proto, "kernel_shape").unwrap(),
                    get_attr_ints(node_proto, "pads").unwrap(),
                    get_attr_ints(node_proto, "strides").unwrap(),
                ),
            }),
            "Relu" => Ok(Self::Relu { attr: ReluOp {} }),
            _ => Err(format!(
                "ONNX op type {} is not supported yet",
                node_proto.get_op_type()
            )),
        }
    }
}

pub fn to_csv_str<T: ToString>(vals: &Vec<T>) -> String {
    let res: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    res.join(",")
}

pub fn compile_unary(op: &Op, _shader_source: &str, _graph: &Graph) -> Result<String, String> {
    let base_shader_source = SHADER_DIR
        .get_file("_unary_elementwise.glsl")
        .unwrap()
        .contents_utf8()
        .unwrap();
    let unary_shader_source = SHADER_DIR
        .get_file(&format!("{}.glsl", op.op_type.to_string()))
        .unwrap()
        .contents_utf8()
        .unwrap();
    let mut tera = tera::Tera::default();
    let mut context = tera::Context::new();

    let input = &_graph.tensor_map[&op.inputs[0]];
    let output = &_graph.tensor_map[&op.outputs[0]];
    context.insert("input_type", &input.type_glsl());
    context.insert("output_type", &output.type_glsl());

    tera.add_raw_templates(vec![
        ("_unary_elementwise", base_shader_source),
        (&op.op_type.to_string(), unary_shader_source),
    ])
    .map_err(|e| e.to_string())?;

    let compiled = tera
        .render(&op.op_type.to_string(), &mut context)
        .map_err(|e| e.to_string())?;
    Ok(compiled)
}
