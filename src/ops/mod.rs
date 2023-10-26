pub mod bin_op;
pub mod clip;
pub mod conv;
pub mod flatten;
pub mod gemm;
mod hard_sigmoid;
pub mod maxpool;
pub mod relu;
mod sigmoid;
mod un_op;

use self::{
    bin_op::BinOpElementwise, conv::ConvOp, flatten::FlattenOp, gemm::GemmOp, maxpool::MaxPoolOp,
    un_op::UnOpElementwise,
};
use crate::{
    attribute,
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
    Add { attr: BinOpElementwise },
    Clip { attr: UnOpElementwise },
    Conv { attr: ConvOp },
    Div { attr: BinOpElementwise },
    Flatten { attr: FlattenOp },
    Gemm { attr: GemmOp },
    HardSigmoid { attr: UnOpElementwise },
    MaxPool { attr: MaxPoolOp },
    Mul { attr: BinOpElementwise },
    Relu { attr: UnOpElementwise },
    Sigmoid { attr: UnOpElementwise },
    //
    // Default OpType when creating a node
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
            OpType::Add { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Clip { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Conv { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Div { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Flatten { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Gemm { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::HardSigmoid { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::MaxPool { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Mul { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Relu { attr } => self._compile(attr, shader_source, op, graph)?,
            OpType::Sigmoid { attr } => self._compile(attr, shader_source, op, graph)?,
            //
            // Default OpType when creating a node
            OpType::Unknown => return Err(format!("Op `{:?}` is unsupported yet", op.op_type)),
        };
        Ok((compiled, wg))
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Add { .. } => write!(f, "Add"),
            OpType::Clip { .. } => write!(f, "Clip"),
            OpType::Conv { .. } => write!(f, "Conv"),
            OpType::Div { .. } => write!(f, "Div"),
            OpType::Flatten { .. } => write!(f, "Flatten"),
            OpType::Gemm { .. } => write!(f, "Gemm"),
            OpType::HardSigmoid { .. } => write!(f, "HardSigmoid"),
            OpType::MaxPool { .. } => write!(f, "MaxPool"),
            OpType::Mul { .. } => write!(f, "Mul"),
            OpType::Relu { .. } => write!(f, "Relu"),
            OpType::Sigmoid { .. } => write!(f, "Sigmoid"),
            OpType::Unknown => write!(f, "Unknown"),
        }
    }
}

impl OpType {
    pub fn from_node_proto(node_proto: &NodeProto) -> Result<Self, String> {
        // attribute!()
        match node_proto.get_op_type() {
            "Add" => Ok(Self::Add {
                attr: BinOpElementwise {},
            }),
            "Clip" => Ok(Self::Clip {
                attr: UnOpElementwise::new(vec![
                    attribute!("min_val", get_attr_f(node_proto, "min")),
                    attribute!("max_val", get_attr_f(node_proto, "max")),
                ]),
            }),
            "Conv" => Ok(Self::Conv {
                attr: ConvOp::new(
                    get_attr_ints(node_proto, "dilations").unwrap(),
                    get_attr_i(node_proto, "group").unwrap(),
                    get_attr_ints(node_proto, "kernel_shape").unwrap(),
                    get_attr_ints(node_proto, "pads").unwrap(),
                    get_attr_ints(node_proto, "strides").unwrap(),
                ),
            }),
            "Div" => Ok(Self::Div {
                attr: BinOpElementwise {},
            }),
            "Gemm" => Ok(Self::Gemm {
                attr: GemmOp::new(
                    get_attr_f(node_proto, "alpha").unwrap(),
                    get_attr_f(node_proto, "beta").unwrap(),
                    get_attr_i(node_proto, "transA").unwrap(),
                    get_attr_i(node_proto, "transB").unwrap(),
                ),
            }),
            "HardSigmoid" => Ok(Self::HardSigmoid {
                attr: UnOpElementwise::new(vec![attribute!(
                    "alpha",
                    get_attr_f(node_proto, "alpha")
                )]),
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
            "Mul" => Ok(Self::Mul {
                attr: BinOpElementwise {},
            }),
            "Relu" => Ok(Self::Relu {
                attr: UnOpElementwise::new(vec![]),
            }),
            "Sigmoid" => Ok(Self::Sigmoid {
                attr: UnOpElementwise::new(vec![]),
            }),
            _ => Err(format!(
                "ONNX op type `{}` is not supported yet",
                node_proto.get_op_type()
            )),
        }
    }
}

impl<'gr, 'gpu> OpType {
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

pub fn to_csv_str<T: ToString>(vals: &Vec<T>) -> String {
    let res: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    res.join(",")
}

pub fn compile_unary(
    op: &Op,
    attr: Option<Vec<(&str, String)>>,
    _shader_source: &str,
    _graph: &Graph,
) -> Result<String, String> {
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

    if let Some(attributes) = attr {
        for (k, v) in attributes {
            context.insert(k, &v);
        }
    }

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

pub fn compile_binary(op: &Op, _shader_source: &str, _graph: &Graph) -> Result<String, String> {
    let base_shader_source = SHADER_DIR
        .get_file("_binary_elementwise.glsl")
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

    let input_1 = &_graph.tensor_map[&op.inputs[0]];
    let input_2 = &_graph.tensor_map[&op.inputs[1]];
    let output = &_graph.tensor_map[&op.outputs[0]];
    context.insert("input_1_type", &input_1.type_glsl());
    context.insert("input_2_type", &input_2.type_glsl());
    context.insert("output_type", &output.type_glsl());

    tera.add_raw_templates(vec![
        ("_binary_elementwise", base_shader_source),
        (&op.op_type.to_string(), unary_shader_source),
    ])
    .map_err(|e| e.to_string())?;

    let compiled = tera
        .render(&op.op_type.to_string(), &mut context)
        .map_err(|e| e.to_string())?;
    Ok(compiled)
}
