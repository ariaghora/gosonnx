use std::fmt::{self, Debug, Display};

use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::{Error, ShaderCompileError, UnsupportedONNXOps};
use crate::ops::clip::ClipOp;
use crate::{
    attribute, define_ops,
    gpu::SHADER_DIR,
    graph::{Graph, Op},
    onnx::onnx::NodeProto,
    utils::{get_attr_f, get_attr_i, get_attr_ints, get_attr_string},
};

use self::{
    average_pool::AveragePoolOp, batch_normalization::BatchNormalizationOp,
    bin_op::BinOpElementwise, concat::ConcatOp, conv::ConvOp, conv_transpose::ConvTransposeOp,
    flatten::FlattenOp, gemm::GemmOp, global_average_pool::GlobalAveragePoolOp, maxpool::MaxPoolOp,
    resize::ResizeOp, un_op::UnOpElementwise,
};

pub mod add;
pub mod average_pool;
mod batch_normalization;
pub mod bin_op;
pub mod clip;
pub mod concat;
pub mod conv;
pub mod conv_transpose;
pub mod flatten;
pub mod gemm;
pub mod global_average_pool;
pub mod hard_sigmoid;
pub mod maxpool;
pub mod mul;
pub mod relu;
pub mod resize;
pub mod sigmoid;
pub mod un_op;

define_ops!(
    Add { BinOpElementwise },
    AveragePool { AveragePoolOp },
    BatchNormalization {
        BatchNormalizationOp
    },
    Clip { ClipOp },
    Concat { ConcatOp },
    Conv { ConvOp },
    ConvTranspose { ConvTransposeOp },
    Div { BinOpElementwise },
    Flatten { FlattenOp },
    Gemm { GemmOp },
    GlobalAveragePool {
        GlobalAveragePoolOp
    },
    HardSigmoid { UnOpElementwise },
    MaxPool { MaxPoolOp },
    Mul { BinOpElementwise },
    Relu { UnOpElementwise },
    Resize { ResizeOp },
    Sigmoid { UnOpElementwise }
);

impl OpType {
    pub fn from_node_proto(node_proto: &NodeProto) -> Result<Self, GosonnxError> {
        match node_proto.get_op_type() {
            "Add" => Ok(Self::Add {
                attr: BinOpElementwise {},
            }),
            "AveragePool" => Ok(Self::AveragePool {
                attr: AveragePoolOp::new(
                    get_attr_string(node_proto, "auto_pad"),
                    get_attr_i(node_proto, "ceil_mode"),
                    get_attr_ints(node_proto, "dilations"),
                    get_attr_ints(node_proto, "kernel_shape"),
                    get_attr_ints(node_proto, "pads"),
                    get_attr_ints(node_proto, "strides"),
                ),
            }),
            "BatchNormalization" => Ok(Self::BatchNormalization {
                attr: BatchNormalizationOp::new(
                    get_attr_f(node_proto, "epsilon"),
                    get_attr_f(node_proto, "momentum"),
                ),
            }),
            "Clip" => Ok(Self::Clip {
                attr: ClipOp::new(),
            }),
            "Concat" => Ok(Self::Concat {
                attr: ConcatOp {
                    axis: get_attr_i(node_proto, "axis").unwrap(),
                },
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
            "ConvTranspose" => Ok(Self::ConvTranspose {
                attr: ConvTransposeOp::new(
                    get_attr_ints(node_proto, "dilations"),
                    get_attr_i(node_proto, "group"),
                    get_attr_ints(node_proto, "kernel_shape"),
                    get_attr_ints(node_proto, "output_padding"),
                    get_attr_ints(node_proto, "output_shape"),
                    get_attr_ints(node_proto, "pads"),
                    get_attr_ints(node_proto, "strides"),
                ),
            }),
            "Div" => Ok(Self::Div {
                attr: BinOpElementwise {},
            }),
            "Gemm" => Ok(Self::Gemm {
                attr: GemmOp::new(
                    get_attr_f(node_proto, "alpha"),
                    get_attr_f(node_proto, "beta"),
                    get_attr_i(node_proto, "transA"),
                    get_attr_i(node_proto, "transB"),
                ),
            }),
            "GlobalAveragePool" => Ok(Self::GlobalAveragePool {
                attr: GlobalAveragePoolOp {},
            }),
            "HardSigmoid" => Ok(Self::HardSigmoid {
                attr: UnOpElementwise::new(vec![
                    attribute!("alpha", get_attr_f(node_proto, "alpha").unwrap_or(0.2)),
                    attribute!("beta", get_attr_f(node_proto, "beta").unwrap_or(0.5)),
                ]),
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
            "Resize" => Ok(Self::Resize {
                attr: ResizeOp::new(
                    get_attr_i(node_proto, "antialias"),
                    get_attr_ints(node_proto, "axes"),
                    get_attr_string(node_proto, "coordinate_transformation_mode"),
                    get_attr_f(node_proto, "cubic_coeff_a"),
                    get_attr_i(node_proto, "exclude_outside"),
                    get_attr_f(node_proto, "extrapolation_value"),
                    get_attr_string(node_proto, "keep_aspect_ratio_policy"),
                    get_attr_string(node_proto, "mode"),
                    get_attr_string(node_proto, "nearest_mode"),
                ),
            }),
            "Sigmoid" => Ok(Self::Sigmoid {
                attr: UnOpElementwise::new(vec![]),
            }),
            _ => Err(UnsupportedONNXOps(node_proto.get_op_type().to_string())),
        }
    }
}

pub trait Compile {
    fn compile(
        &self,
        op: &Op,
        shader_templ: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError>;
    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3];
}

pub struct ShaderTemplate<'templ> {
    tera: tera::Tera,
    ctx: tera::Context,
    template_name: &'templ str,
}

impl<'templ> ShaderTemplate<'templ> {
    pub fn new(
        template_name: &'templ str,
        template_str: &'templ str,
    ) -> Result<Self, GosonnxError> {
        let mut tera = tera::Tera::default();
        let ctx = tera::Context::new();

        // Include common base templates
        let unary_shader_source = SHADER_DIR
            .get_file("_unary_elementwise.glsl")
            .unwrap()
            .contents_utf8()
            .unwrap();
        let binary_shader_source = SHADER_DIR
            .get_file("_binary_elementwise.glsl")
            .unwrap()
            .contents_utf8()
            .unwrap();

        // Include ops specific template
        tera.add_raw_template("_unary_elementwise", unary_shader_source)
            .map_err(|e| Error(e.to_string()))?;
        tera.add_raw_template("_binary_elementwise", binary_shader_source)
            .map_err(|e| Error(e.to_string()))?;

        tera.add_raw_template(template_name, template_str)
            .map_err(|e| Error(e.to_string()))?;

        Ok(Self {
            tera,
            ctx,
            template_name,
        })
    }

    pub fn compile(&self) -> Result<String, GosonnxError> {
        let compiled = self
            .tera
            .render(self.template_name, &self.ctx)
            .map_err(|e| ShaderCompileError(e.to_string()))?;
        Ok(compiled)
    }

    pub fn push_attr<Val: Serialize + ?Sized>(&mut self, attr_name: &str, attr_val: &Val) {
        self.ctx.insert(attr_name, attr_val)
    }

    pub fn add_template(&mut self, name: &'templ str, content: &'templ str) -> Result<(), String> {
        self.tera
            .add_raw_template(name, content)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl<'gr, 'gpu> OpType {
    fn _compile<Compilable: Compile>(
        &self,
        attr: Compilable,
        shader_source: &str,
        op: &'gr Op,
        graph: &'gr Graph,
    ) -> Result<(String, [u32; 3]), GosonnxError> {
        let mut templ = ShaderTemplate::new(&op.op_name, shader_source)?;
        // let compiled = attr.compile(op, shader_source, graph)?;
        attr.compile(op, &mut templ, graph)?;
        let compiled = templ.compile()?;
        let wg = attr.compute_workgroup_size(op, graph);
        Ok((compiled, wg))
    }
}

pub fn to_csv_str<T: ToString + Display>(vals: &Vec<T>) -> String {
    let res: Vec<String> = vals.iter().map(|v| format!("{:.2}", *v)).collect();
    res.join(",")
}

pub fn compile_unary(
    op: &Op,
    attr: Option<Vec<(&str, String)>>,
    _shader_source: &str,
    _graph: &Graph,
) -> Result<String, GosonnxError> {
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
    .map_err(|e| ShaderCompileError(e.to_string()))?;

    let compiled = tera
        .render(&op.op_type.to_string(), &mut context)
        .map_err(|e| ShaderCompileError(e.to_string()))?;
    Ok(compiled)
}
