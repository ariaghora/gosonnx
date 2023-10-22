use crate::graph::{Graph, Op};

use super::Compile;

pub struct ConvOp {
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvOp {
    pub fn new(
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
    ) -> Self {
        Self {
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        }
    }
}

pub fn to_csv_str<T: ToString>(vals: &Vec<T>) -> String {
    let res: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    res.join(",")
}

impl Compile for ConvOp {
    fn compile(&self, op: &Op, shader_template: &str, graph: &Graph) -> Result<String, String> {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template("Conv", shader_template)
            .map_err(|e| e.to_string())?;

        let x = &graph.tensor_map[&op.inputs[0]];
        let w = &graph.tensor_map[&op.inputs[1]];
        let y = &graph.tensor_map[&op.outputs[0]];

        context.insert("X_type", &x.type_glsl());
        context.insert("W_type", &w.type_glsl());
        // Output type is assumed to be identical with input type
        context.insert("Y_type", &x.type_glsl());

        context.insert("in_dim", &to_csv_str(&x.shape()));
        context.insert("weight_dim", &to_csv_str(&w.shape()));
        context.insert("out_dim", &to_csv_str(&y.shape()));

        context.insert("dilations", &to_csv_str(&self.dilations));
        context.insert("group", &self.group);
        context.insert("kernel_shape", &to_csv_str(&self.kernel_shape));
        context.insert("pads", &to_csv_str(&self.pads));
        context.insert("strides", &to_csv_str(&self.strides));
        context.insert("output_channels", &w.shape()[0]);

        if op.inputs.len() > 2 {
            context.insert("use_bias", &true);
            let b = &graph.tensor_map[&op.inputs[2]];
            context.insert("B_type", &b.type_glsl());
        }

        let compiled = tera.render("Conv", &context).map_err(|e| e.to_string())?;
        println!("COMPILED {}", compiled);
        Ok(compiled)
    }
}
