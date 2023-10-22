use crate::ops::to_csv_str;

use super::Compile;

pub struct MaxPoolOp {
    ceil_mode: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl MaxPoolOp {
    pub fn new(ceil_mode: i64, kernel_shape: Vec<i64>, pads: Vec<i64>, strides: Vec<i64>) -> Self {
        Self {
            ceil_mode,
            kernel_shape,
            pads,
            strides,
        }
    }
}

impl Compile for MaxPoolOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_source: &str,
        graph: &crate::graph::Graph,
    ) -> Result<String, String> {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        tera.add_raw_template("MaxPool", shader_source)
            .map_err(|e| e.to_string())?;

        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        context.insert("X_type", &x.type_glsl());
        context.insert("Y_type", &y.type_glsl());
        context.insert("in_dim", &to_csv_str(&x.shape()));
        context.insert("out_dim", &to_csv_str(&y.shape()));

        context.insert("ceil_mode", &self.ceil_mode);
        context.insert("kernel_shape", &to_csv_str(&self.kernel_shape));
        context.insert("pads", &to_csv_str(&self.pads));
        context.insert("strides", &to_csv_str(&self.strides));

        let compiled = tera
            .render("MaxPool", &mut context)
            .map_err(|e| e.to_string())?;
        println!("COMPILED MaxPool shader:\n{}", compiled);

        Ok(compiled)
    }
}
