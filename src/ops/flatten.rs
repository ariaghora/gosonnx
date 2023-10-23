use crate::ops::to_csv_str;

use super::Compile;

pub struct FlattenOp {
    axis: i64,
}

impl FlattenOp {
    pub fn new(axis: i64) -> Self {
        Self { axis }
    }
}

impl Compile for FlattenOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_source: &str,
        graph: &crate::graph::Graph,
    ) -> Result<String, String> {
        if self.axis < 0 {
            return Err("Cannot handle negative axis yet".into());
        }
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        tera.add_raw_template("Flatten", shader_source)
            .map_err(|e| e.to_string())?;

        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        context.insert("X_type", &x.type_glsl());
        context.insert("Y_type", &y.type_glsl());
        context.insert("in_dim", &to_csv_str(&x.shape()));
        context.insert("out_dim", &to_csv_str(&y.shape()));
        context.insert("in_ndim", &x.shape().len());
        context.insert("out_ndim", &y.shape().len());
        context.insert("axis", &self.axis);

        let compiled = tera
            .render("Flatten", &mut context)
            .map_err(|e| e.to_string())?;
        println!("COMPILED FLATTEN\n{}", compiled);

        Ok(compiled)
    }
}
