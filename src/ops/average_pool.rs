use serde::Serialize;

use crate::graph::{Graph, Op};

use super::{to_csv_str, Compile};

#[derive(Debug, Serialize)]
pub struct AveragePoolOp {
    auto_pad: Option<String>,
    ceil_mode: Option<i64>,
    dilations: Option<Vec<i64>>,
    kernel_shape: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
}

impl AveragePoolOp {
    pub fn new(
        auto_pad: Option<String>,
        ceil_mode: Option<i64>,
        dilations: Option<Vec<i64>>,
        kernel_shape: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
    ) -> Self {
        Self {
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            pads,
            strides,
        }
    }
}
impl Compile for &AveragePoolOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_source: &str,
        graph: &crate::graph::Graph,
    ) -> Result<String, String> {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        tera.add_raw_template("AveragePool", shader_source)
            .map_err(|e| e.to_string())?;

        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        if x.shape().len() != 4 {
            return Err(format!(
                "AveragePool expects 4 dimensions, found {}",
                x.shape().len()
            ));
        }

        context.insert("X_type", &x.type_glsl());
        context.insert("Y_type", &y.type_glsl());
        context.insert("in_dim", &to_csv_str(&x.shape()));
        context.insert("out_dim", &to_csv_str(&y.shape()));

        let auto_pad = &self.auto_pad.clone().unwrap_or("NOTSET".to_string());
        let ceil_mode = &self.ceil_mode.unwrap_or(0);
        let dilations = &self.dilations.clone().unwrap_or(vec![1; x.shape().len()]);
        let kernel_shape = if let Some(kernel_shape) = &self.kernel_shape {
            kernel_shape
        } else {
            return Err("Kernel shape is mandatory but not provided".into());
        };

        let pads = &self.pads.clone().unwrap_or(vec![1, 1, 1, 1]);
        let strides = &self.strides.clone().unwrap_or(vec![1, 1]);

        context.insert("auto_pad", &auto_pad);
        context.insert("ceil_mode", &ceil_mode);
        context.insert("dilations", &to_csv_str(dilations));
        context.insert("kernel_shape", &to_csv_str(kernel_shape));
        context.insert("pads", &to_csv_str(pads));
        context.insert("strides", &to_csv_str(strides));

        let compiled = tera
            .render("AveragePool", &mut context)
            .map_err(|e| e.to_string())?;

        Ok(compiled)
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();

        let local_size_x = 16;
        let local_size_y = 16;

        let workgroup_size_x = ((output_dims[0] + local_size_x - 1) / local_size_x) as u32; // N
        let workgroup_size_y = ((output_dims[1] + local_size_y - 1) / local_size_y) as u32; // C

        [workgroup_size_x, workgroup_size_y, 1]
    }
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    use super::AveragePoolOp;

    #[test]
    fn simple_global_average_pool() {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "X",
            Some((1..=18).map(|v| v as f32).collect()),
            vec![1, 2, 3, 3],
        );

        graph.new_tensor_f32("Y", None, vec![1, 2, 2, 2]);
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "avg_pool",
                OpType::AveragePool {
                    attr: AveragePoolOp::new(
                        None,
                        Some(0),
                        Some(vec![1, 1]),
                        Some(vec![2, 2]),
                        Some(vec![0, 0, 0, 0]),
                        Some(vec![1, 1]),
                    ),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(values, &Some(vec![3., 4., 6., 7., 12., 13., 15., 16.]));
        }
    }
}
