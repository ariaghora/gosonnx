use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::InvalidInputDimension;
use crate::graph::{Graph, Op};

use super::{to_csv_str, Compile, ShaderTemplate};

#[derive(Debug, Serialize, Clone)]
pub struct GlobalAveragePoolOp {}

impl GlobalAveragePoolOp {
    pub fn new() -> Self {
        Self {}
    }
}
impl Compile for &GlobalAveragePoolOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), GosonnxError> {
        let x = &graph.tensor_map[&op.inputs[0]];
        let y = &graph.tensor_map[&op.outputs[0]];

        if x.shape().len() != 4 {
            return Err(InvalidInputDimension {
                expected: 4,
                found: x.shape().len(),
            });
        }

        shader_templ.push_attr("X_type", &x.type_glsl());
        shader_templ.push_attr("Y_type", &y.type_glsl());
        shader_templ.push_attr("in_dim", &to_csv_str(&x.shape()));
        shader_templ.push_attr("out_dim", &to_csv_str(&y.shape()));

        // let compiled = tera
        //     .render("GlobaleAveragePool", &mut context)
        //     .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();

        let local_size_x = 16;
        let local_size_y = 16;

        let workgroup_size_x = ((output_dims[0] + local_size_x - 1) / local_size_x) as u32; // N
        let workgroup_size_y = ((output_dims[1] + local_size_y - 1) / local_size_y) as u32; // C

        [workgroup_size_x, workgroup_size_y, 1]
    }

    fn activable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::OpType,
    };

    use super::GlobalAveragePoolOp;

    #[test]
    fn simple_global_average_pool() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32(
                "X",
                Some((1..=16).map(|v| v as f32).collect()),
                vec![2, 2, 2, 2],
            )
            .unwrap();

        graph.new_tensor_f32("Y", None, vec![2, 2, 1, 1]).unwrap();
        graph
            .new_op(
                vec!["X"],
                vec!["Y"],
                "avg_pool",
                OpType::GlobalAveragePool {
                    attr: GlobalAveragePoolOp::new(),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(values, &Some(vec![2.5, 6.5, 10.5, 14.5]));
        }
    }
}
