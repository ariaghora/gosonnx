use crate::errors::GosonnxError;
use crate::errors::GosonnxError::{InvalidInputDimension, InvalidInputNo};
use crate::graph::{Graph, Op};
use crate::ops::{to_csv_str, Compile, ShaderTemplate};
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct BatchNormalizationOp {
    epsilon: Option<f32>,
    momentum: Option<f32>,
}

impl BatchNormalizationOp {
    pub fn new(epsilon: Option<f32>, momentum: Option<f32>) -> Self {
        Self { epsilon, momentum }
    }
}

impl Compile for &BatchNormalizationOp {
    fn compile(
        &self,
        op: &Op,
        shader_templ: &mut ShaderTemplate,
        graph: &Graph,
    ) -> Result<(), GosonnxError> {
        if op.inputs.len() != 5 {
            return Err(InvalidInputNo {
                expected: 5,
                found: op.inputs.len(),
            });
        }

        let input = &graph.tensor_map[&op.inputs[0]];
        if input.shape().len() != 4 {
            return Err(InvalidInputDimension {
                expected: 4,
                found: input.shape().len(),
            });
        }
        let scale = &graph.tensor_map[&op.inputs[1]];
        let b = &graph.tensor_map[&op.inputs[2]];
        let mean = &graph.tensor_map[&op.inputs[3]];
        let var = &graph.tensor_map[&op.inputs[4]];

        let output = &graph.tensor_map[&op.outputs[0]];

        let epsilon = self.epsilon.unwrap_or(1e-5);

        shader_templ.push_attr("input_type", &input.type_glsl());
        shader_templ.push_attr("scale_type", &scale.type_glsl());
        shader_templ.push_attr("b_type", &b.type_glsl());
        shader_templ.push_attr("mean_type", &mean.type_glsl());
        shader_templ.push_attr("var_type", &var.type_glsl());
        shader_templ.push_attr("epsilon", &epsilon);

        shader_templ.push_attr("output_type", &output.type_glsl());

        shader_templ.push_attr("in_dim", &to_csv_str(&input.shape()));
        shader_templ.push_attr("out_dim", &to_csv_str(&output.shape()));
        Ok(())
    }

    fn compute_workgroup_size(&self, op: &Op, graph: &Graph) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();

        let local_size_x = 16; // Or another value based on your hardware's capabilities
        let local_size_y = 16; // Same as above

        let workgroup_size_x = ((output_dims[2] + local_size_x - 1) / local_size_x) as u32; // H
        let workgroup_size_y = ((output_dims[3] + local_size_y - 1) / local_size_y) as u32; // W

        [workgroup_size_x, workgroup_size_y, 1]
    }

    fn activable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::graph::{Graph, Tensor};
    use crate::ops::batch_normalization::BatchNormalizationOp;
    use crate::ops::OpType;
    use crate::utils::vec_close;

    #[test]
    fn test_simple_batch_norm() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "X",
            Some((0..1 * 2 * 3 * 3).map(|v| (v as f32)).collect()),
            vec![1, 2, 3, 3],
        )?;
        graph.new_tensor_f32("scale", Some(vec![0.1, 0.5]), vec![2])?;
        graph.new_tensor_f32("b", Some(vec![1.0, 2.0]), vec![2])?;
        graph.new_tensor_f32("mean", Some(vec![11.0, 11.0]), vec![2])?;
        graph.new_tensor_f32("var", Some(vec![15.0, 15.0]), vec![2])?;
        graph.new_tensor_f32("Y", None, vec![1, 2, 3, 3])?;
        graph
            .new_op(
                vec!["X", "scale", "b", "mean", "var"],
                vec!["Y"],
                "bn",
                OpType::BatchNormalization {
                    attr: BatchNormalizationOp {
                        epsilon: Some(1e-5),
                        momentum: None,
                    },
                },
            )
            .unwrap();
        graph.run()?;
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert!(vec_close(
                values.as_ref().unwrap().clone(),
                vec![
                    0.7159813046455383,
                    0.7418012022972107,
                    0.7676210403442383,
                    0.7934409379959106,
                    0.819260835647583,
                    0.8450807332992554,
                    0.8709006309509277,
                    0.8967204689979553,
                    0.9225403666496277,
                    1.7418012619018555,
                    1.8709006309509277,
                    2.0,
                    2.1290993690490723,
                    2.2581987380981445,
                    2.387298107147217,
                    2.516397476196289,
                    2.6454970836639404,
                    2.7745964527130127,
                ],
            ));
        }

        Ok(())
    }
}
