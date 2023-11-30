use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::{
    Error, InvalidInputDimension, InvalidInputNo, InvalidType, UnknownTensorType,
};
use crate::graph::{Tensor, TensorType};

use super::{to_csv_str, Compile};

#[derive(Debug, Serialize, Clone)]
pub struct ResizeOp {
    antialias: Option<i64>,
    axes: Option<Vec<i64>>,
    coordinate_transformation_mode: Option<String>,
    cubic_coeff_a: Option<f32>,
    exclude_outside: Option<i64>,
    extrapolation_value: Option<f32>,
    keep_aspect_ratio_policy: Option<String>,
    mode: Option<String>,
    nearest_mode: Option<String>,
}

#[derive(Serialize)]
struct InputInfo {
    dtype: String,
    n_dim: usize,
    shape_csv: String,
}

impl ResizeOp {
    pub fn new(
        antialias: Option<i64>,
        axes: Option<Vec<i64>>,
        coordinate_transformation_mode: Option<String>,
        cubic_coeff_a: Option<f32>,
        exclude_outside: Option<i64>,
        extrapolation_value: Option<f32>,
        keep_aspect_ratio_policy: Option<String>,
        mode: Option<String>,
        nearest_mode: Option<String>,
    ) -> Self {
        Self {
            antialias,
            axes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy,
            mode,
            nearest_mode,
        }
    }
}

impl Compile for &ResizeOp {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut super::ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), GosonnxError> {
        let input = &graph.tensor_map[&op.inputs[0]];
        let output = &graph.tensor_map[&op.outputs[0]];
        if input.shape().len() != 4 {
            return Err(InvalidInputDimension {
                expected: 4,
                found: input.shape().len(),
            });
        }

        let inputs: Vec<&Tensor> = op.inputs.iter().map(|v| &graph.tensor_map[v]).collect();
        let input_info_arr: Vec<InputInfo> = inputs
            .iter()
            .map(|v| InputInfo {
                dtype: v.type_glsl(),
                n_dim: v.shape().len(),
                shape_csv: to_csv_str(&v.shape()),
            })
            .collect();
        shader_templ.push_attr("input_info_arr", &input_info_arr);
        shader_templ.push_attr("output_binding_no", &inputs.len());
        shader_templ.push_attr("output_dtype", &output.type_glsl());

        // we assume resize has at least 3 inputs: X, roi, and either scales or sizes
        if op.inputs.len() < 3 {
            return Err(InvalidInputNo {
                expected: 3,
                found: op.inputs.len(),
            });
        }

        let roi = &graph.tensor_map[&op.inputs[1]];
        if let Tensor::F32 { values, .. } = roi {
            if let Some(val) = values {
                let roi_prod = val.iter().fold(1.0, |x, y| x * y);
                if roi_prod > 0.0 {
                    println!("Resize with ROI now is unsupported. It will be ignored.");
                }
            }
        } else {
            return Err(Error(
                "Second input is expected to be roi with f32 type".into(),
            ));
        };

        let s = &graph.tensor_map[&op.inputs[2]];
        match s.tensor_type() {
            TensorType::F32 => {
                // This is probably scales
                let Tensor::F32 { values, .. } = s else {
                    return Err(InvalidType {expected:"f32".to_string(), found:s.type_glsl()});
                };
                shader_templ.push_attr("scales", &to_csv_str(&values.as_ref().unwrap()));
                shader_templ.push_attr("scales_len", &values.as_ref().unwrap().len());
            }
            TensorType::I64 => {
                // This is probably sizes
                let Tensor::I64 { values, .. } = s else {
                    return Err(InvalidType {expected:"i64".to_string(), found:s.type_glsl()});
                };
                shader_templ.push_attr("sizes", &to_csv_str(&values.as_ref().unwrap()));
            }
            _ => return Err(UnknownTensorType(s.type_glsl().to_string())),
        }

        shader_templ.push_attr("in_dim", &to_csv_str(&input.shape()));
        shader_templ.push_attr("in_type", &input.type_glsl());
        shader_templ.push_attr("out_dim", &to_csv_str(&output.shape()));
        shader_templ.push_attr("out_type", &output.type_glsl());

        let antialias = match self.antialias {
            Some(0) => 0,
            None => {
                println!("Defaulting antialias to 0");
                0
            }
            _ => {
                println!("Antialias other than 0 is not supported yet");
                0
            }
        };
        shader_templ.push_attr("antialias", &antialias);

        let default_axes = (0..input.shape().len()).map(|v| v as i64).collect();
        let axes = self.axes.as_ref().unwrap_or(&default_axes);
        shader_templ.push_attr("axes_csv", &to_csv_str(&axes));
        shader_templ.push_attr("axes_len", &axes.len());

        // TODO: implement other than nearest mode
        let mode = self.mode.as_ref().unwrap().as_str();
        shader_templ.push_attr(
            "mode",
            match mode {
                "nearest" => &0,
                "linear" => {
                    println!("linear resize mode is ignored for now, using nearest");
                    &0
                }
                "cubic" => {
                    println!("cubic resize mode is ignored for now, using nearest");
                    &0
                }
                _ => &0,
            },
        );

        let nearest_mode = self
            .nearest_mode
            .clone()
            .unwrap_or("round_prefer_floor".to_string());
        shader_templ.push_attr(
            "nearest_mode",
            match nearest_mode.as_str() {
                "round_prefer_floor" => &0,
                "round_prefer_ceil" => &1,
                "floor" => &2,
                "ceil" => &3,
                _ => &0,
            },
        );
        Ok(())
    }

    fn compute_workgroup_size(
        &self,
        op: &crate::graph::Op,
        graph: &crate::graph::Graph,
    ) -> [u32; 3] {
        let output_dims = &graph.tensor_map[&op.outputs[0]].shape();
        let local_size_x_y = 16;

        let workgroup_size_x = ((output_dims[3] as f64) / (local_size_x_y as f64)).ceil() as u32; // width
        let total_output_height = output_dims[2] as f64 * output_dims[1] as f64; // height * channels
        let workgroup_size_y = (total_output_height / (local_size_x_y as f64)).ceil() as u32;

        let workgroup_size_z = 1;

        [workgroup_size_x, workgroup_size_y, workgroup_size_z]
    }

    fn activable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::errors::GosonnxError;
    use crate::graph::Tensor;
    use crate::{graph::Graph, ops::OpType};

    use super::ResizeOp;

    #[test]
    fn test_resize_simple() -> Result<(), GosonnxError> {
        let mut graph = Graph::new();
        graph.new_tensor_f32(
            "A",
            Some((0..4).map(|v| (v as f32)).collect()),
            vec![1, 1, 2, 2],
        )?;
        graph.new_tensor_f32("roi", None, vec![0])?;
        graph.new_tensor_f32("scales", Some(vec![1.0, 1.0, 2.0, 2.0]), vec![4])?;
        graph.new_tensor_f32("Y", None, vec![1, 1, 4, 4])?;
        graph
            .new_op(
                vec!["A", "roi", "scales"],
                vec!["Y"],
                "resize",
                OpType::Resize {
                    attr: ResizeOp::new(
                        None,
                        None,
                        Some("asymmetric".to_string()),
                        None,
                        None,
                        None,
                        None,
                        Some("nearest".to_string()),
                        Some("floor".to_string()),
                    ),
                },
            )
            .unwrap();
        graph.run().unwrap();
        let out = graph.get_output("Y").unwrap();
        if let Tensor::F32 { values, .. } = out {
            assert_eq!(
                values,
                &Some(vec![
                    0., 0., 1., 1., 0., 0., 1., 1., 2., 2., 3., 3., 2., 2., 3., 3.,
                ])
            );
        }
        Ok(())
    }
}
