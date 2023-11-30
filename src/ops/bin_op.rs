use serde::Serialize;

use crate::errors::GosonnxError;
use crate::errors::GosonnxError::IncompatibleShape;
use crate::{
    graph::{Graph, Op},
    ops::to_csv_str,
    utils::tensor_len,
};

use super::{Compile, ShaderTemplate};

#[derive(Debug, Serialize, Clone)]
pub struct BinOpElementwise;

#[derive(Debug)]
struct BroadcastResult {
    shape: Vec<i64>,
    left_physical_strides: Vec<i64>,
    right_physical_strides: Vec<i64>,
    left_logical_strides: Option<Vec<i64>>,
    right_logical_strides: Option<Vec<i64>>,
}

fn get_broadcast_shape(
    s1: Vec<i64>,
    s2: Vec<i64>,
) -> Result<Option<BroadcastResult>, GosonnxError> {
    if s1 == s2 {
        return Ok(None);
    }

    let l_physical_strides = shape_to_strides(&s1);
    let r_physical_strides = shape_to_strides(&s2);

    let mut l_logical_strides: Vec<i64> = l_physical_strides.iter().rev().map(|v| *v).collect();
    let mut r_logical_strides: Vec<i64> = r_physical_strides.iter().rev().map(|v| *v).collect();

    let maxlen = s1.len().max(s2.len());
    let mut s1_rev: Vec<i64> = s1.iter().rev().map(|v| *v).collect();
    let mut s2_rev: Vec<i64> = s2.iter().rev().map(|v| *v).collect();

    while s1_rev.len() < maxlen {
        s1_rev.push(1);
        l_logical_strides.push(0);
    }
    while s2_rev.len() < maxlen {
        s2_rev.push(1);
        r_logical_strides.push(0);
    }
    s1_rev.reverse();
    s2_rev.reverse();
    l_logical_strides.reverse();
    r_logical_strides.reverse();

    for i in 0..s1_rev.len() {
        if s1_rev[i] != s2_rev[i] {
            match (s1_rev[i], s2_rev[i]) {
                (_, 1) => {
                    s2_rev[i] = s1_rev[i];
                    r_logical_strides[i] = 0;
                }
                (1, _) => {
                    s1_rev[i] = s2_rev[i];
                    l_logical_strides[i] = 0;
                }
                _ => {
                    return Err(IncompatibleShape {
                        msg: "tensor shapes are incompatible".to_string(),
                        expected: s1,
                        found: s2,
                    });
                }
            };
        }
    }

    Ok(Some(BroadcastResult {
        shape: s1_rev.clone(),
        left_physical_strides: l_physical_strides,
        right_physical_strides: r_physical_strides,
        left_logical_strides: if s1_rev == s1 {
            None
        } else {
            Some(l_logical_strides)
        },
        right_logical_strides: if s2_rev == s2 {
            None
        } else {
            Some(r_logical_strides)
        },
    }))
}

pub fn shape_to_strides(shape: &Vec<i64>) -> Vec<i64> {
    let mut res = vec![];
    for i in 0..shape.len() {
        let mut prod: i64 = 1;
        for j in i + 1..shape.len() {
            prod *= shape[j];
        }
        res.push(prod);
    }
    res
}

fn generate_direct_strided_offset_glsl(
    fn_name_suffix: &str,
    common_shape: &Vec<i64>,
    logical_strides: &Vec<i64>,
    actual_strides: &Vec<i64>,
) -> String {
    let mut code = String::new();
    code.push_str(&format!(
        "uint get_direct_strided_offset_{}(uint i) {{\n",
        fn_name_suffix
    ));
    code.push_str("    uint strided_offset = 0;\n");
    code.push_str("    uint idx;\n");

    let mut cumulative_factor = 1;
    for (&shape, &logical_stride, &actual_stride) in itertools::izip!(
        common_shape.iter().rev(),
        logical_strides.iter().rev(),
        actual_strides.iter().rev(),
    ) {
        if logical_stride != 0 {
            code.push_str(&format!(
                "    idx = (i / {}) % {};\n",
                cumulative_factor, shape
            ));
            code.push_str(&format!("    strided_offset += idx * {};\n", actual_stride));
        }
        cumulative_factor *= shape;
    }

    code.push_str("    return strided_offset;\n");
    code.push_str("}\n");
    code
}

pub fn compile_binary(
    op: &Op,
    _shader_templ: &mut ShaderTemplate,
    _graph: &Graph,
) -> Result<(), GosonnxError> {
    let input_1 = &_graph.tensor_map[&op.inputs[0]];
    let input_2 = &_graph.tensor_map[&op.inputs[1]];
    let output = &_graph.tensor_map[&op.outputs[0]];

    let l_len = tensor_len(input_1).unwrap();
    let r_len = tensor_len(input_2).unwrap();
    let (left_oneval, right_oneval) = (l_len <= 1, r_len <= 1);

    if let Some(broadcast_result) = get_broadcast_shape(input_1.shape(), input_2.shape())? {
        let common_shape = broadcast_result.shape;
        _shader_templ.push_attr("common_shape", &to_csv_str(&common_shape));
        _shader_templ.push_attr("common_shape_len", &common_shape.len());

        if let Some(left_logical_strides) = broadcast_result.left_logical_strides {
            let get_direct_strided_offset_fn = generate_direct_strided_offset_glsl(
                "l",
                &common_shape,
                &left_logical_strides,
                &broadcast_result.left_physical_strides,
            );
            _shader_templ.push_attr(
                "get_direct_strided_offset_l_fn",
                &get_direct_strided_offset_fn,
            );
            _shader_templ.push_attr(
                "left_physical_strides",
                &broadcast_result.left_physical_strides,
            );
            _shader_templ.push_attr("left_logical_strides", &left_logical_strides);
        }

        if let Some(right_logical_strides) = broadcast_result.right_logical_strides {
            let get_direct_strided_offset_fn = generate_direct_strided_offset_glsl(
                "r",
                &common_shape,
                &right_logical_strides,
                &broadcast_result.right_physical_strides,
            );
            _shader_templ.push_attr(
                "get_direct_strided_offset_r_fn",
                &get_direct_strided_offset_fn,
            );
            _shader_templ.push_attr(
                "right_physical_strides",
                &broadcast_result.right_physical_strides,
            );
            _shader_templ.push_attr("right_logical_strides", &right_logical_strides);
        }
    }

    _shader_templ.push_attr("input_1_type", &input_1.type_glsl());
    _shader_templ.push_attr("input_2_type", &input_2.type_glsl());
    _shader_templ.push_attr("output_type", &output.type_glsl());

    _shader_templ.push_attr("left_oneval", &left_oneval);
    _shader_templ.push_attr("right_oneval", &right_oneval);

    Ok(())
}

impl Compile for &BinOpElementwise {
    fn compile(
        &self,
        op: &crate::graph::Op,
        shader_templ: &mut ShaderTemplate,
        graph: &crate::graph::Graph,
    ) -> Result<(), GosonnxError> {
        compile_binary(op, shader_templ, graph)
    }

    fn compute_workgroup_size(
        &self,
        op: &crate::graph::Op,
        graph: &crate::graph::Graph,
    ) -> [u32; 3] {
        let local_size_x = 256;
        let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
        let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }

    fn activable(&mut self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, Tensor},
        ops::{bin_op::get_broadcast_shape, OpType},
    };

    #[test]
    fn test_get_broadcast_shape() {
        let s1 = vec![1, 2, 2];
        let s2 = vec![1, 2, 2];
        let res = get_broadcast_shape(s1.clone(), s2.clone()).unwrap();
        assert!(res.is_none());

        let s1 = vec![1, 2, 2];
        let s2 = vec![2];
        let res = get_broadcast_shape(s1.clone(), s2).unwrap().unwrap();
        assert_eq!(res.shape, s1);
        assert!(res.left_logical_strides.is_none());
        assert!(res.right_logical_strides.is_some());

        // Multidirectional broadcasting
        let s1 = vec![3, 1];
        let s2 = vec![5];
        let res = get_broadcast_shape(s1, s2).unwrap().unwrap();
        assert_eq!(res.shape, vec![3, 5]);
        assert!(res.left_logical_strides.is_some());
        assert!(res.right_logical_strides.is_some());

        // Incompatible shape
        let s1 = vec![3, 3, 2];
        let s2 = vec![1, 1, 5];
        let out = get_broadcast_shape(s1, s2);
        assert!(out.is_err());
    }

    #[test]
    fn div_no_bcast() {
        let mut graph = Graph::new();
        graph
            .new_tensor_f32("A", Some((1..10).map(|v| v as f32).collect()), vec![3, 3])
            .unwrap();
        graph
            .new_tensor_f32("B", Some((1..10).map(|_| 2.0).collect()), vec![3, 3])
            .unwrap();
        graph.new_tensor_f32("Y", None, vec![3, 3]).unwrap();
        graph
            .new_op(
                vec!["A", "B"],
                vec!["Y"],
                "add",
                OpType::Div {
                    attr: super::BinOpElementwise {},
                },
            )
            .unwrap();
        graph.run().unwrap();
        if let Some(t) = graph.get_output("Y") {
            if let Tensor::F32 { values, .. } = t {
                assert_eq!(
                    values,
                    &Some((1..10).map(|v| (v as f32) / 2.0).collect::<Vec<f32>>())
                )
            } else {
                panic!("Invalid tensor found")
            }
        } else {
            panic!("No output found")
        }
    }
}
