use crate::{graph::Tensor, onnx::onnx::NodeProto};

pub fn get_attr_ints<'a>(node_proto: &'a NodeProto, attr_name: &str) -> Option<Vec<i64>> {
    for attr in node_proto.get_attribute() {
        if attr.get_name() == attr_name {
            return Some(attr.get_ints().to_vec());
        }
    }
    None
}

pub fn get_attr_f<'a>(node_proto: &'a NodeProto, attr_name: &str) -> Option<f32> {
    for attr in node_proto.get_attribute() {
        if attr.get_name() == attr_name {
            return Some(attr.get_f());
        }
    }
    None
}

pub fn get_attr_i<'a>(node_proto: &'a NodeProto, attr_name: &str) -> Option<i64> {
    for attr in node_proto.get_attribute() {
        if attr.get_name() == attr_name {
            return Some(attr.get_i());
        }
    }
    None
}

pub fn tensor_len(t: &Tensor) -> Result<usize, String> {
    let len = match t {
        Tensor::F32 { values: _, shape } => shape,
        Tensor::F64 { values: _, shape } => shape,
    }
    .iter()
    .fold(1, |x, y| x * y) as usize;
    Ok(len)
}

pub fn vec_close<T: num_traits::Float>(a: Vec<T>, b: Vec<T>) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > T::from(1e-4).unwrap() {
            return false;
        }
    }
    return true;
}
