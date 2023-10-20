use crate::onnx::onnx::NodeProto;

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
