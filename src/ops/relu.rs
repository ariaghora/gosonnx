use std::collections::HashMap;

use crate::Op;

pub fn make_relu<'a>(
    node_map: &'a mut HashMap<String, Op>,
    input_names: Vec<&str>,
) -> Result<(), String> {
    if input_names.len() != 1 {
        return Err("Relu requires exactly 1 input".into());
    }

    let output_name = uuid::Uuid::new_v4().to_string();
    // node_map.insert(output_name, Op::Relu);
    Ok(())
}
