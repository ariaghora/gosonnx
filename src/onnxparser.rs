use crate::{
    onnx::{self, ModelProto, NodeProto, TensorProto},
    Graph, Op, OpType, Tensor,
};
use anyhow::{anyhow, Result};

struct OpNode {
    outs: Vec<String>,
    ins: Vec<String>,
}

fn get_tensor_proto<'a>(name: &str, model_proto: &'a ModelProto) -> Result<&'a TensorProto> {
    let filtered = model_proto
        .get_graph()
        .get_initializer()
        .iter()
        .filter(|t| t.get_name() == name)
        .collect::<Vec<&TensorProto>>();

    match filtered.first().map(|t| *t) {
        Some(v) => Ok(v),
        None => Err(anyhow::anyhow!(format!(
            "Tensor proto associated with `{}` not found",
            name
        ))),
    }
}

pub(crate) fn parse_model_proto(model_proto: &mut onnx::ModelProto) -> Result<Graph> {
    // Add name to unnamed nodes
    let mut cnt = 1;
    for i in 0..model_proto.get_graph().get_node().len() {
        if model_proto.get_graph().get_node()[i].get_name() == "" {
            model_proto.mut_graph().mut_node()[i].set_name(format!("unnamed_{}", cnt));
            cnt += 1;
        }
    }

    let mut graph = Graph::new();

    // Create a tensor for each input and output of the graph
    for input in model_proto.get_graph().get_input() {
        graph.tensor_map.insert(
            input.get_name().into(),
            Tensor::value_from_value_info_proto(input).map_err(|e| anyhow!(e))?,
        );
    }
    for output in model_proto.get_graph().get_output() {
        graph.tensor_map.insert(
            output.get_name().into(),
            Tensor::value_from_value_info_proto(output).map_err(|e| anyhow!(e))?,
        );
    }
    // Also create a tensor for each initializer
    for init in model_proto.get_graph().get_initializer() {
        graph.tensor_map.insert(
            init.get_name().into(),
            Tensor::from_tensor_proto(init, false).map_err(|e| anyhow!(e))?,
        );
    }

    // Ensure each node's output and input tensors are created
    for val in model_proto.get_graph().get_value_info() {
        graph.tensor_map.insert(
            val.get_name().into(),
            Tensor::value_from_value_info_proto(val).map_err(|e| anyhow!(e))?,
        );
    }

    // Finally the nodes themselves
    for node_proto in model_proto.get_graph().get_node() {
        match OpType::from_node_proto(node_proto).map_err(|e| anyhow!(e)) {
            Ok(op_type) => {
                let op = Op {
                    op_name: node_proto.get_name().into(),
                    op_type,
                    outputs: node_proto.get_output().to_vec(),
                    inputs: node_proto.get_input().to_vec(),
                    prevs: vec![],
                    nexts: vec![],
                };
                println!("Op {:#?} created!", op);
            }
            Err(e) => println!("{:#?}", e),
        }
    }
    graph.compile().map_err(|e| anyhow!(e))?;

    Err(anyhow::anyhow!("Error parsing model proto"))
}
