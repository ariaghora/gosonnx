pub mod gpu;
pub mod onnx;
pub mod onnxparser;
pub mod ops;

use gpu::GPUExecutor;
use onnx::{TensorProto, ValueInfoProto};
use protobuf::Message;
use std::fmt;
use std::{cell::RefCell, collections::HashMap};

#[derive(Debug)]
pub enum Tensor {
    F32 {
        values: Option<Vec<f32>>,
        shape: Vec<i64>,
    },
    F64 {
        values: Option<Vec<f64>>,
        shape: Vec<i64>,
    },
}

impl Tensor {
    pub fn shape(&self) -> Vec<i64> {
        match self {
            Tensor::F32 { values: _, shape } => shape.clone(),
            Tensor::F64 { values: _, shape } => shape.clone(),
        }
    }

    pub fn type_glsl(&self) -> String {
        match self {
            Tensor::F32 { .. } => "float".into(),
            Tensor::F64 { .. } => "double".into(),
        }
    }

    pub(crate) fn from_tensor_proto(t: &TensorProto, empty: bool) -> Result<Tensor, String> {
        match t.get_data_type() {
            1 => Ok(Tensor::F32 {
                values: if empty {
                    None
                } else {
                    Some(bytemuck::cast_slice(t.get_raw_data()).to_vec())
                },
                shape: t.get_dims().to_vec(),
            }),
            _ => Err("Unsupported tensor proto data type".into()),
        }
    }

    pub(crate) fn value_from_value_info_proto(value_info: &ValueInfoProto) -> Result<Self, String> {
        if let Some(value) = &value_info.get_field_type().value {
            match value {
                onnx::TypeProto_oneof_value::tensor_type(t) => {
                    return match t.get_elem_type() {
                        1 => Ok(Tensor::F32 {
                            values: None,
                            shape: t
                                .get_shape()
                                .get_dim()
                                .iter()
                                .map(|v| v.get_dim_value())
                                .collect(),
                        }),
                        _ => Err(format!("Type `{}` not supported yet", t.get_elem_type())),
                    };
                }
                onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            }
        }
        Err(format!(
            "Value info proto {} does not have associated field type proto",
            value_info.get_name()
        ))
    }
}

#[derive(Debug)]
pub enum OpType {
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: i32,
        trans_b: i32,
    },
    Relu,
    Unknown,

    Double, //Dummy. TODO: remove
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Gemm { .. } => write!(f, "Gemm"),
            OpType::Relu => write!(f, "Relu"),
            OpType::Unknown => write!(f, "Unknown"),
            OpType::Double => write!(f, "Double"),
        }
    }
}

pub struct Op {
    op_type: OpType,
    op_name: String,
    prevs: Vec<String>,
    nexts: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl Op {
    pub fn new() -> Self {
        Self {
            op_type: OpType::Unknown,
            op_name: String::new(),
            prevs: vec![],
            nexts: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }
}

pub struct Graph {
    pub(crate) executor: Option<RefCell<GPUExecutor>>,
    pub(crate) tensor_map: HashMap<String, Tensor>,
    pub(crate) op_map: HashMap<String, Op>,
    pub(crate) output_tensor_map: HashMap<String, Tensor>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            executor: None,
            tensor_map: HashMap::new(),
            op_map: HashMap::new(),
            output_tensor_map: HashMap::new(),
        }
    }

    pub fn new_op(
        &mut self,
        input_names: Vec<&str>,
        output_names: Vec<&str>,
        op_name: &str,
        op_type: OpType,
    ) -> Result<(), String> {
        self.op_map.insert(
            op_name.into(),
            Op {
                op_type,
                op_name: String::from(op_name),
                prevs: vec![],
                nexts: vec![], // this will be filled later
                inputs: input_names.iter().map(|s| s.to_string()).collect(),
                outputs: output_names.iter().map(|s| s.to_string()).collect(),
            },
        );
        Ok(())
    }

    /// For all (A, B) node pairs in graph, connect A and B if A.outputs intersects
    /// with B.inputs and A != B
    fn compile(&mut self) -> Result<(), String> {
        let node_names = {
            let keys = self.op_map.keys();
            keys.into_iter()
                .map(|f| f.to_string())
                .collect::<Vec<String>>()
        };
        for from in &node_names {
            for to in &node_names {
                if from == to {
                    continue;
                }
                let from_n_outputs = &self.op_map.get(from).unwrap().outputs;
                let to_n_inputs = &self.op_map.get(to).unwrap().inputs;
                let connected = from_n_outputs.iter().all(|n| to_n_inputs.contains(n));
                if !to_n_inputs.contains(from) && connected {
                    self.op_map
                        .get_mut(from)
                        .unwrap()
                        .nexts
                        .push(to.to_string());

                    self.op_map
                        .get_mut(to)
                        .unwrap()
                        .prevs
                        .push(from.to_string());
                }
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), String> {
        self.compile()?;

        // Initialize GPU executor and run it!
        let mut executor = GPUExecutor::new();
        executor.execute(self)?;
        self.executor = Some(RefCell::new(executor));
        Ok(())
    }

    pub fn new_tensor_f32(&mut self, tensor_name: &str, values: Option<Vec<f32>>, shape: Vec<i64>) {
        self.tensor_map
            .insert(tensor_name.into(), Tensor::F32 { values, shape });
    }

    pub fn get_output(&self, arg: &str) -> Option<&Tensor> {
        self.output_tensor_map.get(arg)
    }
}

impl Graph {
    pub(crate) fn terminal_outputs(&self) -> Vec<String> {
        let mut outputs: Vec<String> = vec![];
        let terminal_nodes = self
            .op_map
            .values()
            .filter(|o| o.nexts.len() == 0)
            .collect::<Vec<&Op>>();
        for t_node in terminal_nodes {
            for out in &t_node.outputs {
                outputs.push(out.clone());
            }
        }
        outputs
    }
    pub fn open_onnx(filename: &str) -> Result<Graph, Box<dyn std::error::Error>> {
        let model_bytes = std::fs::read(filename).map_err(|e| e.to_string())?;
        let mut model_proto =
            onnx::ModelProto::parse_from_bytes(&model_bytes).map_err(|e| e.to_string())?;
        let graph = onnxparser::parse_model_proto(&mut model_proto)?;
        Ok(graph)
    }
}

pub fn run() {}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn open_onnx() -> Result<(), Box<dyn Error>> {
        let graph = Graph::open_onnx("data/models/doc_classifier_model.onnx")?;
        assert_eq!(graph.op_map.keys().len(), 13);
        Ok(())
    }

    #[test]
    fn simple_relu() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![0.5, -1.0, 2.0]), vec![1, 3]);
        graph.new_tensor_f32("Y", None, vec![1, 3]);
        graph
            .new_op(vec!["X"], vec!["Y"], "my_relu_1", OpType::Relu)
            .unwrap();

        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, shape } = result {
                assert_eq!(values, &Some(vec![0.5, 0.0, 2.0]));
                assert_eq!(shape, &vec![1, 3]);
            } else {
                panic!("Output should be Tensor::F32")
            }
        } else {
            panic!("Output Y not found")
        }

        Ok(())
    }

    #[test]
    fn multi_module() -> Result<(), Box<dyn Error>> {
        let mut graph = Graph::new();
        graph.new_tensor_f32("X", Some(vec![0.5, -1.0, 2.0]), vec![1, 3]);
        graph.new_tensor_f32("Y", None, vec![1, 3]);
        graph.new_tensor_f32("Z", None, vec![1, 3]);
        graph.new_tensor_f32("final".into(), None, vec![1, 3]);
        graph
            .new_op(vec!["X"], vec!["Y"], "my_relu", OpType::Relu)
            .unwrap();
        graph
            .new_op(vec!["Y"], vec!["Z"], "my_double", OpType::Double)
            .unwrap();
        graph
            .new_op(vec!["Z"], vec!["final"], "my_double2", OpType::Double)
            .unwrap();
        graph.run()?;

        if let Some(result) = graph.get_output("final") {
            if let Tensor::F32 { values, shape } = result {
                assert_eq!(values, &Some(vec![2.0, 0.0, 8.0]));
                assert_eq!(shape, &vec![1, 3]);
            } else {
                panic!("Output should be Tensor::F32")
            }
        } else {
            panic!("Output not found")
        }
        Ok(())
    }
}
