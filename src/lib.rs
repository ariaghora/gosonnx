pub mod gpu;
pub mod ops;

use gpu::GPUExecutor;
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

    pub fn connect(&mut self, from: &str, to: &str) -> Result<(), String> {
        if self.op_map.get(from).is_none() {
            return Err(format!("{} not found in graph", from));
        }
        if self.op_map.get(to).is_none() {
            return Err(format!("{} not found in graph", to));
        }

        self.op_map
            .get_mut(to)
            .unwrap()
            .prevs
            .push(from.to_string());

        self.op_map
            .get_mut(from)
            .unwrap()
            .nexts
            .push(to.to_string());
        Ok(())
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
                nexts: vec![],
                inputs: input_names.iter().map(|s| s.to_string()).collect(),
                outputs: output_names.iter().map(|s| s.to_string()).collect(),
            },
        );
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), String> {
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

pub fn open(_filename: &str) {}

pub fn run() {}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

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
        graph.connect("my_relu", "my_double")?;
        graph.connect("my_double", "my_double2")?;
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
