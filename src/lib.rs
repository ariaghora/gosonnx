pub mod gpu;
pub mod ops;

use std::{cell::RefCell, collections::HashMap};

use gpu::GPUExecutor;
use ops::relu::make_relu;

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

pub struct Op {
    op_type: String,
    op_name: String,
    prevs: Vec<String>,
    nexts: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl Op {
    pub fn new() -> Self {
        Self {
            op_type: "Unknown".into(),
            op_name: String::new(),
            prevs: vec![],
            nexts: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }
}

pub struct Graph {
    executor: Option<RefCell<GPUExecutor>>,
    pub tensor_map: HashMap<String, Tensor>,
    pub op_map: HashMap<String, Op>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            executor: None,
            tensor_map: HashMap::new(),
            op_map: HashMap::new(),
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
        op_type: &str,
    ) -> Result<(), String> {
        self.op_map.insert(
            op_name.into(),
            Op {
                op_type: op_type.into(),
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

    pub fn get_output(&self, name: &str) -> Option<Tensor> {
        if let Some(executor) = &self.executor {
            executor.borrow_mut().get_output(name, self)
        } else {
            None
        }
    }

    pub fn new_tensor_f32(
        &mut self,
        tensor_name: String,
        values: Option<Vec<f32>>,
        shape: Vec<i64>,
    ) {
        self.tensor_map
            .insert(tensor_name, Tensor::F32 { values, shape });
    }
}

pub fn open(filename: &str) {}

pub fn run() {}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn simple_relu() -> Result<(), Box<dyn Error>> {
        // Graph definition
        let mut graph = Graph::new();
        graph.new_tensor_f32("X".into(), Some(vec![0.5, -1.0, 2.0]), vec![1, 3]);
        graph.new_tensor_f32("Y".into(), None, vec![1, 3]);
        graph
            .new_op(vec!["X"], vec!["Y"], "my_relu_1", "Relu")
            .unwrap();

        // Graph execution
        graph.run()?;
        if let Some(result) = graph.get_output("Y") {
            if let Tensor::F32 { values, shape } = result {
                assert_eq!(values, Some(vec![0.5, 0.0, 2.0]));
                assert_eq!(shape, vec![1, 3]);
            } else {
                // panic!("Output should be Tensor::F32")
            }
        } else {
            // panic!("Output Y not found")
        }

        Ok(())
    }
}
