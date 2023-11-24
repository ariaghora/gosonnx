#[derive(Debug)]
pub enum GosonnxError {
    AttributeNotFound(String),
    TensorCreateError(String),
    ShapeMismatchError,
    ShaderCompileError(String),
    UnsupportedONNXOps(String),
    OpsOnIncompatibleTypeError {
        left: String,
        right: String,
    },
    InvalidInputDimension {
        expected: usize,
        found: usize,
    },
    InvalidInputNo {
        expected: i32,
        found: usize,
    },
    InvalidType {
        expected: String,
        found: String,
    },
    IncompatibleShape {
        msg: String,
        expected: Vec<i64>,
        found: Vec<i64>,
    },
    UnknownTensorType(String),
    Error(String),
}
