pub struct ConvOp {
    dilations: Vec<i64>,
}

impl ConvOp {
    pub fn new(dilations: Vec<i64>) -> Self {
        Self { dilations }
    }

    pub fn compile(&self) -> Result<String, String> {
        todo!()
    }
}
