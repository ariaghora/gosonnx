#[macro_export]
macro_rules! attribute {
    ($key:expr, $value:expr) => {
        (
            $key.to_string(),
            Box::new($value) as Box<dyn std::fmt::Display>,
        )
    };
}
