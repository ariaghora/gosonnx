macro_rules! attr {
    ($key:expr, $value:expr) => {
        (
            $key.to_string(),
            Box::new($value) as Box<dyn std::fmt::Display>,
        )
    };
}
