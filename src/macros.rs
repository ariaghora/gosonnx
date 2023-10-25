#[macro_export]
macro_rules! export_attrs {
    ($($variant:ident { $($field:ident),* }),*) => {
        pub trait ExportAttr {
            fn export_attr(&self) -> Vec<(String, String)>;
        }

        impl ExportAttr for &OpType {
             fn export_attr(&self) -> Vec<(String, String)> {
                let mut attrs = vec![];
                match self {
                    $(
                        OpType::$variant { $($field),* } => {
                            $(
                                let mut val = serde_json::to_string($field).unwrap();
                                val = if val.starts_with('[') && val.ends_with(']') {
                                    val[1..val.len() - 1].to_string()
                                } else {
                                    val
                                };
                                attrs.push((stringify!($field).to_string(), val));
                            )*
                        }
                    ),*
                }
                attrs
            }
        }
    };
}
