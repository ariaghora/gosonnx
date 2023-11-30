#[macro_export]
macro_rules! attribute {
    ($key:expr, $value:expr) => {
        ($key.to_string(), format!("{}", $value))
    };
}

#[macro_export]
macro_rules! define_ops {
    ($($variant:ident { $attr_type:ty }),+ $(,)? ) => {
        #[derive(Debug, Serialize, Clone)]
        pub enum OpType {
            $(
                $variant { attr: $attr_type },
            )+
            Unknown,
        }

        impl<'gr, 'gpu> OpType {
            pub fn compile(
                &self,
                shader_source: &str,
                op: &'gr Op,
                graph: &'gr Graph,
            ) -> Result<(String, [u32; 3]), GosonnxError> {
                let (compiled, wg) = match self {
                    $(
                        OpType::$variant { attr } => {
                            self._compile(attr, shader_source, op, graph)?
                        },
                    )+
                    OpType::Unknown => {
                        return Err(Error(format!("Op `{:?}` is unsupported yet", op.op_type)));
                    }
                };
                Ok((compiled, wg))
            }

            pub fn activable(&self)->bool {
                match self {
                    $(
                        OpType::$variant { attr } => {
                            attr.activable()
                        },
                    )+
                    _ => false
                }
            }
        }

        impl fmt::Display for OpType {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(
                        OpType::$variant { .. } => write!(f, stringify!($variant)),
                    )+
                    OpType::Unknown => write!(f, "Unknown"),
                }
            }
        }
    };
}
