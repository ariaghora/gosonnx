![](assets/logo.png)

Ngubrut sampe otakku GOSONNX

# Adding your operator

- `ops/sigmoid.rs`, define struct `SigmoidOp`
- `ops/mod.rs`: At OpType, add your op, e.g., `Sigmoid { attr: SigmoidOp }`
- `ops/mod.rs`: At OpType impl, at compile(), add to match arm
- update `from_node_proto`
- implement the shader `Sigmoid.glsl`