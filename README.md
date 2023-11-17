![](assets/logo.png)

Ngubrut sampe otakku GOSONNX

# Preparation

Please run onnxsim on the model first.

# Adding your operator

- `ops/sigmoid.rs`, define struct `SigmoidOp`
- `ops/mod.rs`: At OpType, add your op, e.g., `Sigmoid { attr: SigmoidOp }`
- `ops/mod.rs`: At OpType impl, at compile(), add to match arm
- update `from_node_proto`
- implement the shader `Sigmoid.glsl`

> Gosonnx is an initiative that builds upon the ideas inspired by [WONNX](https://github.com/webonnx/wonnx).
It's a from-the-ground-up rewrite, aiming to explore alternative methodologies and optimizations for different scenarios.
While it adopts some aspects from WONNX, this project diverges in its approach and implementation.
It is designed as a complementary exploration rather than a replacement, reflecting a personal interpretation and experimentation with the concepts.
All due credit is given to WONNX maintainers for the original work that sparked this direction.