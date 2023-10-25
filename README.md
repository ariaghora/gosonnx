![](assets/logo.png)

Ngubrut sampe otakku GOSONNX

# Adding your operator

- `ops/sigmoid.rs`, define struct `SigmoidOp`
- `ops/mod.rs`: At OpType, add your op, e.g., `Sigmoid { attr: SigmoidOp }`