<p style="text-align: center; font-size: 40px">GOSONNX</p>

<p style="text-align: center">An ONNX run-time engine with GPU acceleration, powered by WebGPU</p>

![](assets/logo.png)

# Preparation

Please run onnxsim on the model first.

# Adding your operator

## Activation function

- Prefix ALL attributes with op type. For example, in HardSigmoid activation op, there are `alpha` and `beta` attributes, we need to implement them as `HardSigmoid_alpha` and `HardSigmoid_beta`.
    This will avoid attribute collision during graph optimization since the activation function shader code will be inserted in other op shader code that may have the same attribute name (e.g., Gemm has `alpha` and `beta` too). 

`TODO`

## 

---

> Gosonnx is an initiative that builds upon the ideas inspired by [WONNX](https://github.com/webonnx/wonnx).
> It's a from-the-ground-up rewrite, aiming to explore alternative methodologies and optimizations for different scenarios.
> While it adopts some aspects from WONNX, this project diverges in its approach and implementation.
> It is designed as a complementary exploration rather than a replacement, reflecting a personal interpretation and experimentation with the concepts.
> All due credit is given to WONNX maintainers for the original work that sparked this direction.