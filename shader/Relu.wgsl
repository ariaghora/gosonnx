@group(0) @binding(0)
var<storage, read_write> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let val = input[idx];
    output[idx] = max(val, 0.0);
}