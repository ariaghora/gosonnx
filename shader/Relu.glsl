#version 450

layout(set = 0, binding = 0) buffer Input {
    float input[];
};
layout(set = 0, binding = 1) buffer Output {
    float output[];
};

layout(local_size_x = 256) in;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    float val = input[idx];
    output[idx] = max(val, 0.0);
}
