#version 450

layout(set = 0, binding = 0) buffer Input {
    {{input_type}} input_buf[];
};

layout(set = 0, binding = 1) buffer Scale {
    {{scale_type}} scale_buf[];
};

layout(set = 0, binding = 2) buffer B {
    {{b_type}} b_buf[];
};

layout(set = 0, binding = 3) buffer Mean {
    {{mean_type}} mean_buf[];
};

layout(set = 0, binding = 4) buffer Var {
    {{var_type}} var_buf[];
};

layout(set = 0, binding = 5) buffer Output {
    {{output_type}} output_buf[];
};

const int in_dim[4] = int[4]({{in_dim}});
const int out_dim[4] = int[4]({{out_dim}});

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    uint gid_x = gl_GlobalInvocationID.x; // For N
    uint gid_y = gl_GlobalInvocationID.y; // For C
    uint gid_z = gl_GlobalInvocationID.z; // For H, W
    float epsilon = {{epsilon}}; // Small constant for numerical stability

    if (gid_x >= out_dim[0] || gid_y >= out_dim[1]) return;

    uint baseIndex = gid_x * in_dim[1] * in_dim[2] * in_dim[3] + gid_y * in_dim[2] * in_dim[3];

    for (int h = 0; h < in_dim[2]; ++h) {
        for (int w = 0; w < in_dim[3]; ++w) {
            uint index = baseIndex + h * in_dim[3] + w;
            float inputValue = input_buf[index];
            float scaleValue = scale_buf[gid_y];
            float bValue = b_buf[gid_y];
            float meanValue = mean_buf[gid_y];
            float varValue = var_buf[gid_y];

            // Batch normalization formula
            float normalized = scaleValue * (inputValue - meanValue) / sqrt(varValue + epsilon) + bValue;

            // Writing to output buffer
            output_buf[index] = normalized;
        }
    }
}
