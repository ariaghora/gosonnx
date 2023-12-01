// This is the base template for activation function ops

#version 450

layout(set = 0, binding = 0) buffer Input {
    {{input_type}} input_buf[];
};
layout(set = 0, binding = 1) buffer Output {
    {{output_type}} output_buf[];
};


layout(local_size_x = 256) in;
void main() {
    {{output_type}} output_val;

    uint idx = gl_GlobalInvocationID.x;
    {{input_type}} input_val = input_buf[idx];

    {% include "_activation_def" %}

    output_buf[idx] = output_val;
}

