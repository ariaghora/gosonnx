#version 450

layout(set = 0, binding = 0) buffer Input {
    {{input_type}} input_buf[];
};

layout(set = 0, binding = 1) buffer MinVal {
    {{min_val_type}} min_val_buf[];
};

layout(set = 0, binding = 2) buffer MaxVal {
    {{max_val_type}} max_val_buf[];
};

layout(set = 0, binding = 3) buffer Output {
    {{output_type}} output_buf[];
};

layout(local_size_x = 256) in;
void main() {
    {{min_val_type}} min_val = min_val_buf[0];
    {{max_val_type}} max_val = max_val_buf[0];

    {{output_type}} output;

    uint idx = gl_GlobalInvocationID.x;
    {{input_type}} input = input_buf[idx];

    output = max( input, {{output_type}}(min_val) );
    output = min( output, {{output_type}}(max_val) );

    output_buf[idx] = output;
}
