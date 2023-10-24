#version 450

layout(set = 0, binding = 0) buffer Input {
    {{X_type}} X[];
};
layout(set = 0, binding = 1) buffer Weight {
    {{W_type}} W[];
};

{% if use_bias %}
layout(set = 0, binding = 2) buffer Bias {
    {{B_type}} B[];
};
layout(set = 0, binding = 3) buffer Output {
    {{Y_type}} Y[];
};
{% else %}
layout(set = 0, binding = 2) buffer Output {
    {{Y_type}} Y[];
};
{% endif %}


const int in_dim[4] = int[4] ({{in_dim}});
const int out_dim[4] = int[4] ({{out_dim}});
const int weight_dim[4] = int[4] ({{weight_dim}});

const int dilations[2] = int[2] ({{dilations}});
const int group = {{group}};
const int kernel_shape[2] = int[2] ({{kernel_shape}});
const int pads[4] = int[4] ({{pads}});
const int strides[2] = int[2] ({{strides}});
const int output_channels = {{output_channels}};

int get_input_pos(int n, int x, int y, int c) {
    return n * in_dim[1] * in_dim[2] * in_dim[3] + c * in_dim[2] * in_dim[3] + y * in_dim[3] + x;
}

int get_output_pos(int n, int x, int y, int c) {
    return n * out_dim[1] * out_dim[2] * out_dim[3] + c * out_dim[2] * out_dim[3] + y * out_dim[3] + x;
}

int get_kernel_pos(int x, int y, int ic, int oc) {
    return oc * in_dim[1] * kernel_shape[0] * kernel_shape[1] + ic * kernel_shape[0] * kernel_shape[1] + y * kernel_shape[1] + x;
}

layout(local_size_x = 16, local_size_y = 16, local_size_z=1) in;
void main() {
    int global_x = int(gl_GlobalInvocationID.x);
    int global_y = int(gl_GlobalInvocationID.y);
    int global_z = int(gl_GlobalInvocationID.z);
    int output_height = (in_dim[2] + 2 * pads[0] - kernel_shape[0]) / strides[0] + 1;
    int output_width = (in_dim[3] + 2 * pads[1] - kernel_shape[1]) / strides[1] + 1;

    if (global_x < output_width && global_y < output_height && global_z < out_dim[0]) {
        for (int oc = 0; oc < output_channels; oc++) {
            int out_idx = get_output_pos(global_z, global_x, global_y, oc);
            {% if use_bias %}
            Y[out_idx] = B[oc];
            {% else %}
            Y[out_idx] = 0.0;
            {% endif %}

            for (int ic = 0; ic < in_dim[1]; ic++) {
                for (int ky = 0; ky < kernel_shape[0]; ky++) {
                    for (int kx = 0; kx < kernel_shape[1]; kx++) {
                        int in_x = global_x * strides[0] - pads[0] + kx;
                        int in_y = global_y * strides[1] - pads[1] + ky;

                        if (in_x >= 0 && in_x < in_dim[3] && in_y >= 0 && in_y < in_dim[2]) {
                            int in_idx = get_input_pos(global_z, in_x, in_y, ic);
                            int k_idx = get_kernel_pos(kx, ky, ic, oc);
                            Y[out_idx] += X[in_idx] * W[k_idx];
                        }
                    }
                }
            }
        }
    }
}
