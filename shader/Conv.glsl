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

layout(local_size_x = 16, local_size_y = 16, local_size_z=1) in;
void main() {
int n = int(gl_GlobalInvocationID.x / (out_dim[1] * out_dim[2] * out_dim[3]));
    int c = int((gl_GlobalInvocationID.x % (out_dim[1] * out_dim[2] * out_dim[3])) / (out_dim[2] * out_dim[3]));
    int h = int((gl_GlobalInvocationID.x % (out_dim[2] * out_dim[3])) / out_dim[3]);
    int w = int(gl_GlobalInvocationID.x % out_dim[3]);

    if (h < out_dim[2] && w < out_dim[3]) {
        int out_idx = n * out_dim[1] * out_dim[2] * out_dim[3] + c * out_dim[2] * out_dim[3] + h * out_dim[3] + w;
        {% if use_bias %}
        Y[out_idx] = B[c];
        {% else %}
        Y[out_idx] = 0.0;
        {% endif %}

        int input_channels_per_group = in_dim[1] / group;
        int output_channels_per_group = output_channels / group;

        int group_idx = c / output_channels_per_group;
        int offset_within_group = c % output_channels_per_group;

        for (int i = 0; i < kernel_shape[0]; i++) {
            for (int j = 0; j < kernel_shape[1]; j++) {
                for (int k = 0; k < input_channels_per_group; k++) {
                    int h_in = h * strides[0] - pads[0] + i;
                    int w_in = w * strides[1] - pads[1] + j;

                    if (h_in >= 0 && h_in < in_dim[2] && w_in >= 0 && w_in < in_dim[3]) {
                        int in_idx = n * in_dim[1] * in_dim[2] * in_dim[3] + (group_idx * input_channels_per_group + k) * in_dim[2] * in_dim[3] + h_in * in_dim[3] + w_in;
                        int k_idx = (group_idx * output_channels_per_group + offset_within_group) * input_channels_per_group * kernel_shape[0] * kernel_shape[1] + k * kernel_shape[0] * kernel_shape[1] + i * kernel_shape[1] + j;

                        Y[out_idx] += X[in_idx] * W[k_idx];
                    }
                }
            }
        }
    }
}