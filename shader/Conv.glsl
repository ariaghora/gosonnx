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
    uint global_x = gl_GlobalInvocationID.x;
    uint global_y = gl_GlobalInvocationID.y;
    uint global_z = gl_GlobalInvocationID.z;
    uint batch = global_z / output_channels;
    uint channel = global_z % output_channels;

    {{Y_type}} result = 0.0;
    for(int c = 0; c < in_dim[1]; c++) {
        for(int kh = 0; kh < kernel_shape[0]; kh++) {
            for(int kw = 0; kw < kernel_shape[1]; kw++) {
                int hIndex = int(global_y) * strides[0] - pads[0] + kh * dilations[0];
                int wIndex = int(global_x) * strides[1] - pads[1] + kw * dilations[1];

                if(hIndex >= 0 && hIndex < in_dim[2] && wIndex >= 0 && wIndex < in_dim[3]) {
                    {{X_type}} inputValue = X[((batch * in_dim[1] + c) * in_dim[2] + hIndex) * in_dim[3] + wIndex];
                    {{W_type}} weightValue = W[((channel * in_dim[1] + c) * kernel_shape[0] + kh) * kernel_shape[1] + kw];
                    result += inputValue * weightValue;
                }
            }
        }
    }
    {% if use_bias %}
    result += B[channel];
    {% endif %}
    Y[((batch * output_channels + channel) * out_dim[2] + global_y) * out_dim[3] + global_x] = result;
}