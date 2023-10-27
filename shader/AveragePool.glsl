#version 450

layout(set = 0, binding = 0) buffer Input {
    {{X_type}} X[];
};

layout(set = 0, binding = 1) buffer Output {
    {{Y_type}} Y[];
};

const int in_dim[4] = int[4]({{in_dim}});
const int out_dim[4] = int[4]({{out_dim}});
const int kernel_shape[2] = int[2]({{kernel_shape}});
const int pads[4] = int[4]({{pads}});
const int strides[2] = int[2]({{strides}});
const int ceil_mode = {{ceil_mode}};

layout(local_size_x = 16, local_size_y = 4, local_size_z=4) in;
void main() {
    uint W = gl_GlobalInvocationID.x;
    uint H = gl_GlobalInvocationID.y;
    uint C = gl_GlobalInvocationID.z % out_dim[1];
    uint N = gl_GlobalInvocationID.z / out_dim[1];

    if (N >= out_dim[0] || C >= out_dim[1] || H >= out_dim[2] || W >= out_dim[3]) return;

    {{Y_type}} sum_val = 0;
    int count = 0;

    for (int kh = 0; kh < kernel_shape[0]; ++kh) {
        for (int kw = 0; kw < kernel_shape[1]; ++kw) {
            uint in_h = ceil_mode == 1 ? uint(ceil(float(H * strides[0] + kh - pads[0]))) : H * strides[0] + kh - pads[0];
            uint in_w = ceil_mode == 1 ? uint(ceil(float(W * strides[1] + kw - pads[1]))) : W * strides[1] + kw - pads[1];

            if (in_h < in_dim[2] && in_w < in_dim[3]) {
                uint index = N * in_dim[1] * in_dim[2] * in_dim[3] + C * in_dim[2] * in_dim[3] + in_h * in_dim[3] + in_w;
                sum_val += X[index];
                count++;
            }
        }
    }

    {{Y_type}} avg_val = sum_val / float(count);

    uint out_index = N * out_dim[1] * out_dim[2] * out_dim[3] + C * out_dim[2] * out_dim[3] + H * out_dim[3] + W;
    Y[out_index] = avg_val;
}
