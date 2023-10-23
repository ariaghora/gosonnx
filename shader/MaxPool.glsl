#version 450

layout(set = 0, binding = 0) buffer Input {
    {{X_type}} X[];
};

layout(set = 0, binding = 1) buffer Output {
    {{Y_type}} Y[];
};

const int in_dim[4] = int[4] ({{in_dim}});
const int out_dim[4] = int[4] ({{out_dim}});

const int ceil_mode = {{ceil_mode}};
const int kernel_shape[2] = int[2] ({{kernel_shape}});
const int pads[4] = int[4] ({{pads}});
const int strides[2] = int[2] ({{strides}});

layout(local_size_x = 16, local_size_y = 16, local_size_z=1) in;
void main() {
int n = int(gl_GlobalInvocationID.x / (out_dim[1] * out_dim[2] * out_dim[3]));
    int c = int((gl_GlobalInvocationID.x % (out_dim[1] * out_dim[2] * out_dim[3])) / (out_dim[2] * out_dim[3]));
    int h = int((gl_GlobalInvocationID.x % (out_dim[2] * out_dim[3])) / out_dim[3]);
    int w = int(gl_GlobalInvocationID.x % out_dim[3]);

    if (h < out_dim[2] && w < out_dim[3]) {
        int out_idx = n * out_dim[1] * out_dim[2] * out_dim[3] + c * out_dim[2] * out_dim[3] + h * out_dim[3] + w;
        {{X_type}} max_val = -1.0/0.0; // Negative infinity

        for (int i = 0; i < kernel_shape[0]; i++) {
            for (int j = 0; j < kernel_shape[1]; j++) {
                int h_in;
                int w_in;

                if (ceil_mode == 1) {
                    h_in = int(ceil(float(h) * strides[0] - pads[0] + i));
                    w_in = int(ceil(float(w) * strides[1] - pads[1] + j));
                } else {
                    h_in = h * strides[0] - pads[0] + i;
                    w_in = w * strides[1] - pads[1] + j;
                }

                if (h_in >= 0 && h_in < in_dim[2] && w_in >= 0 && w_in < in_dim[3]) {
                    int in_idx = n * in_dim[1] * in_dim[2] * in_dim[3] + c * in_dim[2] * in_dim[3] + h_in * in_dim[3] + w_in;
                    max_val = max(max_val, X[in_idx]);
                }
            }
        }

        Y[out_idx] = max_val;
    }
}
