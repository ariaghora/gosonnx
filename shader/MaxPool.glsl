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
    uint global_x = gl_GlobalInvocationID.x;
    uint global_y = gl_GlobalInvocationID.y;
    uint global_z = gl_GlobalInvocationID.z;

    uint N = int(global_z / out_dim[1]); // Calculate batch index
    uint C = global_z % out_dim[1]; // Calculate channel index

    {{Y_type}} max_val = -1.0e10; // Starting with a very low value

    // Iterate over the pooling window in the input tensor
    for(int kh = 0; kh < kernel_shape[0]; kh++) {
        for(int kw = 0; kw < kernel_shape[1]; kw++) {
            int hIndex = int(global_y) * strides[0] - pads[0] + kh;
            int wIndex = int(global_x) * strides[1] - pads[1] + kw;

            if(hIndex >= 0 && hIndex < in_dim[2] && wIndex >= 0 && wIndex < in_dim[3]) {
                {{X_type}} inputValue = X[((N * in_dim[1] + C) * in_dim[2] + hIndex) * in_dim[3] + wIndex];
                max_val = max(max_val, inputValue);
            }
        }
    }

    Y[(global_z * out_dim[2] + global_y) * out_dim[3] + global_x] = max_val;
}
