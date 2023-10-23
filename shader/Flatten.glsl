#version 450

layout(set = 0, binding = 0) buffer Input {
    {{X_type}} X[];
};

layout(set = 0, binding = 1) buffer Output {
    {{Y_type}} Y[];
};

const int axis = {{axis}};
const int in_dim[{{in_ndim}}] = int[{{in_ndim}}] ({{in_dim}});
const int out_dim[{{out_ndim}}] = int[{{out_ndim}}] ({{out_dim}});

layout(local_size_x = 32, local_size_y = 8, local_size_z=1) in;
void main() {
uint flat_idx = gl_GlobalInvocationID.x;
    if (flat_idx >= out_dim[0] * out_dim[1]) return; // Ensure we're within bounds

    int temp_idx = int(flat_idx);

    int idx[{{in_ndim}}];
    int prod = out_dim[1];
    for (int i = 0; i < axis; ++i) {
        idx[i] = 0;
    }

    for (int i = axis; i < {{in_ndim}}; ++i) {
        prod /= in_dim[i];
        idx[i] = temp_idx / prod;
        temp_idx %= prod;
    }

    // Compute linear index for input tensor
    int in_linear_idx = 0;
    prod = 1;
    for (int i = {{in_ndim}} - 1; i >= 0; --i) {
        in_linear_idx += idx[i] * prod;
        prod *= in_dim[i];
    }

    Y[flat_idx] = X[in_linear_idx];
}