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
    uint global_id = gl_GlobalInvocationID.x + 
                     gl_GlobalInvocationID.y * gl_WorkGroupSize.x + 
                     gl_GlobalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    if(global_id >= out_dim[0] * out_dim[1]) return;  // Boundary check
    
    uint out_index = global_id;
    
    // Convert the flattened index to the original dimensions
    uint indices[{{in_ndim}}];
    for(int i = 0; i < {{in_ndim}}; i++) {
        if(i < axis) {
            indices[i] = 0;  // Dimensions before 'axis' aren't collapsed
        } else {
            indices[i] = out_index % in_dim[i];
            out_index /= in_dim[i];
        }
    }

    uint in_index = indices[0];
    for(int i = 1; i < {{in_ndim}}; i++) {
        in_index = in_index * in_dim[i] + indices[i];
    }

    Y[global_id] = X[in_index]; 
}