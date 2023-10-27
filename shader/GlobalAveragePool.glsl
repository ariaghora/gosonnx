#version 450

layout(set = 0, binding = 0) buffer Input {
    {{X_type}} X[];
};

layout(set = 0, binding = 1) buffer Output {
    {{Y_type}} Y[];
};

const int in_dim[4] = int[4]({{in_dim}});
const int out_dim[4] = int[4]({{out_dim}});

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    uint gid_x = gl_GlobalInvocationID.x; // for N
    uint gid_y = gl_GlobalInvocationID.y; // for C
    uint gid_z = gl_GlobalInvocationID.z; // reserved for future use, or other tasks
    
    if (gid_x >= out_dim[0] || gid_y >= out_dim[1]) return;
  
    float sum = 0.0;
    for (int h = 0; h < in_dim[2]; ++h) {
        for (int w = 0; w < in_dim[3]; ++w) {
            uint index = gid_x * in_dim[1] * in_dim[2] * in_dim[3] + gid_y * in_dim[2] * in_dim[3] + h * in_dim[3] + w;
            sum += X[index];
        }
    }
  
    float avg = sum / float(in_dim[2] * in_dim[3]);
  
    uint out_index = gid_x * out_dim[1] + gid_y;
    Y[out_index] = avg;
}
