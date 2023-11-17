#version 450

{% for input in input_info_arr %}
layout(set = 0, binding = {{loop.index0}}) buffer Input_{{loop.index0}} {
{{input.dtype}} input_{{loop.index0}}[];
};
{% endfor %}
layout(set = 0, binding = {{output_binding_no}}) buffer Output {
{{output_dtype}} Y[];
};

{% if sizes %}
const int sizes[4];
{% endif %}

{% if scales %}
const float scales[{{scales_len}}] = float[{{scales_len}}]({{scales}});
{% endif %}

{% if axes %}
const int axes[{{axes_len}}] = int[{{axes_len}}]({{axes_csv}});
{% endif %}

const int in_dim[4] = int[4]({{in_dim}});
const int out_dim[4] = int[4]({{out_dim}});

// 0: nearest
// 1: linear
// 2: cubic
const int mode = {{mode}};

// 0: round_prefer_floor
// 1: round_prefer_ceil
// 2: floor
// 3: ceil
const int nearest_mode = {{nearest_mode}}; // only effective if mode = 0

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    uint gid_x = gl_GlobalInvocationID.x; // Generally used for W
    uint gid_y = gl_GlobalInvocationID.y; // Generally used for H

    if (gid_x >= out_dim[3]) return; // Check if gid_x exceeds the width

    // Calculate the channel and y-coordinate within the channel
    int channel = int(gid_y) / out_dim[2];
    int y_within_channel = int(gid_y) % out_dim[2];

    // Return if the channel or y-coordinate is out of bounds
    if (channel >= in_dim[1] || y_within_channel >= out_dim[2]) return;


    // Scales for each of the 4 dimensions
    {% if scales %}
    float scale_n = scales[0];
    float scale_c = scales[1];
    float scale_h = scales[2];
    float scale_w = scales[3];
    {% else %}
    float scale_n = float(in_dim[0]) / float(out_dim[0]);
    float scale_c = float(in_dim[1]) / float(out_dim[1]);
    float scale_h = float(in_dim[2]) / float(out_dim[2]);
    float scale_w = float(in_dim[3]) / float(out_dim[3]);
    {% endif %}

    // Calculate source coordinates for nearest neighbor
    int src_n, src_c, src_h, src_w;
    if (mode == 0) { // Nearest
        if (nearest_mode == 2) { // floor
            src_n = int(float(gid_y / (out_dim[2] * out_dim[3])) * scale_n);
            src_c = int(float((gid_y % (out_dim[2] * out_dim[3])) / out_dim[2]) * scale_c);
            src_h = int(float(gid_y % out_dim[2]) / scale_h); // Corrected calculation for H
            src_w = int(float(gid_x) / scale_w);

        } else {
            // Handle other nearest_mode cases here if needed
        }
    } else {
        // Handle other modes (linear, cubic) here if needed
    }

    // Bounds checking
    src_n = min(src_n, in_dim[0] - 1);
    src_c = min(src_c, in_dim[1] - 1);
    src_h = min(src_h, in_dim[2] - 1);
    src_w = min(src_w, in_dim[3] - 1);

    // Compute linear indices for flattened array
    uint src_index = src_n * in_dim[1] * in_dim[2] * in_dim[3] + src_c * in_dim[2] * in_dim[3] + src_h * in_dim[3] + src_w;
    uint out_index = gid_y * out_dim[3] + gid_x;

    // Assign value
    Y[out_index] = input_0[src_index];
}
