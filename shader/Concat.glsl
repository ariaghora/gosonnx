#version 450

const int max_dims = {{output_n_dim}}; 

{% for input in input_info_arr %}
layout(set = 0, binding = {{loop.index0}}) buffer Input_{{loop.index0}} {
    {{input.dtype}} input_{{loop.index0}}[];
};
const int input_{{loop.index0}}_shape[max_dims]   = int[{{output_n_dim}}]({{input.shape_csv}});
const int input_{{loop.index0}}_strides[max_dims] = int[{{output_n_dim}}]({{input.strides_csv}});
{% endfor %}

layout(set = 0, binding = {{output_binding_no}}) buffer Output {
    {{output_dtype}} output[];
};
const int output_shape[{{output_n_dim}}] = int[{{output_n_dim}}]({{output_shape_csv}});
const int output_strides[{{output_n_dim}}] = int[{{output_n_dim}}]({{output_strides_csv}});

const int num_inputs = {{n_inputs}}; 
const int concat_axis = {{concat_axis}}; 

int get_flat_index(
    {% for i in range(end=output_n_dim) %}
    const int pos_{{ i }},
    const int strides_{{ i }}{% if not loop.last %}, {% endif %}
    {% endfor %}
) {
    return {% for i in range(end=output_n_dim) %}
            pos_{{ i }} * strides_{{ i }}{% if not loop.last %} + {% endif %}
            {% endfor %};
}

{% for i in range(end=n_inputs) %}
void copy_from_input_{{i}}(int src_index, int dest_index) {
  output[dest_index] = input_{{i}}[src_index];
}
{% endfor %}

int get_src_index(int input_num, int pos[max_dims]) {
    int index = 0;
    switch (input_num) {
        {% for i in range(end=n_inputs) %}
        case {{i}}:
            index = get_flat_index(
                {% for j in range(end=output_n_dim) %}
                pos[{{j}}],
                input_{{i}}_strides[{{j}}] {% if not loop.last %},{% endif %}
                {% endfor %}
            );
            break;
        {% endfor %}
    }
    return index;
}

layout(local_size_x = 256, local_size_y = 1, local_size_z=1) in;
void main() {
    int dest_index = int(gl_GlobalInvocationID.x);
    int pos[max_dims];
    int rem = dest_index;
    for (int i = 0; i < max_dims; ++i) {
        pos[i] = rem / output_strides[i];
        rem = rem % output_strides[i];
    }

    int current_input = -1;
    int concat_offset = 0;

    for (int i = 0; i < num_inputs; ++i) {
        int shape;
        switch (i) {
            {% for i in range(end=n_inputs) %}
            case {{i}}: shape = input_{{i}}_shape[concat_axis]; break;
            {% endfor %}
        }
        if (pos[concat_axis] < concat_offset + shape) {
            pos[concat_axis] -= concat_offset;
            current_input = i;
            break;
        }
        concat_offset += shape;
    }

    if (current_input == -1) {
        return;
    }

    int src_index = get_src_index(current_input, pos);

    {{output_dtype}} value = 0.0;
    switch (current_input) {
        {% for i in range(end=n_inputs) %}
        case {{i}}: value = input_{{i}}[src_index]; break;
        {% endfor %}
    }

    output[dest_index] = value;
}
