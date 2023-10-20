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
    {{A_type}} Y[];
};
{% else %}
layout(set = 0, binding = 2) buffer Output {
    {{A_type}} Y[];
};
{% endif %}

const int dilations[2] = int[2] (1, 1);


layout(local_size_x = 16, local_size_y = 16, local_size_z) in;
void main() {
}