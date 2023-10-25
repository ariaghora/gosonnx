#version 450

layout(set = 0, binding = 0) buffer Input {
    {{input_1_type}} input_1_buf[];
};
layout(set = 0, binding = 1) buffer Input {
    {{input_2_type}} input_2_buf[];
};
layout(set = 0, binding = 2) buffer Output {
    {{output_type}} output_buf[];
};

{% block definition %}
// will be filled with templates that extend this 
{% endblock definition %}

layout(local_size_x = 256) in;
void main() {
    {{output_type}} output;

    uint idx = gl_GlobalInvocationID.x;

    {{input_1_type}} left = input_1_buf[idx];
    {{input_2_type}} right = input_2_buf[idx];

    {% block implementation %}
    // will be filled with templates that extend this. For example:
    // 
    // ```
    // output = max(val, 0.0);
    // ```
    {% endblock implementation %}
    
    output_buf[idx] = output;
}
