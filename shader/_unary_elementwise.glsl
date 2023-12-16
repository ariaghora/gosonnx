#version 450

layout(set = 0, binding = 0) buffer Input {
    {{input_type}} input_buf[];
};
layout(set = 0, binding = 1) buffer Output {
    {{output_type}} output_buf[];
};

{% block definition %}
// will be filled with templates that extend this 
{% endblock definition %}

layout(local_size_x = 256) in;
void main() {
    {{output_type}} output_val;

    uint idx = gl_GlobalInvocationID.x;
    if (idx > {{output_len}}) {
        return;
    }

    {{input_type}} input_val = input_buf[idx];

    {% block implementation %}
    // will be filled with templates that extend this. For example:
    // 
    // ```
    // output = max(val, 0.0);
    // ```
    {% endblock implementation %}

    {% include "_activation" %}

    output_buf[idx] = output_val;
}
