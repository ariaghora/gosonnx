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


{% if get_direct_strided_offset_l_fn %}
{{get_direct_strided_offset_l_fn}}
{% endif %}

{% if get_direct_strided_offset_r_fn %}
{{get_direct_strided_offset_r_fn}}
{% endif %}

{% if common_shape %}
const int common_shape[{{common_shape_len}}] = int [{{common_shape_len}}]({{common_shape}});
{% endif %}

{% block definition %}
// will be filled with templates that extend this 
{% endblock definition %}

layout(local_size_x = 256) in;
void main() {
    {{output_type}} output;

    uint idx = gl_GlobalInvocationID.x;

    {% if left_oneval %}
        {{input_1_type}} left = input_1_buf[0];
    {% else %}
        {% if left_logical_strides %}
            {{input_1_type}} left = input_1_buf[get_direct_strided_offset_l(idx)];
        {% else %}
            {{input_1_type}} left = input_1_buf[idx];
        {% endif %}
    {% endif %}

    {% if right_oneval %}
        {{input_2_type}} right = input_2_buf[0];
    {% else %}
        {% if right_logical_strides %}
            {{input_2_type}} right = input_2_buf[get_direct_strided_offset_r(idx)];
        {% else %}
            {{input_2_type}} right = input_2_buf[idx];
        {% endif %}
    {% endif %}

    {% block implementation %}
    // will be filled with templates that extend this. For example:
    // 
    // ```
    // output = left + right;
    // ```
    {% endblock implementation %}
    
    output_buf[idx] = output;
}
