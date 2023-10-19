#version 450

layout(set = 0, binding = 0) buffer Left {
    {{a_type}} left[];
};
layout(set = 0, binding = 1) buffer Right {
    {{b_type}} right[];
};

{% if use_bias %}
layout(set = 0, binding = 2) buffer Bias {
    {{bias_type}} bias[];
};
layout(set = 0, binding = 3) buffer Output {
    {{a_type}} output[];
};
{% else %}
layout(set = 0, binding = 2) buffer Output {
    {{a_type}} output[];
};
{% endif %}

const float alpha = {{alpha}};
const float beta = {{beta}};
const int trans_a = {{trans_a}};
const int trans_b = {{trans_b}};

const uint m = {{m}};
const uint k = {{k}};
const uint n = {{n}};

{% if use_bias %}
const uint bias_h = {{bias_h}};
const uint bias_w = {{bias_w}};
{% endif %}

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    uint global_x = gl_GlobalInvocationID.x;
    uint global_y = gl_GlobalInvocationID.y;

    if (global_x < n && global_y < m) {
        {{a_type}} sum = 0.0;
        for (uint i = 0u; i < k; ++i) {
            {{a_type}} a;
            {{b_type}} b;
            if ({{trans_a}} == 1) {
                a = left[i * m + global_y];
            } else {
                a = left[global_y * k + i];
            }
            if ({{trans_b}} == 1) {
                b = right[global_x * k + i];
            } else {
                b = right[i * n + global_x];
            }
            sum += a * b;
        }

        {% if use_bias %}
        {{bias_type}} bias_val = bias[global_y * uint(bias_w) + global_x % uint(bias_w)];
        output[global_y * n + global_x] = alpha * sum + beta * bias_val;
        {% else %}
        output[global_y * n + global_x] = alpha * sum;
        {% endif %}
    }
}
