#version 450

layout(set = 0, binding = 0) buffer Left {
    {{in_type}} left[];
};
layout(set = 0, binding = 1) buffer Right {
    {{out_type}} right[];
};

{% if use_bias %}
layout(set = 0, binding = 2) buffer Bias {
    {{bias_type}} bias[];
};
layout(set = 0, binding = 3) buffer Output {
    {{in_type}} output[];
};
{% else %}
layout(set = 0, binding = 2) buffer Output {
    {{in_type}} output[];
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
        {{in_type}} sum = 0.0;
        for (uint i = 0u; i < k; ++i) {
            {{in_type}} a;
            {{out_type}} b;
            if (trans_a == 1) {
                // A transposed
                a = left[i * m + global_y];
            } else {
                a = left[global_y * k + i];
            }
            if (trans_b == 1) {
                // B transposed
                b = right[global_x * k + i];
            } else {
                b = right[i * n + global_x];
            }
            sum += a * b;
        }

        {% if use_bias %}
        {{bias_type}} bias_val = bias[global_y * uint(bias_w) + global_x % uint(bias_w)];
        {{out_type}} out_val = alpha * sum + beta * bias_val;
        {% else %}
        {{out_type}} out_val = alpha * sum;
        {% endif %}

        {{in_type}} in_val = out_val;

        {% if activation %}
            {% if activation == "Relu" %}
            out_val = max(0, in_val);
            {% endif %}
        {% endif %}

        output[global_y * n + global_x] = out_val;

    }
}
