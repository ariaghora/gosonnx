{% extends "_unary_elementwise" %}

{% block implementation %}
    {% if use_min %}
    output = max( input, {{output_type}}({{min_val}}) );
    {% else %}
    output = {{input_type}}(input);
    {% endif %}

    {% if use_max %}
    output = min( output, {{output_type}}({{max_val}}) );
    {% endif %}
{% endblock implementation %}