{% extends "_unary_elementwise" %}

{% block implementation %}
    {% if min_val %}
    output = max( input, {{output_type}}({{min_val}}) );
    {% else %}
    output = {{input_type}}(input);
    {% endif %}

    {% if max_val %}
    output = min( output, {{output_type}}({{max_val}}) );
    {% endif %}
{% endblock implementation %}