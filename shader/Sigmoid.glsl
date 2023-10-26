{% extends "_unary_elementwise" %}

{% block implementation %}
    output = {{output_type}}(1) / ( {{output_type}}(1) + exp(-input) );
{% endblock implementation %}