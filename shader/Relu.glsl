{% extends "_unary_elementwise" %}

{% block implementation %}
    output = max(input, 0.0);
{% endblock implementation %}