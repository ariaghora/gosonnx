{% extends "_unary_elementwise" %}

{% block implementation %}
    {% if alpha %}
        {{ input_type }} alpha = {{ input_type }}( {{alpha}} );
    {% else %}
        {{ input_type }} alpha = {{ input_type }}( 0.2 );
    {% endif %}

    {% if beta %}
        {{ input_type }} beta = {{ input_type }}( {{beta}} );
    {% else %}
        {{ input_type }} beta = {{ input_type }}( 0.5 );
    {% endif %}

    output = max(
        {{input_type}}(0),
        min({{input_type}}(1), alpha * input + beta)
    );
{% endblock implementation %}