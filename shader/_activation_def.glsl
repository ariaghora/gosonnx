// This template is where all necessary activation functions are defined

{% if activation %}

    {% if activation == "HardSigmoid" %}
        {% if HardSigmoid_alpha %}
            {{input_type}} HardSigmoid_alpha = {{input_type}}( {{HardSigmoid_alpha}} );
        {% else %}
            {{input_type}} HardSigmoid_alpha = {{input_type}}( 0.2 );
        {% endif %}

        {% if HardSigmoid_beta %}
            {{input_type}} HardSigmoid_beta = {{input_type}}( {{HardSigmoid_beta}} );
        {% else %}
            {{input_type}} HardSigmoid_beta = {{input_type}}( 0.5 );
        {% endif %}

        output_val = max(
            {{input_type}}(0),
            min({{input_type}}(1), HardSigmoid_alpha * input_val + HardSigmoid_beta)
        );
    {% endif %}


    {% if activation == "Relu" %}
        output_val = max({{input_type}}(0), input_val);
    {% endif %}


    {% if activation == "Sigmoid" %}
        output_val = {{output_type}}(1) / ( {{output_type}}(1) + exp(-input_val) );
    {% endif %}

{% endif %}