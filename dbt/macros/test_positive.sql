{% test positive(model, column_name) %}
-- Fails for any value that is not strictly positive.
select * from {{ model }} where {{ column_name }} <= 0
{% endtest %}
