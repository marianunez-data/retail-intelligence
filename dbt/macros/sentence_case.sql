{% macro sentence_case(column) -%}
upper(left(trim({{ column }}), 1)) || lower(substr(trim({{ column }}), 2))
{%- endmacro %}
