-- Fails if any order used for features is on or after the cutoff date.
select *
from {{ ref('int_customer_history') }}
where last_order_at >= date '{{ var("cutoff_date") }}'
