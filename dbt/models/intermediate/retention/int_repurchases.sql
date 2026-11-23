-- Customers who placed an order in the label window AFTER the cutoff date.
select distinct customer_id
from {{ ref('stg_transactions') }}
where not is_cancellation
  and invoiced_at >= date '{{ var("cutoff_date") }}'
  and invoiced_at <  date '{{ var("cutoff_date") }}' + interval {{ var("horizon_days") }} day
