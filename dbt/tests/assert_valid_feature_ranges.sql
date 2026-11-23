-- Fails if any feature is outside its valid range.
select *
from {{ ref('customer_features') }}
where recency_days < 0
   or tenure_days < recency_days
   or frequency < 1
   or orders_last_90d > frequency
   or return_rate < 0
   or (frequency = 1 and avg_days_between_orders is not null)
