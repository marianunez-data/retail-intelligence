-- Model-ready table: one row per customer, label + features, for one snapshot.
-- The output file name is a variable so several snapshots can be built.
{{ config(materialized='external',
          location='../data/processed/' ~ var('output_name') ~ '.csv',
          format='csv') }}

with history as (

    select * from {{ ref('int_customer_history') }}
    where monetary > 0

),

buyers as (

    select * from {{ ref('int_repurchases') }}

)

select
    h.customer_id                                                            as "CustomerID",
    date '{{ var("cutoff_date") }}'                                          as cutoff_date,
    case when b.customer_id is not null then 1 else 0 end                    as label,
    date_diff('day', h.last_order_at, date '{{ var("cutoff_date") }}')       as recency_days,
    h.frequency,
    h.monetary,
    round(h.monetary / h.frequency, 2)                                       as avg_order_value,
    h.unique_products,
    date_diff('day', h.first_order_at, date '{{ var("cutoff_date") }}')      as tenure_days,
    round(h.n_returns * 1.0 / h.frequency, 3)                                as return_rate,
    h.orders_last_90d,
    case when h.frequency > 1
         then round(date_diff('day', h.first_order_at, h.last_order_at) * 1.0 / (h.frequency - 1), 1)
    end                                                                      as avg_days_between_orders,
    h.is_uk

from history h
left join buyers b using (customer_id)
