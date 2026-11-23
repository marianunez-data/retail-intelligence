-- Per-customer behavior BEFORE the cutoff date (the feature window).
with tx as (

    select * from {{ ref('stg_transactions') }}
    where invoiced_at < date '{{ var("cutoff_date") }}'

),

orders as (

    -- one row per order (non-cancelled invoice)
    select customer_id, invoice_id, min(invoiced_at) as ordered_at
    from tx
    where not is_cancellation
    group by customer_id, invoice_id

),

order_stats as (

    select
        customer_id,
        count(*)                                                       as frequency,
        min(ordered_at)                                                as first_order_at,
        max(ordered_at)                                                as last_order_at,
        count(*) filter (where ordered_at >= date '{{ var("cutoff_date") }}' - interval 90 day) as orders_last_90d
    from orders
    group by customer_id

),

sales as (

    select
        customer_id,
        round(sum(revenue), 2)                                         as monetary,
        count(distinct product_id)                                     as unique_products,
        max(case when country = 'United Kingdom' then 1 else 0 end)    as is_uk
    from tx
    where not is_cancellation
    group by customer_id

),

returns as (

    select customer_id, count(distinct invoice_id) as n_returns
    from tx
    where is_cancellation
    group by customer_id

)

select
    o.*,
    s.monetary,
    s.unique_products,
    s.is_uk,
    coalesce(r.n_returns, 0) as n_returns
from order_stats o
join sales s using (customer_id)
left join returns r using (customer_id)
