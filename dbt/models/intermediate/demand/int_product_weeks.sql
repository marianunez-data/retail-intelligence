-- Units sold per product and calendar week (weeks start on Monday), all channels.
-- Only complete weeks are kept: the first and last weeks of the extract are partial.
with sales as (

    select *, cast(date_trunc('week', invoiced_at) as date) as week_start
    from {{ ref('stg_sales_lines') }}
    where not is_cancellation and quantity > 0

),

bounds as (

    select min(invoiced_at)::date as first_day, max(invoiced_at)::date as last_day from sales

),

-- List price: the highest registered-channel price that accounts for at least
-- 10% of a product's lines in that week. Lower prices are volume tiers;
-- guest (retail) prices are about twice the wholesale list.
price_points as (

    select product_id, week_start, unit_price, count(*) as n,
           count(*) / sum(count(*)) over (partition by product_id, week_start) as share
    from sales
    where channel = 'Registered'
    group by 1, 2, 3

),

list_price as (

    select product_id, week_start, max(unit_price) as list_price
    from price_points
    where share >= 0.10
    group by 1, 2

)

select
    s.product_id,
    s.week_start,
    sum(s.quantity)                                                as units,
    round(sum(s.revenue), 2)                                       as revenue,
    count(distinct s.invoice_id)                                   as orders,
    count(distinct s.customer_id)                                  as customers,
    sum(case when s.channel = 'Guest' then s.quantity else 0 end)  as guest_units,
    round(sum(s.revenue) / sum(s.quantity), 4)                     as avg_unit_price,
    lp.list_price
from sales s
cross join bounds b
left join list_price lp using (product_id, week_start)
where s.week_start >= b.first_day                                  -- drop the partial first week
  and s.week_start + interval 6 day <= b.last_day                  -- drop the partial last week
group by s.product_id, s.week_start, lp.list_price
