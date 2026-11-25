{{ config(materialized='external', location='../data/processed/product_weekly_demand.parquet', format='parquet') }}

-- Complete weekly demand grid for the forecasting universe: products sold in at
-- least var('min_active_weeks') weeks. Weeks without sales are explicit zeros,
-- which a forecasting model needs to learn intermittent demand.
with weeks as (

    -- every calendar week between the first and last complete week, including
    -- weeks when the shop was closed (Christmas), so lags line up across years
    select cast(unnest(generate_series(min(week_start), max(week_start), interval 7 day)) as date) as week_start
    from {{ ref('int_product_weeks') }}

),

universe as (

    select product_id
    from {{ ref('int_product_weeks') }}
    group by 1
    having count(*) >= {{ var('min_active_weeks') }}

),

grid as (

    select u.product_id, w.week_start
    from universe u cross join weeks w

)

select
    g.product_id,
    g.week_start,
    coalesce(p.units, 0)        as units,
    coalesce(p.revenue, 0)      as revenue,
    coalesce(p.orders, 0)       as orders,
    coalesce(p.customers, 0)    as customers,
    coalesce(p.guest_units, 0)  as guest_units,
    p.avg_unit_price,
    p.list_price
from grid g
left join {{ ref('int_product_weeks') }} p using (product_id, week_start)
order by g.product_id, g.week_start
