{{ config(materialized='external', location='../data/processed/pricing_store_weeks.parquet', format='parquet') }}

-- Model-ready pricing panel: one row per store, product and week, with product
-- and store attributes. Derived from licensed data, so it is not committed.
select
    s.week_end,
    s.store_id,
    st.price_tier,
    st.state,
    s.upc,
    p.description,
    p.manufacturer,
    p.category,
    p.sub_category,
    s.units,
    s.visits,
    s.households,
    s.spend,
    s.price,
    s.base_price,
    s.discount_depth,
    s.is_feature,
    s.is_display,
    s.is_tpr_only,
    s.is_promo
from {{ ref('stg_frat_sales') }} s
join {{ ref('stg_frat_products') }} p using (upc)
join {{ ref('stg_frat_stores') }} st using (store_id)
