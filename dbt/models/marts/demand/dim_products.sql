{{ config(materialized='external', location='../data/processed/dim_products.csv', format='csv') }}

-- One row per product: the most frequent description and a merchandise category
-- from the keyword rules in seeds/product_category_rules.csv.
with descriptions as (

    select product_id, description, count(*) as n
    from {{ ref('stg_sales_lines') }}
    where description is not null and not is_cancellation
    group by 1, 2

),

main_description as (

    select product_id, description
    from descriptions
    qualify row_number() over (partition by product_id order by n desc, description) = 1

),

matches as (

    select
        d.product_id,
        r.category,
        row_number() over (partition by d.product_id order by r.priority, r.keyword) as rn
    from main_description d
    join {{ ref('product_category_rules') }} r
      on regexp_matches(upper(d.description), '\b' || r.keyword || '(S|ES)?\b')

)

select
    d.product_id,
    d.description,
    coalesce(m.category, 'Other') as category
from main_description d
left join matches m on m.product_id = d.product_id and m.rn = 1
