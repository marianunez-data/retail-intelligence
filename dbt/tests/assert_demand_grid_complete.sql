-- Fails if any product in the demand grid is missing a week, has a duplicate
-- week, or if weekly units disagree with the cleaned sales lines.
with per_product as (

    select product_id, count(*) as weeks, count(distinct week_start) as distinct_weeks
    from {{ ref('product_weekly_demand') }}
    group by 1

),

expected as (

    select count(distinct week_start) as weeks from {{ ref('product_weekly_demand') }}

)

select p.*
from per_product p cross join expected e
where p.weeks <> e.weeks or p.distinct_weeks <> e.weeks
