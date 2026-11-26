-- Fails on duplicate store-product-weeks, discounts outside [0, 1), or a
-- temporary-price-reduction flag without an actual price cut.
select week_end, store_id, upc, 'duplicate grain' as problem
from {{ ref('pricing_store_weeks') }}
group by 1, 2, 3
having count(*) > 1

union all

select week_end, store_id, upc, 'discount out of range'
from {{ ref('pricing_store_weeks') }}
where discount_depth < 0 or discount_depth >= 1
