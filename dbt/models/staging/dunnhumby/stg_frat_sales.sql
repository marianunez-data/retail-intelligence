-- Weekly store-product sales with prices and promotion flags. Rules come from the
-- data audit in notebooks/03_pricing_and_promotions.ipynb, section 3.
select
    cast(week_end_date as date)                          as week_end,
    store_num                                            as store_id,
    cast(upc as varchar)                                 as upc,
    units,
    visits,
    hhs                                                  as households,
    spend,
    price,
    base_price,
    -- Rule 4: a shelf price above the base price is not a discount
    greatest(0, 1 - price / base_price)                  as discount_depth,
    feature = 1                                          as is_feature,
    display = 1                                          as is_display,
    tpr_only = 1                                         as is_tpr_only,
    (feature = 1 or display = 1 or tpr_only = 1)         as is_promo
from {{ source('dunnhumby', 'transactions') }}
where price > 0 and base_price > 0                       -- Rule 1: missing or zero prices cannot be modelled
  and units > 0                                          -- Rule 2: zero-unit rows are out of stock or delisted
  and units <= 5 * visits                                -- Rule 3: more than 5 units per basket is bulk buying, not shopper demand
