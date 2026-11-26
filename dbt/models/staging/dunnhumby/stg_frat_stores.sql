-- Rule 5: stores 4503 and 17627 appear twice in the lookup, once as Mainstream
-- and once as Upscale. One row per store is kept, with the Mainstream tier,
-- so that joining the lookup cannot duplicate sales rows.
select
    store_id,
    trim(store_name)                       as store_name,
    trim(address_city_name)                as city,
    trim(address_state_prov_code)          as state,
    {{ sentence_case('seg_value_name') }}  as price_tier,
    sales_area_size_num                    as sales_area,
    avg_weekly_baskets
from {{ source('dunnhumby', 'stores') }}
qualify row_number() over (
    partition by store_id
    order by case upper(trim(seg_value_name)) when 'MAINSTREAM' then 0 else 1 end
) = 1
