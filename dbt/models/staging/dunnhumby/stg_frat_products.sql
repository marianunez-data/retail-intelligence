select
    cast(upc as varchar)      as upc,
    trim(description)         as description,
    trim(manufacturer)        as manufacturer,
    {{ sentence_case('category') }} as category,
    {{ sentence_case('sub_category') }} as sub_category,
    trim(product_size)        as product_size
from {{ source('dunnhumby', 'products') }}
