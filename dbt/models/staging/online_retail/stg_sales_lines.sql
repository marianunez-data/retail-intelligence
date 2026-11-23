-- Clean, typed invoice lines for every sales channel. Every rule below comes
-- from the data audits in notebooks 01 (section 3) and 02 (section 3).
with source as (

    select * from {{ source('raw', 'transactions') }}

),

deduplicated as (

    -- Rule 1: exact duplicate lines (34k rows) are system double-posts
    select distinct * from source

),

cleaned as (

    select
        invoice                                           as invoice_id,
        stock_code                                        as product_id,
        trim(description)                                 as description,
        cast(customer_id as integer)                      as customer_id,
        case when customer_id is null then 'Guest' else 'Registered' end as channel,
        cast(invoice_date as timestamp)                   as invoiced_at,
        country,
        cast(quantity as integer)                         as quantity,
        cast(price as double)                             as unit_price,
        cast(quantity as integer) * cast(price as double) as revenue,
        starts_with(invoice, 'C')                         as is_cancellation
    from deduplicated
    where price > 0                                       -- Rule 3: free or negative-price lines are not sales
      and not starts_with(invoice, 'A')                   -- Rule 4: 'A' invoices are bad-debt adjustments
      -- Rule 5: postage, fees, manual adjustments and test codes are not merchandise
      and upper(stock_code) not in (
          {% for code in var('non_product_codes') %}'{{ code }}'{% if not loop.last %}, {% endif %}{% endfor %}
      )

)

select * from cleaned
