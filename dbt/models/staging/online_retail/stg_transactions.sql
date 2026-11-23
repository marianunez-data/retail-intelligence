-- Registered-customer invoice lines, the input of the retention model.
-- Rule 2: guest lines (no customer ID) cannot be linked to a customer.
select
    invoice_id,
    product_id,
    customer_id,
    invoiced_at,
    country,
    quantity,
    unit_price,
    revenue,
    is_cancellation
from {{ ref('stg_sales_lines') }}
where customer_id is not null
