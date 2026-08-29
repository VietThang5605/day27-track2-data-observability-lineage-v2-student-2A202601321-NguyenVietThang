-- Singular business test: multiple active versions for one customer would
-- inflate revenue when joined (the failure mode fct_daily_revenue guards
-- against). Returns rows when the assertion is VIOLATED.
select
    customer_id,
    count(*) as active_versions
from {{ ref('stg_customers') }}
where is_active = true
group by 1
having count(*) > 1
