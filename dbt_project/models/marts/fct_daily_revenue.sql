-- NOTE: This model is intentionally simple. If the customer dimension has more
-- than one active row per customer, the join can inflate revenue without a SQL
-- error. Students should add tests/unit tests that expose this failure mode.

with completed_orders as (
    select *
    from {{ ref('stg_orders') }}
    where status = 'completed'
),
-- Guard against revenue inflation: if the customer dimension carries more
-- than one active row per customer (e.g. overlapping SCD versions), the join
-- would multiply order revenue. Deterministically keep one active version.
active_customers as (
    select customer_id
    from (
        select
            customer_id,
            row_number() over (
                partition by customer_id
                order by valid_from desc
            ) as rn
        from {{ ref('stg_customers') }}
        where is_active = true
    )
    where rn = 1
)
select
    o.order_date,
    count(*) as completed_order_rows,
    sum(o.amount_usd) as daily_revenue
from completed_orders o
left join active_customers c
    on o.customer_id = c.customer_id
group by 1
order by 1
