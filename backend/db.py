import clickhouse_connect
import pandas as pd
from typing import Any
from config import get_settings

settings = get_settings()


def get_client() -> clickhouse_connect.driver.Client:
    """Returns a ClickHouse HTTP client."""
    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
    )


def query_df(sql: str) -> pd.DataFrame:
    """Run a SQL query and return a pandas DataFrame."""
    client = get_client()
    return client.query_df(sql)


def query_rows(sql: str) -> list[dict]:
    """Run a SQL query and return a list of dicts."""
    client = get_client()
    result = client.query(sql)
    columns = result.column_names
    return [dict(zip(columns, row)) for row in result.result_rows]


def query_scalar(sql: str) -> Any:
    """Run a SQL query and return a single scalar value."""
    client = get_client()
    result = client.query(sql)
    if result.result_rows:
        return result.result_rows[0][0]
    return None


# ─────────────────────────────────────────────────────────────
# REUSABLE QUERY HELPERS
# ─────────────────────────────────────────────────────────────

def get_metric_timeseries(
    metric_name: str,
    table: str = "otel_metrics_gauge",
    hours: int = 24,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """
    Fetch a metric time series bucketed into intervals.
    Returns columns: [timestamp, value]
    """
    sql = f"""
        SELECT
            toStartOfInterval(TimeUnix, INTERVAL {interval_minutes} MINUTE) AS timestamp,
            avg(Value) AS value
        FROM otel.{table}
        WHERE MetricName = '{metric_name}'
          AND TimeUnix >= now() - INTERVAL {hours} HOUR
        GROUP BY timestamp
        ORDER BY timestamp ASC
    """
    return query_df(sql)


def get_multiple_metrics_timeseries(
    metric_names: list[str],
    tables: list[str],
    hours: int = 6,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """
    Fetch multiple metrics as a wide DataFrame for multivariate analysis.
    Returns columns: [timestamp, metric1, metric2, ...]
    """
    subqueries = []
    for metric, table in zip(metric_names, tables):
        subqueries.append(f"""
            SELECT
                toStartOfInterval(TimeUnix, INTERVAL {interval_minutes} MINUTE) AS timestamp,
                '{metric}' AS metric_name,
                avg(Value) AS value
            FROM otel.{table}
            WHERE MetricName = '{metric}'
              AND TimeUnix >= now() - INTERVAL {hours} HOUR
            GROUP BY timestamp, metric_name
        """)

    union_sql = " UNION ALL ".join(subqueries)
    sql = f"""
        SELECT timestamp, metric_name, value
        FROM ({union_sql})
        ORDER BY timestamp ASC, metric_name ASC
    """
    long_df = query_df(sql)
    if long_df.empty:
        return long_df
    # Pivot to wide format
    wide_df = long_df.pivot(index="timestamp", columns="metric_name", values="value")
    wide_df = wide_df.reset_index().fillna(method="ffill").dropna()
    return wide_df


def get_recent_logs(
    hours: int = 1,
    severity: str = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch recent log lines from otel_logs."""
    severity_filter = f"AND SeverityText = '{severity}'" if severity else ""
    sql = f"""
        SELECT
            Timestamp,
            SeverityText,
            Body,
            TraceId,
            SpanId,
            ResourceAttributes['service.name'] AS service_name
        FROM otel.otel_logs
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          {severity_filter}
        ORDER BY Timestamp DESC
        LIMIT {limit}
    """
    return query_df(sql)


def get_recent_traces(
    hours: int = 1,
    service_name: str = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch recent spans from otel_traces."""
    service_filter = f"AND ServiceName = '{service_name}'" if service_name else ""
    sql = f"""
        SELECT
            TraceId,
            SpanId,
            ParentSpanId,
            ServiceName,
            SpanName,
            Duration,
            StatusCode,
            Timestamp,
            SpanAttributes
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          {service_filter}
        ORDER BY Timestamp DESC
        LIMIT {limit}
    """
    return query_df(sql)


def get_error_rate(hours: int = 1, interval_minutes: int = 1) -> pd.DataFrame:
    """Fetch payment error rate over time."""
    sql = f"""
        SELECT
            toStartOfInterval(TimeUnix, INTERVAL {interval_minutes} MINUTE) AS timestamp,
            sum(Value) AS failures
        FROM otel.otel_metrics_sum
        WHERE MetricName = 'demo_payment_failures_total'
          AND TimeUnix >= now() - INTERVAL {hours} HOUR
        GROUP BY timestamp
        ORDER BY timestamp ASC
    """
    return query_df(sql)


def get_available_metrics() -> list[str]:
    """Return all unique metric names across all metric tables."""
    sql = """
        SELECT DISTINCT MetricName FROM otel.otel_metrics_sum
        UNION ALL
        SELECT DISTINCT MetricName FROM otel.otel_metrics_gauge
        UNION ALL
        SELECT DISTINCT MetricName FROM otel.otel_metrics_histogram
        ORDER BY MetricName ASC
    """
    return [row["MetricName"] for row in query_rows(sql)]


def get_clickhouse_schema() -> str:
    """
    Returns a description of the ClickHouse schema for use in LLM prompts.
    This tells the LLM what tables and columns are available.
    """
    return """
You have access to a ClickHouse database called `otel` with the following tables:

1. otel.otel_traces
   - TraceId String           -- unique trace identifier
   - SpanId String            -- unique span identifier
   - ParentSpanId String      -- parent span (empty for root spans)
   - ServiceName String       -- e.g. 'order-service', 'inventory-service', 'payment-service'
   - SpanName String          -- e.g. 'POST /orders', 'payment-gateway-call', 'db-stock-lookup'
   - Duration UInt64          -- span duration in nanoseconds
   - StatusCode String        -- 'STATUS_CODE_OK' or 'STATUS_CODE_ERROR'
   - StatusMessage String     -- error message if failed
   - Timestamp DateTime       -- when the span started
   - SpanAttributes Map(String, String) -- custom attributes like order.item_id, payment.amount

2. otel.otel_logs
   - Timestamp DateTime
   - SeverityText String      -- 'INFO', 'WARNING', 'ERROR'
   - Body String              -- the log message
   - TraceId String           -- links to otel_traces
   - SpanId String
   - ResourceAttributes Map(String, String) -- includes service.name

3. otel.otel_metrics_sum  (counters)
   - MetricName String        -- e.g. 'demo_orders_total', 'demo_payment_failures_total'
   - TimeUnix DateTime
   - Value Float64
   - Attributes Map(String, String) -- labels like item_id, status, reason

4. otel.otel_metrics_gauge  (current values)
   - MetricName String        -- e.g. 'demo_node_memory_MemAvailable_bytes', 'demo_node_load1'
   - TimeUnix DateTime
   - Value Float64
   - Attributes Map(String, String)

5. otel.otel_metrics_histogram  (distributions)
   - MetricName String        -- e.g. 'demo_order_duration_seconds'
   - TimeUnix DateTime
   - Count UInt64
   - Sum Float64
   - Attributes Map(String, String)

Services in this system: order-service, inventory-service, payment-service
Available items: laptop, headphones, keyboard, monitor, mouse
All timestamps are UTC.
"""
