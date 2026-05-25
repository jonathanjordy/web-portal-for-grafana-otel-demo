from fastapi import APIRouter

from db import query_rows
from query_filters import HOST_ATTRIBUTE_KEYS

router = APIRouter()


@router.get("")
async def get_filters():
    """Return available filter values discovered from recent telemetry."""
    services = query_rows("""
        SELECT ServiceName AS value, count() AS count
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL 24 HOUR
          AND ServiceName != ''
        GROUP BY ServiceName
        ORDER BY count DESC, value ASC
    """)

    host_key_expr = ", ".join(f"'{key}'" for key in HOST_ATTRIBUTE_KEYS)
    hosts = query_rows(f"""
        SELECT value, any(key) AS source, sum(count) AS count
        FROM (
            SELECT
                MetricName,
                arrayJoin(mapKeys(Attributes)) AS key,
                Attributes[key] AS value,
                count() AS count
            FROM otel.otel_metrics_gauge
            WHERE MetricName IN (
                'node_memory_MemAvailable_bytes',
                'node_memory_MemTotal_bytes',
                'node_load1'
            )
              AND TimeUnix >= now() - INTERVAL 24 HOUR
              AND key IN ({host_key_expr})
              AND value != ''
            GROUP BY MetricName, key, value
        )
        GROUP BY value
        ORDER BY count DESC, value ASC
        LIMIT 100
    """)

    return {
        "services": services,
        "hosts": hosts,
        "host_attribute_keys": list(HOST_ATTRIBUTE_KEYS),
    }
