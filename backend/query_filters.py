from __future__ import annotations


HOST_ATTRIBUTE_KEYS = ("nodename", "host.name", "instance")


def sql_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def service_trace_filter(service: str | None) -> str:
    if not service:
        return ""
    return f"AND ServiceName = '{sql_quote(service)}'"


def service_log_filter(service: str | None) -> str:
    if not service:
        return ""
    return f"AND ResourceAttributes['service.name'] = '{sql_quote(service)}'"


def metric_host_filter(host: str | None) -> str:
    if not host:
        return ""
    quoted = sql_quote(host)
    checks = " OR ".join(f"Attributes['{key}'] = '{quoted}'" for key in HOST_ATTRIBUTE_KEYS)
    return f"AND ({checks})"
