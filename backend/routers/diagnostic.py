import os
import json
import httpx
from collections import defaultdict
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from db import query_df, query_rows, get_clickhouse_schema

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — CAUSAL GRAPH (SERVICE DEPENDENCY MAP)
# Builds a real-time dependency map from trace parent-child
# relationships. Walks the graph to identify root cause when
# a service has elevated error rate.
# ─────────────────────────────────────────────────────────────
@router.get("/causal-graph")
async def get_causal_graph(
    hours: int = Query(default=1, description="Hours of traces to analyse"),
):
    """
    Returns service nodes and edges with request rate,
    error rate, and p95 latency per edge.
    Also identifies the most likely root cause service.
    """
    # Get all spans with their parent info
    sql = f"""
        SELECT
            ServiceName,
            SpanName,
            ParentSpanId,
            SpanId,
            TraceId,
            Duration,
            StatusCode
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          AND TraceId != ''
        LIMIT 5000
    """
    df = query_df(sql)

    if df.empty:
        raise HTTPException(status_code=404, detail="No traces found.")

    # Build span lookup: spanId → service
    span_service = {}
    for _, row in df.iterrows():
        span_service[str(row["SpanId"])] = str(row["ServiceName"])

    # Build edges: caller → callee with metrics
    edges = defaultdict(lambda: {"calls": 0, "errors": 0, "total_duration": 0})
    node_metrics = defaultdict(lambda: {"calls": 0, "errors": 0, "total_duration": 0})

    for _, row in df.iterrows():
        service    = str(row["ServiceName"])
        parent_id  = str(row["ParentSpanId"])
        is_error   = str(row["StatusCode"]) == "STATUS_CODE_ERROR"
        duration   = int(row["Duration"])

        node_metrics[service]["calls"] += 1
        node_metrics[service]["total_duration"] += duration
        if is_error:
            node_metrics[service]["errors"] += 1

        # If this span has a parent from a different service, create an edge
        if parent_id and parent_id in span_service:
            parent_service = span_service[parent_id]
            if parent_service != service:
                edge_key = f"{parent_service}→{service}"
                edges[edge_key]["calls"] += 1
                edges[edge_key]["total_duration"] += duration
                if is_error:
                    edges[edge_key]["errors"] += 1

    # Build nodes list
    nodes = []
    for service, metrics in node_metrics.items():
        calls     = metrics["calls"]
        errors    = metrics["errors"]
        err_rate  = round(errors / calls * 100, 1) if calls > 0 else 0
        avg_dur_ms = round(metrics["total_duration"] / calls / 1e6, 1) if calls > 0 else 0
        nodes.append({
            "id":          service,
            "calls":       calls,
            "errors":      errors,
            "error_rate":  err_rate,
            "avg_duration_ms": avg_dur_ms,
            "status":      "error" if err_rate > 10 else "degraded" if err_rate > 2 else "ok",
        })

    # Build edges list
    edge_list = []
    for edge_key, metrics in edges.items():
        caller, callee = edge_key.split("→")
        calls    = metrics["calls"]
        errors   = metrics["errors"]
        err_rate = round(errors / calls * 100, 1) if calls > 0 else 0
        avg_ms   = round(metrics["total_duration"] / calls / 1e6, 1) if calls > 0 else 0
        edge_list.append({
            "source":       caller,
            "target":       callee,
            "calls":        calls,
            "errors":       errors,
            "error_rate":   err_rate,
            "avg_duration_ms": avg_ms,
        })

    # Root cause: find the service with highest error rate
    # that is NOT called by another erroring service
    # (i.e., the deepest node in the error chain)
    error_services = {n["id"] for n in nodes if n["error_rate"] > 5}
    called_by_error = set()
    for e in edge_list:
        if e["source"] in error_services and e["target"] in error_services:
            called_by_error.add(e["target"])

    root_cause_candidates = error_services - called_by_error
    root_cause = list(root_cause_candidates)[0] if root_cause_candidates else None

    return {
        "nodes":      nodes,
        "edges":      edge_list,
        "root_cause": root_cause,
        "total_spans": len(df),
        "hours":      hours,
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — TELEMETRY CORRELATION ENGINE
# Given a time window, surfaces the slowest traces,
# attached error logs, and metric spikes side by side.
# ─────────────────────────────────────────────────────────────
@router.get("/correlate")
async def correlate_telemetry(
    hours:   int = Query(default=1,  description="Hours to look back"),
    service: str = Query(default="", description="Filter by service name"),
):
    """
    Correlates metrics + traces + logs for a time window.
    Returns slowest traces, error logs, and metric summary
    all aligned to the same time window.
    """
    service_filter = f"AND ServiceName = '{service}'" if service else ""
    log_service_filter = f"AND ResourceAttributes['service.name'] = '{service}'" if service else ""

    # Slowest traces
    slow_sql = f"""
        SELECT
            TraceId,
            ServiceName,
            SpanName,
            Duration,
            StatusCode,
            Timestamp
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          AND ParentSpanId = ''
          {service_filter}
        ORDER BY Duration DESC
        LIMIT 10
    """
    slow_df = query_df(slow_sql)

    # Error traces
    error_sql = f"""
        SELECT
            TraceId,
            ServiceName,
            SpanName,
            Duration,
            StatusCode,
            Timestamp
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          AND StatusCode = 'STATUS_CODE_ERROR'
          AND ParentSpanId = ''
          {service_filter}
        ORDER BY Timestamp DESC
        LIMIT 10
    """
    error_df = query_df(error_sql)

    # Error logs
    log_sql = f"""
        SELECT
            Timestamp,
            SeverityText,
            Body,
            TraceId,
            ResourceAttributes['service.name'] AS service_name
        FROM otel.otel_logs
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          AND SeverityText = 'ERROR'
          {log_service_filter}
        ORDER BY Timestamp DESC
        LIMIT 20
    """
    log_df = query_df(log_sql)

    # Metric summary
    metric_sql = f"""
        SELECT
            MetricName,
            avg(Value) AS avg_val,
            max(Value) AS max_val
        FROM otel.otel_metrics_sum
        WHERE TimeUnix >= now() - INTERVAL {hours} HOUR
          AND MetricName IN (
              'payment_failures_total',
              'order_errors_total',
              'orders_total',
              'inventory_cache_misses_total'
          )
        GROUP BY MetricName
    """
    metric_df = query_df(metric_sql)

    def df_to_list(df):
        if df.empty:
            return []
        df = df.copy()
        for col in df.columns:
            if hasattr(df[col], 'dt'):
                df[col] = df[col].astype(str)
        return df.fillna("").to_dict(orient="records")

    return {
        "window_hours":  hours,
        "service_filter": service or "all",
        "slow_traces":   df_to_list(slow_df),
        "error_traces":  df_to_list(error_df),
        "error_logs":    df_to_list(log_df),
        "metric_summary": df_to_list(metric_df),
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3 — LLM INCIDENT SUMMARIZATION
# Feeds correlated telemetry into Claude API and returns
# a human-readable incident summary.
# ─────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    hours:   int = 1
    service: str = ""


@router.post("/summarize")
async def summarize_incident(req: SummarizeRequest):
    """
    Calls the Anthropic Claude API with correlated telemetry
    data and returns a human-readable incident summary with
    root cause, impact, and recommended next steps.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not set in .env. Add it to enable LLM summarization."
        )

    # First get the correlated data
    service_filter    = f"AND ServiceName = '{req.service}'" if req.service else ""
    log_service_filter = f"AND ResourceAttributes['service.name'] = '{req.service}'" if req.service else ""

    # Collect key signals
    signals = {}

    # Error rate
    err_sql = f"""
        SELECT
            ServiceName,
            countIf(StatusCode = 'STATUS_CODE_ERROR') AS errors,
            count() AS total,
            round(countIf(StatusCode = 'STATUS_CODE_ERROR') / count() * 100, 1) AS error_rate_pct
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {req.hours} HOUR
          AND ParentSpanId = ''
          {service_filter}
        GROUP BY ServiceName
        ORDER BY error_rate_pct DESC
    """
    err_df = query_df(err_sql)
    signals["error_rates"] = err_df.to_dict(orient="records") if not err_df.empty else []

    # Slowest spans
    slow_sql = f"""
        SELECT SpanName, ServiceName,
               round(avg(Duration)/1e6, 1) AS avg_ms,
               round(quantile(0.95)(Duration)/1e6, 1) AS p95_ms
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {req.hours} HOUR
          {service_filter}
        GROUP BY SpanName, ServiceName
        ORDER BY p95_ms DESC
        LIMIT 5
    """
    slow_df = query_df(slow_sql)
    signals["slowest_spans"] = slow_df.to_dict(orient="records") if not slow_df.empty else []

    # Recent error logs
    log_sql = f"""
        SELECT Body, ResourceAttributes['service.name'] AS service, TraceId
        FROM otel.otel_logs
        WHERE Timestamp >= now() - INTERVAL {req.hours} HOUR
          AND SeverityText = 'ERROR'
          {log_service_filter}
        ORDER BY Timestamp DESC
        LIMIT 5
    """
    log_df = query_df(log_sql)
    signals["error_logs"] = log_df.to_dict(orient="records") if not log_df.empty else []

    # Metric anomalies
    metric_sql = f"""
        SELECT MetricName, sum(Value) AS total
        FROM otel.otel_metrics_sum
        WHERE TimeUnix >= now() - INTERVAL {req.hours} HOUR
          AND MetricName IN ('payment_failures_total', 'order_errors_total')
        GROUP BY MetricName
    """
    metric_df = query_df(metric_sql)
    signals["metric_totals"] = metric_df.to_dict(orient="records") if not metric_df.empty else []

    # Build the prompt
    service_ctx = f" for service '{req.service}'" if req.service else " across all services"
    prompt = f"""You are an expert SRE analyzing a production incident.

Here is the telemetry data from the last {req.hours} hour(s){service_ctx}:

ERROR RATES BY SERVICE:
{json.dumps(signals['error_rates'], indent=2)}

SLOWEST SPANS (p95 latency):
{json.dumps(signals['slowest_spans'], indent=2)}

RECENT ERROR LOGS:
{json.dumps(signals['error_logs'], indent=2)}

METRIC TOTALS:
{json.dumps(signals['metric_totals'], indent=2)}

Based on this telemetry, provide a concise incident summary with:
1. **What is happening** — describe the symptoms clearly
2. **Root cause** — identify the most likely root cause based on the data
3. **Impact** — which services and users are affected
4. **Recommended actions** — 3 concrete next steps for the on-call engineer

Keep the summary concise and actionable. Format it in plain text suitable for a Slack message."""

    # Call Gemini API
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            gemini_url,
            headers={"content-type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature":     0.3,
                    "maxOutputTokens": 1000,
                }
            }
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {response.status_code} — {response.text[:200]}"
        )

    result  = response.json()
    summary = result["candidates"][0]["content"]["parts"][0]["text"]

    return {
        "summary":  summary,
        "signals":  signals,
        "hours":    req.hours,
        "service":  req.service or "all services",
    }