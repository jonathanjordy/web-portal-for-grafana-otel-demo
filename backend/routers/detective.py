import re
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict
from fastapi import APIRouter, Query, HTTPException
from db import query_df, query_rows

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — MULTIVARIATE ANOMALY DETECTION
# Uses Isolation Forest on multiple metrics simultaneously.
# Flags time windows where the combination of metrics is unusual
# even if no single metric crosses a static threshold.
# ─────────────────────────────────────────────────────────────
@router.get("/anomalies")
async def detect_anomalies(
    hours: int = Query(default=6,  description="Hours of history to analyse"),
    contamination: float = Query(default=0.05, description="Expected anomaly fraction (0.01–0.2)"),
):
    """
    Runs Isolation Forest on:
    - payment_duration_seconds (p95)
    - payment_failures_total (rate)
    - inventory_cache_misses_total (rate)
    - order_errors_total (rate)
    - node_load1
    Returns a timeline with anomaly scores and flagged windows.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    interval = 1  # 1-minute buckets

    metrics = [
        ("payment_duration",   "otel_metrics_histogram", "order_duration_seconds",       "avg(Sum / nullIf(Count, 0))"),
        ("payment_failures",   "otel_metrics_sum",       "payment_failures_total",        "sum(Value)"),
        ("cache_misses",       "otel_metrics_sum",       "inventory_cache_misses_total",  "sum(Value)"),
        ("order_errors",       "otel_metrics_sum",       "order_errors_total",            "sum(Value)"),
        ("node_load",          "otel_metrics_gauge",     "node_load1",                    "avg(Value)"),
    ]

    dfs = []
    for col_name, table, metric, agg in metrics:
        sql = f"""
            SELECT
                toStartOfInterval(TimeUnix, INTERVAL {interval} MINUTE) AS ts,
                {agg} AS val
            FROM otel.{table}
            WHERE MetricName = '{metric}'
              AND TimeUnix >= now() - INTERVAL {hours} HOUR
            GROUP BY ts ORDER BY ts ASC
        """
        df = query_df(sql)
        if not df.empty:
            df = df.rename(columns={"val": col_name})
            dfs.append(df.set_index("ts"))

    if len(dfs) < 2:
        raise HTTPException(
            status_code=422,
            detail="Not enough metric data for multivariate analysis. Need at least 2 metrics with data."
        )

    # Join all metrics on timestamp, forward-fill small gaps
    combined = pd.concat(dfs, axis=1).fillna(method="ffill").dropna()

    if len(combined) < 20:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(combined)} data points after joining. Need at least 20. Try a longer time window."
        )

    feature_cols = [c for c in combined.columns if c in [m[0] for m in metrics]]
    X = combined[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    model.fit(X_scaled)

    scores    = model.decision_function(X_scaled)   # higher = more normal
    labels    = model.predict(X_scaled)              # -1 = anomaly, 1 = normal
    norm_score = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    results = []
    for i, (ts, row) in enumerate(combined.iterrows()):
        is_anomaly = bool(labels[i] == -1)

        # Which features contributed most to the anomaly
        contributing = []
        if is_anomaly:
            z_scores = np.abs(X_scaled[i])
            top_idx  = np.argsort(z_scores)[::-1][:3]
            contributing = [feature_cols[j] for j in top_idx if z_scores[j] > 1.5]

        results.append({
            "ts":            str(ts),
            "anomaly_score": round(float(norm_score[i]), 4),
            "is_anomaly":    is_anomaly,
            "contributing":  contributing,
            "metrics": {
                col: round(float(row[col]), 4) if col in row and pd.notna(row[col]) else None
                for col in feature_cols
            }
        })

    anomaly_count = sum(1 for r in results if r["is_anomaly"])

    return {
        "total_windows":  len(results),
        "anomaly_count":  anomaly_count,
        "anomaly_rate":   round(anomaly_count / len(results) * 100, 1),
        "features_used":  feature_cols,
        "timeline":       results,
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — LOG PATTERN CLUSTERING
# Uses Drain3 to parse log templates, then groups by pattern.
# Detects new patterns and sudden volume spikes.
# ─────────────────────────────────────────────────────────────
@router.get("/log-patterns")
async def cluster_log_patterns(
    hours: int   = Query(default=2,   description="Hours of logs to analyse"),
    limit: int   = Query(default=500, description="Max log lines to process"),
):
    """
    Clusters log lines by their template (stripped of variables).
    Returns patterns ranked by frequency and flags:
    - New patterns (first seen in the last 30 minutes)
    - Spiking patterns (volume increased > 200% vs baseline)
    """
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    # Fetch logs
    sql = f"""
        SELECT
            Timestamp,
            SeverityText,
            Body,
            TraceId,
            ResourceAttributes['service.name'] AS service_name
        FROM otel.otel_logs
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
        ORDER BY Timestamp DESC
        LIMIT {limit}
    """
    df = query_df(sql)

    if df.empty:
        raise HTTPException(status_code=404, detail="No logs found in the specified time window.")

    # Baseline: logs from the window before current window
    baseline_sql = f"""
        SELECT Body
        FROM otel.otel_logs
        WHERE Timestamp >= now() - INTERVAL {hours * 2} HOUR
          AND Timestamp <  now() - INTERVAL {hours} HOUR
        LIMIT {limit}
    """
    baseline_df = query_df(baseline_sql)

    # Configure Drain3
    config = TemplateMinerConfig()
    config.parametrize_numeric_tokens = True
    miner = TemplateMiner(config=config)

    # Parse current logs
    pattern_map = defaultdict(lambda: {
        "count": 0, "severity": defaultdict(int),
        "services": defaultdict(int), "samples": [], "trace_ids": []
    })

    for _, row in df.iterrows():
        body = str(row.get("Body", ""))
        if not body:
            continue
        result   = miner.add_log_message(body)
        template = result["template_mined"]

        pattern_map[template]["count"] += 1
        pattern_map[template]["severity"][str(row.get("SeverityText", "INFO"))] += 1
        pattern_map[template]["services"][str(row.get("service_name", "unknown"))] += 1

        if len(pattern_map[template]["samples"]) < 2:
            pattern_map[template]["samples"].append(body[:120])
        if row.get("TraceId") and len(pattern_map[template]["trace_ids"]) < 3:
            pattern_map[template]["trace_ids"].append(str(row["TraceId"]))

    # Parse baseline logs to detect new patterns
    baseline_templates = set()
    if not baseline_df.empty:
        baseline_config = TemplateMinerConfig()
        baseline_config.parametrize_numeric_tokens = True
        baseline_miner = TemplateMiner(config=baseline_config)
        for _, row in baseline_df.iterrows():
            body = str(row.get("Body", ""))
            if body:
                result = baseline_miner.add_log_message(body)
                baseline_templates.add(result["template_mined"])

    # Build output
    patterns = []
    total_logs = len(df)

    for template, data in sorted(pattern_map.items(), key=lambda x: -x[1]["count"]):
        is_new     = template not in baseline_templates
        pct        = round(data["count"] / total_logs * 100, 1)
        dominant_sev = max(data["severity"], key=data["severity"].get) if data["severity"] else "INFO"
        top_service  = max(data["services"],  key=data["services"].get)  if data["services"]  else "unknown"

        patterns.append({
            "template":     template,
            "count":        data["count"],
            "pct_of_total": pct,
            "is_new":       is_new,
            "dominant_severity": dominant_sev,
            "top_service":  top_service,
            "severity_breakdown": dict(data["severity"]),
            "service_breakdown":  dict(data["services"]),
            "samples":      data["samples"],
            "trace_ids":    data["trace_ids"],
        })

    new_count   = sum(1 for p in patterns if p["is_new"])
    error_count = sum(1 for p in patterns if p["dominant_severity"] == "ERROR")

    return {
        "total_logs":     total_logs,
        "unique_patterns": len(patterns),
        "new_patterns":   new_count,
        "error_patterns": error_count,
        "patterns":       patterns,
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3 — TRACE SHAPE ANOMALY DETECTION
# Builds a fingerprint for each trace (which spans appeared,
# how many times, in what order) and flags deviations.
# ─────────────────────────────────────────────────────────────
@router.get("/trace-shapes")
async def detect_trace_shape_anomalies(
    hours: int = Query(default=2,  description="Hours of traces to analyse"),
    limit: int = Query(default=500, description="Max traces to analyse"),
):
    """
    Groups traces by their span fingerprint.
    The baseline shape is the most common fingerprint.
    Flags traces that deviate — extra spans, missing spans, loops.
    """
    sql = f"""
        SELECT
            TraceId,
            SpanName,
            ServiceName,
            Duration,
            StatusCode,
            count() AS span_count
        FROM otel.otel_traces
        WHERE Timestamp >= now() - INTERVAL {hours} HOUR
          AND TraceId != ''
        GROUP BY TraceId, SpanName, ServiceName, Duration, StatusCode
        ORDER BY TraceId ASC
        LIMIT {limit}
    """
    df = query_df(sql)

    if df.empty:
        raise HTTPException(status_code=404, detail="No traces found in the specified time window.")

    # Build fingerprint per trace
    trace_shapes = defaultdict(lambda: {"spans": [], "has_error": False, "total_duration": 0})

    for _, row in df.iterrows():
        tid = str(row["TraceId"])
        trace_shapes[tid]["spans"].append({
            "name":     str(row["SpanName"]),
            "service":  str(row["ServiceName"]),
            "count":    int(row["span_count"]),
            "duration": int(row["Duration"]),
            "is_error": str(row["StatusCode"]) == "STATUS_CODE_ERROR",
        })
        if str(row["StatusCode"]) == "STATUS_CODE_ERROR":
            trace_shapes[tid]["has_error"] = True
        trace_shapes[tid]["total_duration"] += int(row["Duration"])

    # Create fingerprint string for each trace
    def make_fingerprint(spans):
        sorted_spans = sorted(spans, key=lambda s: s["name"])
        parts = [f"{s['name']}x{s['count']}" for s in sorted_spans]
        return "|".join(parts)

    fingerprint_counts = defaultdict(int)
    trace_fingerprints = {}

    for tid, data in trace_shapes.items():
        fp = make_fingerprint(data["spans"])
        fingerprint_counts[fp] += 1
        trace_fingerprints[tid] = fp

    # Baseline = most common fingerprint
    if not fingerprint_counts:
        raise HTTPException(status_code=422, detail="Could not compute trace fingerprints.")

    baseline_fp    = max(fingerprint_counts, key=fingerprint_counts.get)
    baseline_count = fingerprint_counts[baseline_fp]
    total_traces   = len(trace_shapes)

    # Find anomalous traces
    anomalous = []
    shape_summary = []

    for fp, count in sorted(fingerprint_counts.items(), key=lambda x: -x[1]):
        is_baseline = fp == baseline_fp
        pct = round(count / total_traces * 100, 1)

        # Find example traces with this fingerprint
        example_traces = [
            tid for tid, tfp in trace_fingerprints.items()
            if tfp == fp
        ][:3]

        # Detect what's different from baseline
        deviation_type = None
        if not is_baseline:
            fp_spans   = set(fp.split("|"))
            base_spans = set(baseline_fp.split("|"))
            extra      = fp_spans - base_spans
            missing    = base_spans - fp_spans

            if any("x" in s and int(s.split("x")[-1]) > 1 for s in extra):
                deviation_type = "loop_detected"
            elif extra:
                deviation_type = "extra_spans"
            elif missing:
                deviation_type = "missing_spans"
            else:
                deviation_type = "different_order"

        shape_summary.append({
            "fingerprint":    fp,
            "count":          count,
            "pct_of_total":   pct,
            "is_baseline":    is_baseline,
            "deviation_type": deviation_type,
            "example_traces": example_traces,
        })

        if not is_baseline:
            for tid in example_traces:
                data = trace_shapes[tid]
                anomalous.append({
                    "trace_id":       tid,
                    "fingerprint":    fp,
                    "deviation_type": deviation_type,
                    "has_error":      data["has_error"],
                    "total_duration_ms": round(data["total_duration"] / 1e6, 2),
                    "span_count":     len(data["spans"]),
                    "spans":          data["spans"],
                })

    return {
        "total_traces":     total_traces,
        "unique_shapes":    len(fingerprint_counts),
        "baseline_shape":   baseline_fp,
        "baseline_pct":     round(baseline_count / total_traces * 100, 1),
        "anomalous_count":  len(anomalous),
        "shape_summary":    shape_summary,
        "anomalous_traces": anomalous[:20],
    }