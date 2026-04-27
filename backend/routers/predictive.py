import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from db import query_df

router = APIRouter()


def fit_and_forecast(df: pd.DataFrame, periods: int, freq: str = "5min") -> dict:
    """
    Fit a Prophet model on df (must have columns: ds, y)
    and return forecast as a JSON-serializable dict.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise HTTPException(status_code=500, detail="Prophet not installed")

    if df.empty or len(df) < 10:
        raise HTTPException(
            status_code=422,
            detail="Not enough historical data to forecast. Need at least 10 data points."
        )

    df = df.dropna(subset=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(df)

    future   = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    merged = merged.merge(df[["ds", "y"]], on="ds", how="left")

    return {
        "historical": [
            {"ds": str(r["ds"]), "y": round(float(r["y"]), 4) if pd.notna(r["y"]) else None}
            for _, r in merged[merged["y"].notna()].iterrows()
        ],
        "forecast": [
            {
                "ds":         str(r["ds"]),
                "yhat":       round(float(r["yhat"]), 4),
                "yhat_lower": round(float(r["yhat_lower"]), 4),
                "yhat_upper": round(float(r["yhat_upper"]), 4),
            }
            for _, r in merged[merged["y"].isna()].iterrows()
        ],
        "data_points": len(df),
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — MEMORY FORECAST
# ─────────────────────────────────────────────────────────────
@router.get("/memory")
async def forecast_memory(
    hours_history: int = Query(default=48),
    hours_ahead:   int = Query(default=24),
):
    sql = f"""
        SELECT
            toStartOfInterval(TimeUnix, INTERVAL 5 MINUTE) AS ds,
            avg(Value) AS y
        FROM otel.otel_metrics_gauge
        WHERE MetricName = 'node_memory_MemAvailable_bytes'
          AND TimeUnix >= now() - INTERVAL {hours_history} HOUR
        GROUP BY ds ORDER BY ds ASC
    """
    df = query_df(sql)
    if df.empty:
        raise HTTPException(status_code=404, detail="No memory metrics found.")

    df["y"] = df["y"] / (1024 ** 3)

    total_df = query_df("""
        SELECT avg(Value) / (1024*1024*1024) AS total_gb
        FROM otel.otel_metrics_gauge
        WHERE MetricName = 'node_memory_MemTotal_bytes'
          AND TimeUnix >= now() - INTERVAL 1 HOUR
    """)
    total_gb = float(total_df.iloc[0]["total_gb"]) if not total_df.empty else None

    result = fit_and_forecast(df, periods=(hours_ahead * 60) // 5, freq="5min")
    result["unit"]     = "GB available"
    result["total_gb"] = round(total_gb, 2) if total_gb else None
    return result


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — CPU FORECAST
# ─────────────────────────────────────────────────────────────
@router.get("/cpu")
async def forecast_cpu(
    hours_history: int = Query(default=48),
    hours_ahead:   int = Query(default=24),
):
    sql = f"""
        SELECT
            toStartOfInterval(TimeUnix, INTERVAL 5 MINUTE) AS ds,
            avg(Value) AS idle_seconds
        FROM otel.otel_metrics_sum
        WHERE MetricName = 'node_cpu_seconds_total'
          AND Attributes['mode'] = 'idle'
          AND TimeUnix >= now() - INTERVAL {hours_history} HOUR
        GROUP BY ds ORDER BY ds ASC
    """
    df = query_df(sql)
    if df.empty:
        raise HTTPException(status_code=404, detail="No CPU metrics found.")

    df["idle_rate"] = df["idle_seconds"].diff() / 300
    df["y"]         = (1 - df["idle_rate"].clip(0, 1)) * 100
    df              = df.dropna(subset=["y"])
    df              = df[df["y"].between(0, 100)]

    result = fit_and_forecast(df, periods=(hours_ahead * 60) // 5, freq="5min")
    result["unit"] = "% CPU usage"
    return result


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3 — TRAFFIC FORECAST
# ─────────────────────────────────────────────────────────────
@router.get("/traffic")
async def forecast_traffic(
    hours_history: int = Query(default=48),
    hours_ahead:   int = Query(default=12),
):
    sql = f"""
        SELECT
            toStartOfInterval(Timestamp, INTERVAL 5 MINUTE) AS ds,
            count() AS y
        FROM otel.otel_traces
        WHERE SpanName = 'POST /orders'
          AND ParentSpanId = ''
          AND Timestamp >= now() - INTERVAL {hours_history} HOUR
        GROUP BY ds ORDER BY ds ASC
    """
    df = query_df(sql)
    if df.empty:
        raise HTTPException(status_code=404, detail="No order traces found.")

    df["y"]  = df["y"].astype(float)
    result   = fit_and_forecast(df, periods=(hours_ahead * 60) // 5, freq="5min")
    result["unit"] = "orders per 5 min"

    forecast_vals = [f["yhat"] for f in result["forecast"]]
    if forecast_vals:
        peak_val = max(forecast_vals)
        peak_idx = forecast_vals.index(peak_val)
        result["peak_predicted"] = {
            "value": round(peak_val, 1),
            "at":    result["forecast"][peak_idx]["ds"],
        }
    return result


# ─────────────────────────────────────────────────────────────
# ENDPOINT 4 — SUMMARY CARDS (no Prophet, fast)
# ─────────────────────────────────────────────────────────────
@router.get("/summary")
async def forecast_summary():
    results = {}

    try:
        mem_df = query_df("""
            SELECT
                avg(Value) / (1024*1024*1024) AS available_gb,
                (SELECT avg(Value) / (1024*1024*1024)
                 FROM otel.otel_metrics_gauge
                 WHERE MetricName = 'node_memory_MemTotal_bytes'
                   AND TimeUnix >= now() - INTERVAL 5 MINUTE) AS total_gb
            FROM otel.otel_metrics_gauge
            WHERE MetricName = 'node_memory_MemAvailable_bytes'
              AND TimeUnix >= now() - INTERVAL 5 MINUTE
        """)
        if not mem_df.empty:
            avail    = float(mem_df.iloc[0]["available_gb"])
            total    = float(mem_df.iloc[0]["total_gb"])
            used_pct = round((1 - avail / total) * 100, 1) if total > 0 else 0
            results["memory"] = {
                "used_pct":     used_pct,
                "available_gb": round(avail, 2),
                "total_gb":     round(total, 2),
            }
    except Exception as e:
        results["memory"] = {"error": str(e)}

    try:
        load_df = query_df("""
            SELECT avg(Value) AS load1 FROM otel.otel_metrics_gauge
            WHERE MetricName = 'node_load1'
              AND TimeUnix >= now() - INTERVAL 5 MINUTE
        """)
        if not load_df.empty:
            results["load"] = {"load1": round(float(load_df.iloc[0]["load1"]), 2)}
    except Exception as e:
        results["load"] = {"error": str(e)}

    try:
        ord_df = query_df("""
            SELECT
                countIf(Timestamp >= now() - INTERVAL 1 HOUR)  AS last_hour,
                countIf(Timestamp >= now() - INTERVAL 2 HOUR
                    AND Timestamp <  now() - INTERVAL 1 HOUR)  AS prev_hour
            FROM otel.otel_traces
            WHERE SpanName = 'POST /orders' AND ParentSpanId = ''
              AND Timestamp >= now() - INTERVAL 2 HOUR
        """)
        if not ord_df.empty:
            last       = int(ord_df.iloc[0]["last_hour"])
            prev       = int(ord_df.iloc[0]["prev_hour"])
            change_pct = round(((last - prev) / prev) * 100, 1) if prev > 0 else 0
            results["orders"] = {
                "last_hour":  last,
                "prev_hour":  prev,
                "change_pct": change_pct,
            }
    except Exception as e:
        results["orders"] = {"error": str(e)}

    return results