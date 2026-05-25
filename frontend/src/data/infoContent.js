export const INFO_CONTENT = {
  memory: {
    title: 'Memory Availability Forecast',
    desc: 'Uses Facebook Prophet to analyze historical memory telemetry. It accounts for daily and weekly seasonality to predict future resource availability.',
    read: 'The solid black line represents actual historical memory availability. The dashed teal line is the forecasted future availability. The lightly shaded area represents the confidence interval. If the forecast approaches 0, a saturation incident is imminent.',
    sample: '{\n  "ds": "2026-04-30 15:00:00",\n  "y": null,\n  "yhat": 14.2,\n  "yhat_lower": 13.8,\n  "yhat_upper": 14.6\n}',
  },
  cpu: {
    title: 'CPU Usage Forecast',
    desc: 'Uses Facebook Prophet to analyze historical CPU load. It learns from your daily traffic spikes to predict future processor saturation.',
    read: 'The solid black line is historical CPU usage. The dashed yellow line is the forecast. Approaching 100% indicates likely node starvation.',
    sample: '{\n  "ds": "2026-04-30 15:00:00",\n  "yhat": 85.5,\n  "yhat_lower": 78.0,\n  "yhat_upper": 92.1\n}',
  },
  traffic: {
    title: 'Order Traffic Forecast',
    desc: 'Forecasts business metrics using historical patterns. Helps capacity planning by predicting inbound load.',
    read: 'The solid black line shows actual orders. The dashed magenta line shows expected future volume. Large deviations in real-time can indicate a business-level outage.',
    sample: '{\n  "ds": "2026-04-30 15:00:00",\n  "yhat": 1200,\n  "yhat_lower": 1150,\n  "yhat_upper": 1260\n}',
  },
  anomalies: {
    title: 'Multivariate Anomaly Detection',
    desc: 'Uses Isolation Forest. Instead of a single static threshold, it looks at multiple metrics simultaneously and flags windows where the combination of metrics is highly unusual.',
    read: 'Magenta bars are anomalies. Click any bar or table row to see the detail panel showing which metrics deviated the most at that moment.',
    sample: '{\n  "ts": "2026-04-30 08:12:00",\n  "anomaly_score": 0.89,\n  "is_anomaly": true,\n  "contributing": ["payment_failures", "cache_misses"],\n  "metrics": { "payment_failures": 45.2, "cache_misses": 8920 }\n}',
  },
  logs: {
    title: 'Log Pattern Clustering',
    desc: 'Uses Drain3 to strip dynamic variables from log lines and group the remaining structure into templates.',
    read: 'Look for patterns tagged NEW or ERROR. Dynamic variables are replaced with <*>. Hover the template cell to see the full pattern.',
    sample: '// Raw: "User user-8891 failed from IP 192.168.1.5"\n// Template: "User <*> failed from IP <*>"\n{\n  "template": "User <*> failed from IP <*>",\n  "count": 451,\n  "dominant_severity": "WARNING",\n  "is_new": false\n}',
  },
  graph: {
    title: 'Service Dependency Graph',
    desc: 'Built entirely from OTel trace data. Every time service A calls service B, a parent-child span relationship is created. This engine reads those relationships and builds a live topology map.',
    read: 'Nodes with red borders have high error rates. The node labelled root cause is the deepest node in the error chain. Edges show request rate and error rate per call path.',
    sample: '{\n  "nodes": [{ "id": "payment-service", "error_rate": 45.2, "status": "error" }],\n  "edges": [{ "source": "order-service", "target": "payment-service", "error_rate": 45.2 }],\n  "root_cause": "payment-service"\n}',
  },
  correlate: {
    title: 'Telemetry Correlation Engine',
    desc: 'Queries ClickHouse across all three signal types simultaneously for the selected time window. Surfaces the slowest traces, error traces, error logs, and metric anomalies in one view.',
    read: 'The slowest traces table shows the top 10 requests by duration. Use the copy buttons next to trace IDs to paste them into Grafana Tempo for the full waterfall.',
    sample: '{\n  "slow_traces": [{ "TraceId": "a3f9c12d...", "Duration": 2100000000, "ServiceName": "inventory-service" }],\n  "error_logs":  [{ "Body": "Gateway rejected charge", "TraceId": "a3f9c12d..." }]\n}',
  },
  llm: {
    title: 'AI Incident Summarization',
    desc: 'Collects error rates, slowest spans, error logs, and metric totals from ClickHouse, then sends them to the LLM API. It produces a plain-English incident report with root cause, impact, and recommended actions.',
    read: 'The generated summary is formatted for Slack. Use the Copy button to paste it directly into your incident channel.',
    sample: 'Payment failure rate spiked to 45% at 08:12 UTC.\nRoot cause: payment-gateway-call span timing out after 2000ms.\nImpact: all checkout flows affected.\nActions: 1. Check Stripe API status 2. Review payment-service logs 3. Consider circuit breaker',
  },
  traces: {
    title: 'Trace Shape Anomaly Detection',
    desc: 'Every distributed trace has a shape defined by its parent-child spans. This engine fingerprints those shapes and flags deviations: missing spans, extra calls, or loops.',
    read: 'Green baseline tag means normal execution path. Red tags like extra spans or missing spans indicate structural failures. Copy trace IDs to search in Grafana Tempo.',
    sample: '{\n  "fingerprint": "POST /orders x1|check-inventory x1|payment-gateway-call x1",\n  "is_baseline": false,\n  "deviation_type": "missing_spans",\n  "example_traces": ["a3f9c12d..."]\n}',
  },
};
