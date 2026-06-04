"use client";
import React from 'react';

export interface InfoModule {
  title: string;
  desc: string;
  read: string;
  sample: string;
}

export const INFO_CONTENT: Record<string, InfoModule> = {
  memory: { 
    title: 'Memory Availability Forecast', 
    desc: 'Uses Facebook Prophet (an open-source time-series forecasting algorithm) to analyze historical memory telemetry. It accounts for daily and weekly seasonality to predict future resource availability.', 
    read: 'The solid black line represents actual historical memory availability. The dashed teal line is the forecasted future availability. The lightly shaded area represents the confidence interval. If the forecast approaches 0, a saturation incident is imminent.', 
    sample: `{
  "ds": "2026-04-30 15:00:00",
  "y": null,
  "yhat": 14.2,
  "yhat_lower": 13.8,
  "yhat_upper": 14.6
}` 
  },
  cpu: { 
    title: 'CPU Usage Forecast', 
    desc: 'Uses Facebook Prophet to analyze historical CPU load. It learns from your daily traffic spikes to predict future processor saturation.', 
    read: 'The solid black line is historical CPU usage. The dashed yellow line is the forecast. Approaching 100% indicates likely node starvation.', 
    sample: `{
  "ds": "2026-04-30 15:00:00",
  "yhat": 85.5,
  "yhat_lower": 78.0,
  "yhat_upper": 92.1
}` 
  },
  traffic: { 
    title: 'Order Traffic Forecast', 
    desc: 'Forecasts business metrics (orders per 5 minutes) using historical patterns. Helps capacity planning by predicting inbound load.', 
    read: 'The solid black line shows actual orders. The dashed magenta line shows expected future volume. Large deviations in real-time can indicate a business-level outage.', 
    sample: `{
  "ds": "2026-04-30 15:00:00",
  "yhat": 1200,
  "yhat_lower": 1150,
  "yhat_upper": 1260
}` 
  },
  anomalies: { 
    title: 'Multivariate Anomaly Detection', 
    desc: 'Uses Isolation Forest ML model. Instead of a single static threshold, it looks at multiple metrics simultaneously and flags windows where the combination of metrics is highly unusual.', 
    read: 'Magenta bars are anomalies. Click any bar or table row to see the detail panel showing which metrics deviated the most at that moment.', 
    sample: `{
  "ts": "2026-04-30 08:12:00",
  "anomaly_score": 0.89,
  "is_anomaly": true,
  "contributing": ["payment_failures", "cache_misses"],
  "metrics": { "payment_failures": 45.2, "cache_misses": 8920 }
}` 
  },
  logs: { 
    title: 'Log Pattern Clustering', 
    desc: 'Uses the Drain3 algorithm to strip dynamic variables (IDs, IPs, UUIDs) from log lines and group the remaining structure into templates.', 
    read: 'Look for patterns tagged NEW or ERROR. Dynamic variables are replaced with <*>. Hover the template cell to see the full pattern.', 
    sample: `// Raw: "User user-8891 failed from IP 192.168.1.5"
// Template: "User <*> failed from IP <*>"
{
  "template": "User <*> failed from IP <*>",
  "count": 451,
  "dominant_severity": "WARNING",
  "is_new": false
}` 
  },
  graph: { 
    title: 'Service Dependency Graph', 
    desc: 'Built entirely from your OTel trace data. Every time service A calls service B, a parent-child span relationship is created. This engine reads those relationships and builds a live topology map showing which services talk to which.', 
    read: 'Nodes with red borders have high error rates. The node labelled "root cause" is the deepest node in the error chain — the origin of failures propagating upstream. Edges show request rate and error rate per call path.', 
    sample: `{
  "nodes": [{ "id": "payment-service", "error_rate": 45.2, "status": "error" }],
  "edges": [{ "source": "order-service", "target": "payment-service", "error_rate": 45.2 }],
  "root_cause": "payment-service"
}` 
  },
  correlate: { 
    title: 'Telemetry Correlation Engine', 
    desc: 'Queries ClickHouse across all three signal types simultaneously for the selected time window. Surfaces the slowest traces, error traces, error logs, and metric anomalies in one unified view without you having to jump between tools.', 
    read: 'The slowest traces table shows the top 10 requests by duration. Use the copy buttons next to trace IDs to paste them into Grafana Tempo for the full waterfall. The error logs section shows raw log messages attached to the same time window.', 
    sample: `{
  "slow_traces": [{ "TraceId": "a3f9c12d...", "Duration": 2100000000, "ServiceName": "inventory-service" }],
  "error_logs":  [{ "Body": "Gateway rejected charge", "TraceId": "a3f9c12d..." }]
}` 
  },
  llm: { 
    title: 'AI Incident Summarization', 
    desc: 'Collects error rates, slowest spans, error logs, and metric totals from ClickHouse, then sends them to the Claude/Gemini API. The AI produces a plain-English incident report with root cause, impact, and recommended actions.', 
    read: 'The generated summary is formatted for Slack. Use the Copy button to paste it directly into your incident channel. The quality of the summary improves with more data — run it during or after an incident for best results.', 
    sample: `// Generated output example:
"Payment failure rate spiked to 45% at 08:12 UTC.
Root cause: payment-gateway-call span timing out after 2000ms.
Impact: all checkout flows affected.
Actions: 1) Check Stripe API status 2) Review payment-service logs 3) Consider circuit breaker"` 
  },
  traces: { 
    title: 'Trace Shape Anomaly Detection', 
    desc: 'Every distributed trace has a shape defined by its parent-child spans. This engine fingerprints those shapes and flags deviations — missing spans, extra calls, or N+1 loops.', 
    read: 'Green "baseline" tag = normal execution path. Red tags like "extra spans" or "missing spans" indicate structural failures. Copy trace IDs to search in Grafana Tempo.', 
    sample: `{
  "fingerprint": "POST /orders x1|check-inventory x1|payment-gateway-call x1",
  "is_baseline": false,
  "deviation_type": "missing_spans",
  "example_traces": ["a3f9c12d..."]
}` 
  },
};

interface InfoOverlayProps {
  moduleId: string | null;
  onClose: () => void;
}

export default function InfoOverlay({ moduleId, onClose }: InfoOverlayProps) {
  if (!moduleId) return null;

  const content = INFO_CONTENT[moduleId] || {
    title: 'Module Info',
    desc: 'No information available for this module.',
    read: '—',
    sample: '—'
  };

  return (
    <div className="modal-overlay open" onClick={onClose}>
      <div className="modal" style={{ maxWidth: '650px' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="modal-title">{content.title}</span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body" style={{ gap: '1.25rem', maxHeight: '75vh', overflowY: 'auto' }}>
          <div className="detail-field">
            <div className="detail-label">How it works</div>
            <div className="detail-desc">{content.desc}</div>
          </div>
          <div className="detail-field">
            <div className="detail-label">How to read the data</div>
            <div className="detail-desc">{content.read}</div>
          </div>
          <div className="detail-field">
            <div className="detail-label">Sample Data (JSON / Format)</div>
            <div className="code-block" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '250px', overflowY: 'auto' }}>{content.sample}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
