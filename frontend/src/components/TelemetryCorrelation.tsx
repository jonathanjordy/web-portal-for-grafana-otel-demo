'use client';

import React, { useState } from 'react';
import { TelemetryCorrelationResponse } from '../types/otel';

interface TelemetryCorrelationProps {
  apiBase: string;
  onShowInfo: (id: string) => void;
}

export default function TelemetryCorrelation({ apiBase, onShowInfo }: TelemetryCorrelationProps) {
  const [service, setService] = useState('');
  const [hours, setHours] = useState('1');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<TelemetryCorrelationResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const fetchCorrelation = async () => {
    setLoading(true);
    setErrorText(null);
    try {
      const params = new URLSearchParams({ hours, service });
      const res = await fetch(`${apiBase}/diagnostic/correlate?${params}`);
      const result = await res.json();
      
      if (result.detail) {
        setErrorText(result.detail);
        return;
      }
      
      setData(result);
    } catch (err: any) {
      setErrorText('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(text);
      setTimeout(() => setCopiedId(null), 1500);
    });
  };

  const renderSection = (title: string, rows: any[], cols: string[], renderRow: (r: any, i: number) => React.ReactNode) => (
    <div style={{ marginBottom: '1.25rem' }}>
      <div
        style={{
          fontSize: '0.78rem',
          fontWeight: 700,
          color: 'var(--text-3)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          marginBottom: '0.5rem'
        }}
      >
        {title} ({rows.length})
      </div>
      {rows.length > 0 ? (
        <table className="det-table">
          <thead>
            <tr>
              {cols.map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>{rows.map(renderRow)}</tbody>
        </table>
      ) : (
        <div style={{ color: 'var(--text-3)', fontSize: '0.82rem' }}>None found.</div>
      )}
    </div>
  );

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="panel-title">Telemetry correlation engine</div>
          <div className="panel-meta">Slowest traces · error logs · metric spikes — all in one view</div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            value={service}
            onChange={(e) => setService(e.target.value)}
            className="select-sm"
          >
            <option value="">All services</option>
            <option value="order-service">order-service</option>
            <option value="inventory-service">inventory-service</option>
            <option value="payment-service">payment-service</option>
          </select>
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="select-sm"
          >
            <option value="1">Last 1h</option>
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchCorrelation}
            disabled={loading}
            className="btn-sm"
          >
            {loading ? 'Correlating...' : 'Correlate'}
          </button>
          <button
            onClick={() => onShowInfo('correlate')}
            className="btn-info"
            title="How this works"
          >
            i
          </button>
        </div>
      </div>

      <div className="panel-body">
        {loading && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Correlating telemetry signals...
          </div>
        )}

        {!loading && errorText && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none', color: 'var(--red)' }}>
            {errorText}
          </div>
        )}

        {!loading && !errorText && !data && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Click &quot;Correlate&quot; to surface the slowest traces, error logs, and metric anomalies side by side.
          </div>
        )}

        {!loading && !errorText && data && (
          <div id="corr-wrap">
            {renderSection(
              'Slowest traces',
              data.slow_traces,
              ['Trace ID', 'Service', 'Span', 'Duration', 'Status'],
              (r, idx) => (
                <tr key={idx}>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}>
                    {r.TraceId.slice(0, 16)}…
                    <button
                      onClick={() => handleCopyText(r.TraceId)}
                      style={{
                        background: 'none',
                        border: '1px solid var(--border)',
                        borderRadius: '4px',
                        padding: '1px 4px',
                        fontSize: '0.62rem',
                        cursor: 'pointer',
                        color: 'var(--text-3)',
                        marginLeft: '4px'
                      }}
                    >
                      {copiedId === r.TraceId ? 'copied' : 'copy'}
                    </button>
                  </td>
                  <td>
                    <span className="tag">{r.ServiceName}</span>
                  </td>
                  <td style={{ fontSize: '0.78rem' }}>{r.SpanName}</td>
                  <td>
                    <span className={`tag ${Number(r.Duration) / 1e6 > 1000 ? 'red' : 'amber'}`}>
                      {(Number(r.Duration) / 1e6).toFixed(0)}ms
                    </span>
                  </td>
                  <td>
                    <span className={`tag ${r.StatusCode === 'STATUS_CODE_ERROR' ? 'red' : 'green'}`}>
                      {r.StatusCode === 'STATUS_CODE_ERROR' ? 'error' : 'ok'}
                    </span>
                  </td>
                </tr>
              )
            )}

            {renderSection(
              'Error traces',
              data.error_traces,
              ['Trace ID', 'Service', 'Span', 'Timestamp'],
              (r, idx) => (
                <tr key={idx}>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}>
                    {r.TraceId.slice(0, 16)}…
                    <button
                      onClick={() => handleCopyText(r.TraceId)}
                      style={{
                        background: 'none',
                        border: '1px solid var(--border)',
                        borderRadius: '4px',
                        padding: '1px 4px',
                        fontSize: '0.62rem',
                        cursor: 'pointer',
                        color: 'var(--text-3)',
                        marginLeft: '4px'
                      }}
                    >
                      {copiedId === r.TraceId ? 'copied' : 'copy'}
                    </button>
                  </td>
                  <td>
                    <span className="tag red">{r.ServiceName}</span>
                  </td>
                  <td style={{ fontSize: '0.78rem' }}>{r.SpanName}</td>
                  <td style={{ fontSize: '0.75rem', color: 'var(--text-3)' }}>
                    {String(r.Timestamp).slice(0, 16)}
                  </td>
                </tr>
              )
            )}

            {renderSection(
              'Error logs',
              data.error_logs,
              ['Trace ID', 'Service', 'Message', 'Time'],
              (r, idx) => (
                <tr key={idx}>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}>
                    {r.TraceId ? (
                      <>
                        <span style={{ color: 'var(--text-3)' }}>{r.TraceId.slice(0, 16)}…</span>
                        <button
                          onClick={() => handleCopyText(r.TraceId)}
                          style={{
                            background: 'none',
                            border: '1px solid var(--border)',
                            borderRadius: '4px',
                            padding: '1px 4px',
                            cursor: 'pointer',
                            color: 'var(--text-3)',
                            marginLeft: '4px'
                          }}
                        >
                          {copiedId === r.TraceId ? 'copied' : 'copy'}
                        </button>
                      </>
                    ) : (
                      <span style={{ color: 'var(--text-3)' }}>—</span>
                    )}
                  </td>
                  <td>
                    <span className="tag red">{r.service_name || '—'}</span>
                  </td>
                  <td
                    style={{
                      fontSize: '0.78rem',
                      maxWidth: '300px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}
                    title={r.Body}
                  >
                    {r.Body}
                  </td>
                  <td style={{ fontSize: '0.75rem', color: 'var(--text-3)', whiteSpace: 'nowrap' }}>
                    {String(r.Timestamp).slice(0, 16)}
                  </td>
                </tr>
              )
            )}

            {renderSection(
              'Metric totals',
              data.metric_summary,
              ['Metric', 'Total', 'Avg per min'],
              (r, idx) => (
                <tr key={idx}>
                  <td>
                    <span className="tag">{r.MetricName}</span>
                  </td>
                  <td style={{ fontWeight: 700 }}>{Number(r.total).toFixed(0)}</td>
                  <td style={{ color: 'var(--text-3)' }}>{(Number(r.avg_val) || 0).toFixed(2)}</td>
                </tr>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
