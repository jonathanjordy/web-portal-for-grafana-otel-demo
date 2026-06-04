'use client';

import React, { useState } from 'react';
import { TraceShapeResponse } from '../types/otel';

interface TraceShapesProps {
  apiBase: string;
  onShowInfo: (id: string) => void;
}

export default function TraceShapes({ apiBase, onShowInfo }: TraceShapesProps) {
  const [hours, setHours] = useState('2');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<TraceShapeResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const fetchTraceShapes = async () => {
    setLoading(true);
    setErrorText(null);
    try {
      const res = await fetch(`${apiBase}/detective/trace-shapes?hours=${hours}`);
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

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="panel-title">Trace shape anomaly detection</div>
          <div className="panel-meta">Fingerprints each trace&apos;s span structure — flags deviations from baseline</div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="select-sm"
          >
            <option value="1">Last 1h</option>
            <option value="2">Last 2h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchTraceShapes}
            disabled={loading}
            className="btn-sm"
          >
            {loading ? 'Analyzing...' : 'Analyse traces'}
          </button>
          <button
            onClick={() => onShowInfo('traces')}
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
            Fingerprinting trace shapes...
          </div>
        )}

        {!loading && errorText && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none', color: 'var(--red)' }}>
            {errorText}
          </div>
        )}

        {!loading && !errorText && !data && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Click &quot;Analyse traces&quot; to fingerprint trace shapes.
          </div>
        )}

        {!loading && !errorText && data && (
          <div>
            <div className="stat-row" style={{ marginBottom: '1rem' }}>
              <div className="stat">
                <div className="stat-label">Total traces</div>
                <div className="stat-value">{data.total_traces}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Unique shapes</div>
                <div className="stat-value blue">{data.unique_shapes}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Anomalous traces</div>
                <div className="stat-value red">{data.anomalous_count}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Baseline coverage</div>
                <div className="stat-value green">{data.baseline_pct}%</div>
              </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="det-table">
                <thead>
                  <tr>
                    <th>Shape</th>
                    <th>Count</th>
                    <th>%</th>
                    <th>Deviation</th>
                    <th>Example trace IDs</th>
                  </tr>
                </thead>
                <tbody>
                  {data.shape_summary.slice(0, 10).map((s, idx) => (
                    <tr key={idx}>
                      <td
                        style={{
                          maxWidth: '260px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontFamily: 'monospace',
                          fontSize: '0.72rem',
                          cursor: 'help'
                        }}
                        title={s.fingerprint}
                      >
                        {s.fingerprint}
                      </td>
                      <td>{s.count}</td>
                      <td style={{ color: 'var(--text-3)' }}>{s.pct_of_total}%</td>
                      <td>
                        {s.is_baseline ? (
                          <span className="tag green">baseline</span>
                        ) : s.deviation_type ? (
                          <span className="tag red">{s.deviation_type.replace(/_/g, ' ')}</span>
                        ) : (
                          <span className="tag">—</span>
                        )}
                      </td>
                      <td>
                        {s.example_traces && s.example_traces.length > 0 ? (
                          s.example_traces.map((tid, tIdx) => (
                            <div key={tIdx} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '3px' }}>
                              <span style={{ fontFamily: 'monospace', fontSize: '0.72rem', color: 'var(--text-3)' }}>{tid}</span>
                              <button
                                onClick={() => handleCopyText(tid)}
                                title="Copy trace ID"
                                style={{
                                  background: 'none',
                                  border: '1px solid var(--border)',
                                  borderRadius: '4px',
                                  padding: '1px 5px',
                                  fontSize: '0.65rem',
                                  cursor: 'pointer',
                                  color: 'var(--text-3)',
                                  fontFamily: 'var(--sans)'
                                }}
                              >
                                {copiedId === tid ? 'copied' : 'copy'}
                              </button>
                            </div>
                          ))
                        ) : (
                          <span>—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
