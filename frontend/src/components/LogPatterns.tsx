'use client';

import React, { useState } from 'react';
import { LogPatternResponse } from '../types/otel';

interface LogPatternsProps {
  apiBase: string;
  onShowInfo: (id: string) => void;
}

export default function LogPatterns({ apiBase, onShowInfo }: LogPatternsProps) {
  const [hours, setHours] = useState('2');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<LogPatternResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);

  const fetchPatterns = async () => {
    setLoading(true);
    setErrorText(null);
    try {
      const res = await fetch(`${apiBase}/detective/log-patterns?hours=${hours}`);
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

  return (
    <div className="panel">
      {/* Header */}
      <div className="panel-head">
        <div>
          <div className="panel-title">Log pattern clustering</div>
          <div className="panel-meta">Drain3 template mining — strips variables, groups logs by structure</div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            id="log-hours"
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="select-sm"
          >
            <option value="1">Last 1h</option>
            <option value="2">Last 2h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchPatterns}
            disabled={loading}
            className="btn-sm"
          >
            {loading ? 'Clustering...' : 'Analyse logs'}
          </button>
          <button
            onClick={() => onShowInfo('logs')}
            className="btn-info"
            title="How this works"
          >
            i
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="panel-body">
        {errorText ? (
          <div className="empty" style={{ padding: '1.5rem', border: 'none', color: 'var(--red)' }}>
            {errorText}
          </div>
        ) : loading ? (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Clustering log templates with Drain3...
          </div>
        ) : data ? (
          <div>
            {/* Stats */}
            <div className="stat-row" style={{ marginBottom: '1rem' }}>
              <div className="stat">
                <div className="stat-label">Total logs</div>
                <div className="stat-value">{data.total_logs}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Unique patterns</div>
                <div className="stat-value blue">{data.unique_patterns}</div>
              </div>
              <div className="stat">
                <div className="stat-label">New patterns</div>
                <div className="stat-value amber">{data.new_patterns}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Error patterns</div>
                <div className="stat-value red">{data.error_patterns}</div>
              </div>
            </div>

            {/* Pattern Table */}
            <div style={{ overflowX: 'auto' }}>
              <table className="det-table">
                <thead>
                  <tr>
                    <th>Pattern template</th>
                    <th>Count</th>
                    <th>%</th>
                    <th>Service</th>
                    <th>Severity</th>
                    <th>Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {data.patterns.map((p, idx) => (
                    <tr key={idx}>
                      <td 
                        style={{ 
                          maxWidth: '320px', 
                          overflow: 'hidden', 
                          textOverflow: 'ellipsis', 
                          whiteSpace: 'nowrap', 
                          fontFamily: 'monospace', 
                          fontSize: '0.75rem' 
                        }} 
                        title={p.template}
                      >
                        {p.template}
                      </td>
                      <td>{p.count}</td>
                      <td style={{ color: 'var(--text-3)' }}>{p.pct_of_total}%</td>
                      <td>
                        <span className="tag">{p.top_service}</span>
                      </td>
                      <td>
                        <span className={`tag ${p.dominant_severity === 'ERROR' ? 'red' : p.dominant_severity === 'WARNING' ? 'amber' : 'blue'}`}>
                          {p.dominant_severity}
                        </span>
                      </td>
                      <td>
                        {p.is_new && <span className="tag amber">NEW</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Click &quot;Analyse logs&quot; to cluster log templates using Drain3.
          </div>
        )}
      </div>
    </div>
  );
}
