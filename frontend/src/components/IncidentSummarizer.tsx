'use client';

import React, { useState } from 'react';
import { LLMIncidentSummaryResponse } from '../types/otel';

interface IncidentSummarizerProps {
  apiBase: string;
  onShowInfo: (id: string) => void;
}

export default function IncidentSummarizer({ apiBase, onShowInfo }: IncidentSummarizerProps) {
  const [service, setService] = useState('');
  const [hours, setHours] = useState('1');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<LLMIncidentSummaryResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const generateSummary = async () => {
    setLoading(true);
    setErrorText(null);
    setData(null);
    try {
      const res = await fetch(`${apiBase}/diagnostic/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hours: Number(hours), service })
      });
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

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(data.summary).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="panel-title">AI incident summarization</div>
          <div className="panel-meta">Powered by Gemini 2.5 Flash — generates a plain-English RCA ready to post to Slack</div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            className="select-sm"
            value={service}
            onChange={(e) => setService(e.target.value)}
          >
            <option value="">All services</option>
            <option value="order-service">order-service</option>
            <option value="inventory-service">inventory-service</option>
            <option value="payment-service">payment-service</option>
          </select>
          <select
            className="select-sm"
            value={hours}
            onChange={(e) => setHours(e.target.value)}
          >
            <option value="1">Last 1h</option>
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            className="btn-sm"
            onClick={generateSummary}
            disabled={loading}
          >
            {loading ? 'Synthesizing...' : 'Generate summary'}
          </button>
          <button
            className="btn-info"
            onClick={() => onShowInfo('llm')}
            title="How this works"
          >
            i
          </button>
        </div>
      </div>
      <div className="panel-body">
        {loading && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Gemini 2.5 Flash is analyzing logs, error rates, and metrics to compile RCA...
          </div>
        )}

        {!loading && errorText && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none', color: 'var(--red)' }}>
            {errorText}
          </div>
        )}

        {!loading && !errorText && !data && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Click &quot;Generate summary&quot; to have Gemini 2.5 Flash analyse your ClickHouse telemetry and produce an incident report.
          </div>
        )}

        {!loading && !errorText && data && (
          <div id="llm-wrap">
            <div
              id="llm-summary"
              style={{
                background: 'var(--surface2)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                padding: '1.25rem',
                fontSize: '0.9rem',
                lineHeight: 1.7,
                color: 'var(--text-2)',
                whiteSpace: 'pre-wrap',
                marginBottom: '1rem'
              }}
            >
              {data.summary}
            </div>
            <button className="btn-sm" onClick={handleCopy}>
              {copied ? 'Copied!' : 'Copy to clipboard'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
