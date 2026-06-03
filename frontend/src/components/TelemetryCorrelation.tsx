'use client';

import React, { useState } from 'react';
import { Info, Copy, Clock, AlertTriangle, AlertCircle, BarChart3 } from 'lucide-react';
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

  return (
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex flex-wrap items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <div>
          <span className="font-bold text-sm text-text-primary block">Telemetry Correlation Engine</span>
          <span className="text-[11px] font-semibold text-text-tertiary">
            Correlates slowest execution spans, application error logs, and metric totals in a single temporal window
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={service}
            onChange={(e) => setService(e.target.value)}
            className="text-xs px-2.5 py-1.5 border border-border-medium rounded-md font-semibold bg-bg-main text-text-secondary outline-none focus:border-indosat-teal transition-all"
          >
            <option value="">All services</option>
            <option value="order-service">order-service</option>
            <option value="inventory-service">inventory-service</option>
            <option value="payment-service">payment-service</option>
          </select>
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="text-xs px-2.5 py-1.5 border border-border-medium rounded-md font-semibold bg-bg-main text-text-secondary outline-none focus:border-indosat-teal transition-all"
          >
            <option value="1">Last 1h</option>
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchCorrelation}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm"
          >
            {loading ? 'Correlating...' : 'Correlate'}
          </button>
          <button
            onClick={() => onShowInfo('correlate')}
            className="w-6 h-6 rounded-full border border-border-medium bg-surface-card hover:bg-indosat-teal hover:border-indosat-teal hover:text-white flex items-center justify-center cursor-pointer transition-all duration-150"
            title="How this works"
          >
            <Info className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-5">
        {errorText ? (
          <div className="text-center text-status-error font-semibold text-sm py-4">{errorText}</div>
        ) : loading ? (
          <div className="flex flex-col items-center justify-center py-12 select-none">
            <div className="w-8 h-8 border-4 border-indosat-teal border-t-transparent rounded-full animate-spin mb-3" />
            <div className="text-xs font-bold text-text-secondary">
              Querying traces, metrics, and logs in ClickHouse for aligned temporal bounds...
            </div>
          </div>
        ) : data ? (
          <div className="space-y-6 animate-fade">
            {/* 1. Slowest Traces */}
            <div>
              <div className="flex items-center gap-2 mb-2 select-none text-text-primary">
                <Clock className="w-4 h-4 text-text-tertiary" />
                <span className="text-xs font-bold uppercase tracking-wider">Slowest Spans ({data.slow_traces.length})</span>
              </div>
              {data.slow_traces.length > 0 ? (
                <div className="overflow-x-auto border border-border-subtle rounded-lg">
                  <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                    <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                      <tr>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Trace ID</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Service</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Span Name</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Duration</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">OTel Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
                      {data.slow_traces.map((trace, idx) => (
                        <tr key={idx} className="hover:bg-surface-hover/30 transition-colors">
                          <td className="px-4 py-2 font-mono text-[10px] text-text-tertiary">
                            <div className="flex items-center gap-1.5">
                              <span>{trace.TraceId.slice(0, 16)}…</span>
                              <button
                                onClick={() => handleCopyText(trace.TraceId)}
                                className={`px-1 py-0.5 rounded border border-border-subtle text-[8px] font-bold transition-all flex items-center gap-1 cursor-pointer ${
                                  copiedId === trace.TraceId ? 'text-indosat-teal border-indosat-teal/30 bg-status-ok-bg' : 'text-text-tertiary'
                                }`}
                              >
                                {copiedId === trace.TraceId ? 'copied' : 'copy'}
                              </button>
                            </div>
                          </td>
                          <td className="px-4 py-2">
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-bg-main border border-border-subtle font-bold">
                              {trace.ServiceName}
                            </span>
                          </td>
                          <td className="px-4 py-2 font-mono text-text-primary text-[11px]">{trace.SpanName}</td>
                          <td className="px-4 py-2 select-none">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${Number(trace.Duration) / 1e6 > 1000 ? 'bg-status-error-bg text-indosat-magenta' : 'bg-status-warning-bg text-status-warning'}`}>
                              {(Number(trace.Duration) / 1e6).toFixed(0)}ms
                            </span>
                          </td>
                          <td className="px-4 py-2 select-none">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${trace.StatusCode === 'STATUS_CODE_ERROR' ? 'bg-status-error-bg text-indosat-magenta' : 'bg-status-ok-bg text-indosat-teal'}`}>
                              {trace.StatusCode === 'STATUS_CODE_ERROR' ? 'error' : 'ok'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs font-bold text-text-tertiary py-1 text-center border-b border-border-subtle pb-3">None found.</div>
              )}
            </div>

            {/* 2. Error Traces */}
            <div>
              <div className="flex items-center gap-2 mb-2 select-none text-text-primary">
                <AlertTriangle className="w-4 h-4 text-indosat-magenta" />
                <span className="text-xs font-bold uppercase tracking-wider">Error Traces ({data.error_traces.length})</span>
              </div>
              {data.error_traces.length > 0 ? (
                <div className="overflow-x-auto border border-border-subtle rounded-lg">
                  <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                    <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                      <tr>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Trace ID</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Service</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Failed Span</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Timestamp</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
                      {data.error_traces.map((trace, idx) => (
                        <tr key={idx} className="hover:bg-surface-hover/30 transition-colors">
                          <td className="px-4 py-2 font-mono text-[10px] text-text-tertiary">
                            <div className="flex items-center gap-1.5">
                              <span>{trace.TraceId.slice(0, 16)}…</span>
                              <button
                                onClick={() => handleCopyText(trace.TraceId)}
                                className={`px-1 py-0.5 rounded border border-border-subtle text-[8px] font-bold transition-all flex items-center gap-1 cursor-pointer ${
                                  copiedId === trace.TraceId ? 'text-indosat-teal border-indosat-teal/30 bg-status-ok-bg' : 'text-text-tertiary'
                                }`}
                              >
                                {copiedId === trace.TraceId ? 'copied' : 'copy'}
                              </button>
                            </div>
                          </td>
                          <td className="px-4 py-2">
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-status-error-bg text-indosat-magenta border border-indosat-magenta/10 font-bold">
                              {trace.ServiceName}
                            </span>
                          </td>
                          <td className="px-4 py-2 font-mono text-text-primary text-[11px]">{trace.SpanName}</td>
                          <td className="px-4 py-2 text-text-tertiary select-none font-mono">
                            {String(trace.Timestamp).replace('T', ' ').slice(0, 16)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs font-bold text-text-tertiary py-1 text-center border-b border-border-subtle pb-3">None found.</div>
              )}
            </div>

            {/* 3. Error Logs */}
            <div>
              <div className="flex items-center gap-2 mb-2 select-none text-text-primary">
                <AlertCircle className="w-4 h-4 text-indosat-magenta" />
                <span className="text-xs font-bold uppercase tracking-wider">Error Logs ({data.error_logs.length})</span>
              </div>
              {data.error_logs.length > 0 ? (
                <div className="overflow-x-auto border border-border-subtle rounded-lg">
                  <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                    <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                      <tr>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Trace Link</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Service</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Log Message (Body)</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary font-semibold">
                      {data.error_logs.map((log, idx) => (
                        <tr key={idx} className="hover:bg-surface-hover/30 transition-colors">
                          <td className="px-4 py-2 font-mono text-[10px] text-text-tertiary">
                            {log.TraceId ? (
                              <div className="flex items-center gap-1.5">
                                <span>{log.TraceId.slice(0, 16)}…</span>
                                <button
                                  onClick={() => handleCopyText(log.TraceId!)}
                                  className={`px-1 py-0.5 rounded border border-border-subtle text-[8px] font-bold transition-all flex items-center gap-1 cursor-pointer ${
                                    copiedId === log.TraceId ? 'text-indosat-teal border-indosat-teal/30 bg-status-ok-bg' : 'text-text-tertiary'
                                  }`}
                                >
                                  {copiedId === log.TraceId ? 'copied' : 'copy'}
                                </button>
                              </div>
                            ) : (
                              <span className="text-text-tertiary">—</span>
                            )}
                          </td>
                          <td className="px-4 py-2">
                            {log.service_name ? (
                              <span className="px-1.5 py-0.5 rounded text-[10px] bg-status-error-bg text-indosat-magenta border border-indosat-magenta/10 font-bold">
                                {log.service_name}
                              </span>
                            ) : (
                              <span className="text-text-tertiary">—</span>
                            )}
                          </td>
                          <td className="px-4 py-2 text-text-primary text-[11px] max-w-[320px] truncate select-all" title={log.Body}>
                            {log.Body}
                          </td>
                          <td className="px-4 py-2 text-text-tertiary select-none font-mono whitespace-nowrap">
                            {String(log.Timestamp).replace('T', ' ').slice(0, 16)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs font-bold text-text-tertiary py-1 text-center border-b border-border-subtle pb-3">None found.</div>
              )}
            </div>

            {/* 4. Metric Totals */}
            <div>
              <div className="flex items-center gap-2 mb-2 select-none text-text-primary">
                <BarChart3 className="w-4 h-4 text-text-tertiary" />
                <span className="text-xs font-bold uppercase tracking-wider">Metric Totals ({data.metric_summary.length})</span>
              </div>
              {data.metric_summary.length > 0 ? (
                <div className="overflow-x-auto border border-border-subtle rounded-lg">
                  <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                    <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                      <tr>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Metric Name</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Total Sum / Count</th>
                        <th className="px-4 py-2 text-left uppercase tracking-wider">Avg Rate Per Minute</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
                      {data.metric_summary.map((metric, idx) => (
                        <tr key={idx} className="hover:bg-surface-hover/30 transition-colors">
                          <td className="px-4 py-2 font-mono text-text-primary text-[11.5px]">{metric.MetricName}</td>
                          <td className="px-4 py-2 font-bold text-text-primary select-none">{Number(metric.total).toFixed(0)}</td>
                          <td className="px-4 py-2 text-text-tertiary select-none">{(Number(metric.avg_val) || 0).toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs font-bold text-text-tertiary py-1 text-center border-b border-border-subtle pb-3">None found.</div>
              )}
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            Click &quot;Correlate&quot; to surface aligned traces, error logs, and metric anomalies side-by-side.
          </div>
        )}
      </div>
    </div>
  );
}
