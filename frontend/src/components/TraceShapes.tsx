'use client';

import React, { useState } from 'react';
import { Info, Cpu, Network, CheckCircle2, Copy } from 'lucide-react';
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
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex flex-wrap items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <div>
          <span className="font-bold text-sm text-text-primary block">Trace Shape Anomaly Detection</span>
          <span className="text-[11px] font-semibold text-text-tertiary">
            Fingerprints each trace&apos;s span structure — flags deviations (loops, omissions) from baseline
          </span>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="text-xs px-2.5 py-1.5 border border-border-medium rounded-md font-semibold bg-bg-main text-text-secondary outline-none focus:border-indosat-teal transition-all"
          >
            <option value="1">Last 1h</option>
            <option value="2">Last 2h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchTraceShapes}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm"
          >
            {loading ? 'Analyzing...' : 'Analyse traces'}
          </button>
          <button
            onClick={() => onShowInfo('traces')}
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
              Mapping child trace relationships and computing topological structural signatures...
            </div>
          </div>
        ) : data ? (
          <div className="animate-fade">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5 select-none">
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-text-secondary">
                  <Network className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Total Traces</span>
                  <span className="text-lg font-extrabold text-text-primary">{data.total_traces}</span>
                </div>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-indosat-teal bg-status-ok-bg">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Unique Shapes</span>
                  <span className="text-lg font-extrabold text-indosat-teal">{data.unique_shapes}</span>
                </div>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-indosat-magenta bg-status-error-bg">
                  <span className="text-xs font-bold">⚠</span>
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Anomalous Traces</span>
                  <span className={`text-lg font-extrabold ${data.anomalous_count > 0 ? 'text-indosat-magenta' : 'text-text-primary'}`}>
                    {data.anomalous_count}
                  </span>
                </div>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-indosat-teal bg-status-ok-bg">
                  <CheckCircle2 className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Baseline Coverage</span>
                  <span className="text-lg font-extrabold text-indosat-teal">{data.baseline_pct}%</span>
                </div>
              </div>
            </div>

            {/* Shape table */}
            <div className="overflow-x-auto border border-border-subtle rounded-lg">
              <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                  <tr>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Topology Shape (Spans x Call Counts)</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Count</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">%</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Deviation Type</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Example Trace IDs</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
                  {data.shape_summary.slice(0, 10).map((shape, idx) => (
                    <tr key={idx} className="hover:bg-surface-hover/30 transition-colors">
                      <td className="px-4 py-3 max-w-[280px] font-mono text-text-primary text-[11px] truncate cursor-help select-all" title={shape.fingerprint}>
                        {shape.fingerprint}
                      </td>
                      <td className="px-4 py-3 font-bold select-none">{shape.count}</td>
                      <td className="px-4 py-3 text-text-tertiary select-none">{shape.pct_of_total}%</td>
                      <td className="px-4 py-3 select-none">
                        {shape.is_baseline ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-status-ok-bg text-indosat-teal border border-indosat-teal/10">
                            baseline
                          </span>
                        ) : shape.deviation_type ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-status-error-bg text-indosat-magenta border border-indosat-magenta/10">
                            {shape.deviation_type.replace(/_/g, ' ')}
                          </span>
                        ) : (
                          <span className="text-text-tertiary">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 select-none">
                        <div className="flex flex-col gap-1.5">
                          {shape.example_traces.map((tid, tIdx) => (
                            <div key={tIdx} className="flex items-center gap-2">
                              <span className="font-mono text-[10px] text-text-tertiary">{tid}</span>
                              <button
                                onClick={() => handleCopyText(tid)}
                                className={`px-1.5 py-0.5 rounded border border-border-subtle text-[9px] font-bold transition-all flex items-center gap-1 cursor-pointer hover:bg-surface-hover ${
                                  copiedId === tid ? 'text-indosat-teal border-indosat-teal/30 bg-status-ok-bg' : 'text-text-tertiary'
                                }`}
                              >
                                <Copy className="w-2.5 h-2.5" />
                                {copiedId === tid ? 'copied' : 'copy'}
                              </button>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            Click &quot;Analyse traces&quot; to fingerprint span topological shapes.
          </div>
        )}
      </div>
    </div>
  );
}
