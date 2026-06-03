'use client';

import React, { useState } from 'react';
import { Info, Logs, FileText, Sparkles, AlertOctagon } from 'lucide-react';
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

  const getSeverityBadgeColor = (sev: string) => {
    switch (sev) {
      case 'ERROR': return 'bg-status-error-bg text-indosat-magenta border border-indosat-magenta/10';
      case 'WARNING': return 'bg-status-warning-bg text-status-warning border border-status-warning/10';
      case 'INFO': default: return 'bg-status-ok-bg text-indosat-teal border border-indosat-teal/10';
    }
  };

  return (
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex flex-wrap items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <div>
          <span className="font-bold text-sm text-text-primary block">Log Pattern Clustering</span>
          <span className="text-[11px] font-semibold text-text-tertiary">
            Drain3 template mining — strips dynamic variables, clusters logs by structure
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
            onClick={fetchPatterns}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm"
          >
            {loading ? 'Clustering...' : 'Analyse logs'}
          </button>
          <button
            onClick={() => onShowInfo('logs')}
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
              Parsing log streams and running Drain3 online template clustering...
            </div>
          </div>
        ) : data ? (
          <div className="animate-fade">
            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5 select-none">
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-text-secondary">
                  <Logs className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Total Logs</span>
                  <span className="text-lg font-extrabold text-text-primary">{data.total_logs}</span>
                </div>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-indosat-teal bg-status-ok-bg">
                  <FileText className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Unique Templates</span>
                  <span className="text-lg font-extrabold text-indosat-teal">{data.unique_patterns}</span>
                </div>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-status-warning bg-status-warning-bg">
                  <Sparkles className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">New Patterns</span>
                  <span className="text-lg font-extrabold text-status-warning">{data.new_patterns}</span>
                </div>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface-card border border-border-subtle text-indosat-magenta bg-status-error-bg">
                  <AlertOctagon className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Error Patterns</span>
                  <span className="text-lg font-extrabold text-indosat-magenta">{data.error_patterns}</span>
                </div>
              </div>
            </div>

            {/* Pattern Table */}
            <div className="overflow-x-auto border border-border-subtle rounded-lg">
              <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                  <tr>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Pattern Template</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Count</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">%</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Top Service</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Dominant Sev</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Flags</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
                  {data.patterns.map((pat, idx) => (
                    <tr key={idx} className="hover:bg-surface-hover/30 transition-colors">
                      <td className="px-4 py-3 max-w-[350px] font-mono text-text-primary text-[11px] truncate select-all" title={pat.template}>
                        {pat.template}
                      </td>
                      <td className="px-4 py-3 font-bold select-none">{pat.count}</td>
                      <td className="px-4 py-3 text-text-tertiary select-none">{pat.pct_of_total}%</td>
                      <td className="px-4 py-3">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-bg-main border border-border-subtle">
                          {pat.top_service}
                        </span>
                      </td>
                      <td className="px-4 py-3 select-none">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${getSeverityBadgeColor(pat.dominant_severity)}`}>
                          {pat.dominant_severity}
                        </span>
                      </td>
                      <td className="px-4 py-3 select-none">
                        {pat.is_new && (
                          <span className="px-2 py-0.5 rounded text-[10px] bg-status-warning-bg text-status-warning font-bold animate-pulse">
                            NEW
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            Click &quot;Analyse logs&quot; to cluster log messages using Drain3.
          </div>
        )}
      </div>
    </div>
  );
}
