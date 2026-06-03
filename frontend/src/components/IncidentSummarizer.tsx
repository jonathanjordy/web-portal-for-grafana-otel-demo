'use client';

import React, { useState } from 'react';
import { Info, Copy, Cpu, Sparkles, AlertCircle } from 'lucide-react';
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
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex flex-wrap items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indosat-magenta animate-pulse" />
          <div>
            <span className="font-bold text-sm text-text-primary block">AI Incident Summarization</span>
            <span className="text-[11px] font-semibold text-text-tertiary">
              Powered by Gemini 2.5 Flash — aggregates error anomalies, slow logs, and executes root cause reasoning
            </span>
          </div>
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
            onClick={generateSummary}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm"
          >
            {loading ? 'Synthesizing...' : 'Generate summary'}
          </button>
          <button
            onClick={() => onShowInfo('llm')}
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
          <div className="text-center text-status-error font-semibold text-sm py-4 flex items-center justify-center gap-1.5">
            <AlertCircle className="w-4 h-4" />
            {errorText}
          </div>
        ) : loading ? (
          <div className="flex flex-col items-center justify-center py-12 select-none">
            <div className="w-8 h-8 border-4 border-indosat-magenta border-t-transparent rounded-full animate-spin mb-3" />
            <div className="text-xs font-bold text-text-secondary">
              Gemini 2.5 Flash is analyzing logs, error rates, and metrics to compile RCA...
            </div>
          </div>
        ) : data ? (
          <div className="animate-fade space-y-4">
            {/* Summary Text Panel */}
            <div className="bg-bg-main border border-border-subtle rounded-xl p-5 text-sm leading-relaxed font-medium text-text-secondary whitespace-pre-wrap font-sans select-text select-all">
              {data.summary}
            </div>

            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className={`px-3.5 py-2 rounded-md text-xs font-bold transition-all flex items-center gap-2 cursor-pointer ${
                copied ? 'bg-status-ok-bg text-indosat-teal border border-indosat-teal/20' : 'bg-surface-card border border-border-medium text-text-secondary hover:bg-surface-hover hover:text-text-primary'
              }`}
            >
              <Copy className="w-3.5 h-3.5" />
              {copied ? 'Copied to clipboard!' : 'Copy to clipboard'}
            </button>
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            Click &quot;Generate summary&quot; to have Gemini 2.5 Flash analyse your ClickHouse telemetry and produce an incident post-mortem brief.
          </div>
        )}
      </div>
    </div>
  );
}
