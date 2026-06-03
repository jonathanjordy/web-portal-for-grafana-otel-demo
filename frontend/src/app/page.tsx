"use client";

import React, { useState, useEffect } from 'react';
import Sidebar from '../components/Sidebar';
import StatCard from '../components/StatCard';
import MetricForecast from '../components/MetricForecast';
import AnomalyDetector from '../components/AnomalyDetector';
import LogPatterns from '../components/LogPatterns';
import TraceShapes from '../components/TraceShapes';
import DependencyGraph from '../components/DependencyGraph';
import TelemetryCorrelation from '../components/TelemetryCorrelation';
import IncidentSummarizer from '../components/IncidentSummarizer';
import AIOpsChat from '../components/AIOpsChat';
import IncidentRegistry from '../components/IncidentRegistry';
import InfoOverlay from '../components/InfoOverlay';
import { useHealthCheck } from '../hooks/useHealthCheck';
import { PredictiveSummary } from '../types/otel';
import { Info, HelpCircle, RefreshCw } from 'lucide-react';

const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://35.219.90.43:8080/api';

export default function Home() {
  const [currentPage, setCurrentPage] = useState<string>('predictive');
  const [infoModuleId, setInfoModuleId] = useState<string | null>(null);
  const [summary, setSummary] = useState<PredictiveSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState<boolean>(false);
  
  const health = useHealthCheck(apiBase);

  const fetchSummary = async () => {
    setSummaryLoading(true);
    try {
      const res = await fetch(`${apiBase}/predictive/summary`);
      const data = await res.json();
      setSummary(data);
    } catch (err) {
      console.error("Error loading predictive summary:", err);
    } finally {
      setSummaryLoading(false);
    }
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  useEffect(() => {
    if (currentPage === 'predictive') {
      fetchSummary();
    }
  }, [currentPage]);

  const renderActiveTab = () => {
    switch (currentPage) {
      case 'predictive':
        return (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <StatCard
                label="System Load (1m avg)"
                value={summary?.load?.load1 !== undefined ? summary.load.load1.toFixed(2) : '—'}
                subText={summaryLoading ? "Refreshing..." : "Calculated from node_load1"}
                color={summary?.load?.load1 && summary.load.load1 > 4 ? 'amber' : 'default'}
              />
              <StatCard
                label="Available Memory"
                value={summary?.memory?.used_pct !== undefined ? `${(100 - summary.memory.used_pct).toFixed(1)}%` : '—'}
                subText={summaryLoading ? "Refreshing..." : `${summary?.memory?.used_pct !== undefined ? summary.memory.used_pct.toFixed(1) : '—'}% utilized`}
                color={summary?.memory?.used_pct && summary.memory.used_pct > 85 ? 'red' : 'teal'}
              />
              <StatCard
                label="Inbound Order Volume"
                value={summary?.orders?.last_hour !== undefined ? `${summary.orders.last_hour} /hr` : '—'}
                subText={summary?.orders?.change_pct !== undefined 
                  ? `${summary.orders.change_pct >= 0 ? '+' : ''}${summary.orders.change_pct.toFixed(1)}% compared to prev hour`
                  : "Compared to last hour"
                }
                color={summary?.orders?.change_pct !== undefined && summary.orders.change_pct < -5 ? 'magenta' : 'green'}
              />
            </div>

            {/* Prophet Analytics Rows */}
            <div className="space-y-6">
              <MetricForecast
                type="memory"
                title="Memory Availability Forecast"
                apiBase={apiBase}
                color="#24BCAD" // Indosat Teal
                yAxisLabel="Free Memory (GB)"
                placeholderText="Click 'Run forecast' to generate 24-hour memory depletion prediction"
                onShowInfo={setInfoModuleId}
              />
              <MetricForecast
                type="cpu"
                title="CPU Utilization Forecast"
                apiBase={apiBase}
                color="#FFCA09" // Indosat Yellow
                yAxisLabel="CPU Utilization (%)"
                placeholderText="Click 'Run forecast' to calculate CPU saturation baseline & forecast"
                onShowInfo={setInfoModuleId}
              />
              <MetricForecast
                type="traffic"
                title="Order Traffic Forecast"
                apiBase={apiBase}
                color="#EB008C" // Indosat Magenta
                yAxisLabel="Inbound Transactions (Orders / 5m)"
                placeholderText="Click 'Run forecast' to predict transactional volume limits"
                onShowInfo={setInfoModuleId}
              />
            </div>
          </div>
        );

      case 'detective':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <AnomalyDetector 
                apiBase={apiBase} 
                onShowInfo={setInfoModuleId} 
              />
              <LogPatterns 
                apiBase={apiBase} 
                onShowInfo={setInfoModuleId} 
              />
              <TraceShapes 
                apiBase={apiBase} 
                onShowInfo={setInfoModuleId} 
              />
            </div>
          </div>
        );

      case 'diagnostic':
        return (
          <div className="space-y-6">
            {/* Dependency Graph Component */}
            <DependencyGraph 
              apiBase={apiBase} 
              onShowInfo={setInfoModuleId} 
            />

            {/* Cross-Telemetry and Summarization */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <TelemetryCorrelation 
                apiBase={apiBase} 
                onShowInfo={setInfoModuleId} 
              />
              <IncidentSummarizer 
                apiBase={apiBase} 
                onShowInfo={setInfoModuleId} 
              />
            </div>
          </div>
        );

      case 'chatbot':
        return (
          <div className="space-y-6">
            <div className="glass-panel p-6 bg-surface-card select-none">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-extrabold text-text-primary">Natural Language ClickHouse Terminal</h2>
                  <span className="bg-status-ok-bg text-indosat-teal px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider">
                    Powered by Gemini 2.5 Flash
                  </span>
                </div>
                <button
                  onClick={() => setInfoModuleId('chatbot')}
                  className="w-6 h-6 rounded-full border border-border-medium bg-surface-card hover:bg-indosat-teal hover:border-indosat-teal hover:text-white flex items-center justify-center cursor-pointer transition-all"
                  title="How this works"
                >
                  <HelpCircle className="w-3.5 h-3.5" />
                </button>
              </div>
              <p className="text-xs font-semibold text-text-secondary mb-6 leading-relaxed">
                Translate natural language statements into high-efficiency ClickHouse SQL commands. Query spans, system resources, and metrics directly without writing raw database queries.
              </p>
              
              <AIOpsChat apiBase={apiBase} />
            </div>
          </div>
        );

      case 'incidents':
        return (
          <IncidentRegistry apiBase={apiBase} />
        );

      default:
        return (
          <div className="flex items-center justify-center py-20 text-text-tertiary">
            Tab not implemented.
          </div>
        );
    }
  };

  const getPageHeaderTitle = () => {
    switch (currentPage) {
      case 'predictive': return 'Predictive Capacity Analytics';
      case 'detective': return 'Detective Anomaly Finder';
      case 'diagnostic': return 'Diagnostic Root-Cause Agent';
      case 'chatbot': return 'AIOps Conversational Terminal';
      case 'incidents': return 'System Incident Registry';
      default: return 'AIOps Dashboard';
    }
  };

  return (
    <div className="flex h-screen bg-bg-main font-sans antialiased overflow-hidden">
      {/* Dynamic Navigation Sidebar */}
      <Sidebar 
        currentPage={currentPage} 
        onPageChange={setCurrentPage} 
        health={health} 
      />

      {/* Main Panel Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Navbar */}
        <header className="h-[74px] border-b border-border-subtle bg-surface-card px-8 flex items-center justify-between flex-shrink-0 select-none">
          <div className="flex items-center gap-3">
            <h1 className="font-sans text-lg font-extrabold tracking-tight text-text-primary">
              {getPageHeaderTitle()}
            </h1>
          </div>

          <div className="flex items-center gap-4">
            {currentPage === 'predictive' && (
              <button
                onClick={fetchSummary}
                disabled={summaryLoading}
                className="p-2 rounded-lg bg-surface-hover hover:bg-border-subtle text-text-secondary hover:text-text-primary transition-all disabled:opacity-50 flex items-center gap-1.5 cursor-pointer text-xs font-bold"
                title="Refresh stats"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${summaryLoading ? 'animate-spin' : ''}`} />
                <span>Refresh stats</span>
              </button>
            )}
            <div className="text-xs font-bold text-text-tertiary">
              Indosat Ooredoo Hutchison
            </div>
          </div>
        </header>

        {/* Tab Canvas (Scrollable) */}
        <main className="flex-1 overflow-y-auto p-8 max-w-[1600px] w-full mx-auto">
          {renderActiveTab()}
        </main>
      </div>

      {/* Shared Info Educational Overlay Modal */}
      <InfoOverlay 
        moduleId={infoModuleId} 
        onClose={() => setInfoModuleId(null)} 
      />
    </div>
  );
}
