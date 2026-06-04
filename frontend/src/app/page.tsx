"use client";

import React, { useState, useEffect } from 'react';
import Sidebar from '../components/Sidebar';
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

  return (
    <div className="shell">
      {/* Sidebar navigation */}
      <Sidebar 
        currentPage={currentPage} 
        onPageChange={setCurrentPage} 
        health={health} 
      />

      <main>
        {/* Page 1: Predictive Capacity Analytics */}
        <div className={`page ${currentPage === 'predictive' ? 'active' : ''}`} id="page-predictive">
          <div className="page-eyebrow">Page 1 — Predictive Analytics</div>
          <h1 className="page-title">Forecasting & Capacity</h1>
          <p className="page-desc">Predict resource saturation and traffic volume before they become incidents, using time-series models trained on your telemetry history.</p>
          
          <div className="stat-row">
            <div className="stat">
              <div className="stat-label">Memory used</div>
              <div className="stat-value" id="stat-mem">
                {summaryLoading ? '...' : (summary?.memory?.used_pct !== undefined ? `${summary.memory.used_pct.toFixed(1)}%` : '—')}
              </div>
              <div className="stat-sub">of total RAM</div>
            </div>
            <div className="stat">
              <div className="stat-label">Load average (1m)</div>
              <div className="stat-value" id="stat-load">
                {summaryLoading ? '...' : (summary?.load?.load1 !== undefined ? summary.load.load1.toFixed(2) : '—')}
              </div>
              <div className="stat-sub">system load</div>
            </div>
            <div className="stat">
              <div className="stat-label">Orders last hour</div>
              <div className="stat-value" id="stat-orders">
                {summaryLoading ? '...' : (summary?.orders?.last_hour !== undefined ? summary.orders.last_hour : '—')}
              </div>
              <div className="stat-sub" id="stat-orders-trend">
                {summary?.orders?.change_pct !== undefined 
                  ? `${summary.orders.change_pct >= 0 ? '+' : ''}${summary.orders.change_pct.toFixed(1)}% vs previous hour`
                  : "vs previous hour"
                }
              </div>
            </div>
          </div>

          <MetricForecast
            type="memory"
            title="Memory availability forecast"
            apiBase={apiBase}
            color="#24BCAD" // Indosat Teal
            yAxisLabel="Free Memory (GB)"
            placeholderText="Click 'Run forecast' to generate a 24-hour prediction using Facebook Prophet."
            onShowInfo={setInfoModuleId}
          />
          <MetricForecast
            type="cpu"
            title="CPU usage forecast"
            apiBase={apiBase}
            color="#FFCA09" // Indosat Yellow
            yAxisLabel="CPU Utilization (%)"
            placeholderText="Click 'Run forecast' to generate a 24-hour CPU prediction."
            onShowInfo={setInfoModuleId}
          />
          <MetricForecast
            type="traffic"
            title="Order traffic forecast"
            apiBase={apiBase}
            color="#EB008C" // Indosat Magenta
            yAxisLabel="Inbound Transactions (Orders / 5m)"
            placeholderText="Click 'Run forecast' to predict order volume for the next 12 hours."
            onShowInfo={setInfoModuleId}
          />
        </div>

        {/* Page 2: Detective Anomaly Finder */}
        <div className={`page ${currentPage === 'detective' ? 'active' : ''}`} id="page-detective">
          <div className="page-eyebrow">Page 2 — Detective Analytics</div>
          <h1 className="page-title">Smart Anomaly Detection</h1>
          <p className="page-desc">Context-aware anomaly detection across metrics, logs, and traces — moving beyond static thresholds to dynamic, multivariate intelligence.</p>

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

        {/* Page 3: Diagnostic Root-Cause Agent */}
        <div className={`page ${currentPage === 'diagnostic' ? 'active' : ''}`} id="page-diagnostic">
          <div className="page-eyebrow">Page 3 — Diagnostic Analytics</div>
          <h1 className="page-title">Root Cause Analysis</h1>
          <p className="page-desc">Automated RCA that connects the dots across metrics, traces, and logs — and hands you a human-readable incident summary.</p>

          <DependencyGraph 
            apiBase={apiBase} 
            onShowInfo={setInfoModuleId} 
          />
          <TelemetryCorrelation 
            apiBase={apiBase} 
            onShowInfo={setInfoModuleId} 
          />
          <IncidentSummarizer 
            apiBase={apiBase} 
            onShowInfo={setInfoModuleId} 
          />
        </div>

        {/* Page 4: AIOps Assistant Chatbot */}
        <div className={`page ${currentPage === 'chatbot' ? 'active' : ''}`} id="page-chatbot">
          <div className="page-eyebrow">Page 4 — AIOps Assistant</div>
          <h1 className="page-title">Talk to Your Data</h1>
          <p className="page-desc">Ask questions in plain English. The assistant translates them into ClickHouse SQL, runs the query, and returns results as tables or charts.</p>
          
          <AIOpsChat apiBase={apiBase} />
        </div>

        {/* Page 5: Operations Incident Registry */}
        <div className={`page ${currentPage === 'incidents' ? 'active' : ''}`} id="page-incidents">
          <div className="page-eyebrow">Operations</div>
          <h1 className="page-title">Incident Registry</h1>
          <p className="page-desc">Track, manage, and resolve operational incidents. Create new incidents manually or let the Diagnostic engine raise them automatically.</p>
          
          <IncidentRegistry />
        </div>
      </main>

      {/* Educational info overlay */}
      <InfoOverlay 
        moduleId={infoModuleId} 
        onClose={() => setInfoModuleId(null)} 
      />
    </div>
  );
}
