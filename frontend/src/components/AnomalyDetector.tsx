'use client';

import React, { useState, useRef } from 'react';
import { Bar, getElementAtEvent } from 'react-chartjs-2';
import { Info, AlertTriangle, CheckCircle2 } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { AnomalyResponse, AnomalyTimelineItem } from '../types/otel';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

interface AnomalyDetectorProps {
  apiBase: string;
  onShowInfo: (id: string) => void;
}

export default function AnomalyDetector({ apiBase, onShowInfo }: AnomalyDetectorProps) {
  const [hours, setHours] = useState('6');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AnomalyResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyTimelineItem | null>(null);
  
  const chartRef = useRef<any>(null);

  const fetchAnomalies = async () => {
    setLoading(true);
    setErrorText(null);
    setSelectedAnomaly(null);
    try {
      const res = await fetch(`${apiBase}/detective/anomalies?hours=${hours}`);
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

  const onChartClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!chartRef.current || !data) return;
    
    const elements = getElementAtEvent(chartRef.current, event as any);
    if (!elements.length) return;
    
    const itemIndex = elements[0].index;
    setSelectedAnomaly(data.timeline[itemIndex]);
  };

  // Prepare chart datasets
  const chartData = data ? {
    labels: data.timeline.map(d => d.ts.replace('T', ' ').slice(11, 16)),
    datasets: [{
      label: 'Anomaly score',
      data: data.timeline.map(d => d.anomaly_score),
      backgroundColor: data.timeline.map(d => d.is_anomaly ? 'rgba(235, 0, 140, 0.85)' : 'rgba(36, 188, 173, 0.55)'),
      borderRadius: 4,
      barPercentage: 0.9,
      categoryPercentage: 1.0
    }]
  } : null;

  const chartOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          generateLabels: () => [
            { text: 'Normal', fillStyle: 'rgba(36, 188, 173, 0.55)', strokeStyle: 'transparent' },
            { text: 'Anomaly', fillStyle: 'rgba(235, 0, 140, 0.85)', strokeStyle: 'transparent' }
          ] as any,
          font: { family: 'Nunito', size: 11, weight: 'bold' }
        }
      },
      tooltip: {
        callbacks: {
          title: (items) => {
            if (!data) return '';
            return data.timeline[items[0].dataIndex].ts.replace('T', ' ').slice(0, 16);
          },
          label: (item) => {
            if (!data) return '';
            const d = data.timeline[item.dataIndex];
            return `Score: ${(d.anomaly_score * 100).toFixed(1)}%${d.is_anomaly ? ' ⚠ ANOMALY' : ''}`;
          },
          afterLabel: (item) => {
            if (!data) return '';
            const d = data.timeline[item.dataIndex];
            return d.contributing.length ? 'Contributing: ' + d.contributing.join(', ') : '';
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 10 }, maxTicksLimit: 12 },
        grid: { display: false }
      },
      y: {
        min: 0,
        max: data ? Math.min(1.0, Math.ceil(Math.max(...data.timeline.map(t => t.anomaly_score), 0.1) * 10) / 10) : 1.0,
        title: {
          display: true,
          text: 'Anomaly score',
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 11, weight: 'bold' }
        },
        ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 11, weight: 'bold' } },
        grid: { color: 'rgba(0,0,0,0.03)' }
      }
    }
  };

  const topAnomalies = data ? data.timeline.filter(t => t.is_anomaly).slice(0, 10) : [];

  return (
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex flex-wrap items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <div>
          <span className="font-bold text-sm text-text-primary block">Multivariate Anomaly Detection</span>
          <span className="text-[11px] font-semibold text-text-tertiary">
            Isolation Forest · payment duration, failures, cache misses, order errors, node load
          </span>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="text-xs px-2.5 py-1.5 border border-border-medium rounded-md font-semibold bg-bg-main text-text-secondary outline-none focus:border-indosat-teal transition-all"
          >
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
            <option value="12">Last 12h</option>
            <option value="24">Last 24h</option>
          </select>
          <button
            onClick={fetchAnomalies}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm"
          >
            {loading ? 'Analyzing...' : 'Run detection'}
          </button>
          <button
            onClick={() => onShowInfo('anomalies')}
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
              Fitting Isolation Forest & calculating dynamic anomaly contamination...
            </div>
          </div>
        ) : data && chartData ? (
          <div>
            {/* Stats Summary Row */}
            <div className="grid grid-template-cols grid-cols-3 gap-4 mb-4 select-none animate-fade">
              <div className="glass-panel p-3.5 bg-bg-main/50">
                <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Windows Analysed</span>
                <span className="text-xl font-extrabold text-text-primary">{data.total_windows}</span>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50">
                <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Anomalies Found</span>
                <span className={`text-xl font-extrabold ${data.anomaly_count > 0 ? 'text-indosat-magenta' : 'text-text-primary'}`}>
                  {data.anomaly_count}
                </span>
              </div>
              <div className="glass-panel p-3.5 bg-bg-main/50">
                <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Anomaly Rate</span>
                <span className={`text-xl font-extrabold ${data.anomaly_rate > 5 ? 'text-indosat-magenta' : data.anomaly_rate > 0 ? 'text-status-warning' : 'text-indosat-teal'}`}>
                  {data.anomaly_rate}%
                </span>
              </div>
            </div>

            {/* Bar Chart */}
            <div className="h-[200px] w-full mb-5">
              <Bar ref={chartRef} data={chartData as any} options={chartOptions as any} onClick={onChartClick} />
            </div>

            {/* Split Grid: Table & Inspection Detail */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
              {/* Anomalies Table */}
              <div className="animate-fade">
                <div className="text-[10px] font-bold text-text-tertiary uppercase tracking-widest mb-2">
                  Top Anomalous Windows (Select a window to inspect)
                </div>
                {topAnomalies.length > 0 ? (
                  <div className="overflow-x-auto border border-border-subtle rounded-lg">
                    <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                      <thead className="bg-surface-hover/30">
                        <tr>
                          <th className="px-3 py-2 text-left text-text-tertiary">Timestamp</th>
                          <th className="px-3 py-2 text-left text-text-tertiary">Score</th>
                          <th className="px-3 py-2 text-left text-text-tertiary">Primary Factors</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border-subtle bg-surface-card">
                        {topAnomalies.map((item, idx) => (
                          <tr
                            key={idx}
                            onClick={() => setSelectedAnomaly(item)}
                            className={`cursor-pointer transition-colors ${
                              selectedAnomaly?.ts === item.ts
                                ? 'bg-indosat-magenta/5 hover:bg-indosat-magenta/10'
                                : 'hover:bg-surface-hover/50'
                            }`}
                          >
                            <td className="px-3 py-2.5 font-mono text-text-secondary">
                              {item.ts.replace('T', ' ').slice(0, 16)}
                            </td>
                            <td className="px-3 py-2.5">
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-status-error-bg text-indosat-magenta border border-indosat-magenta/10">
                                {(item.anomaly_score * 100).toFixed(0)}%
                              </span>
                            </td>
                            <td className="px-3 py-2.5 flex flex-wrap gap-1">
                              {item.contributing.map((c, i) => (
                                <span key={i} className="px-1.5 py-0.5 rounded text-[10px] bg-status-warning-bg text-status-warning font-bold">
                                  {c.replace(/_/g, ' ')}
                                </span>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="glass-panel p-5 text-center text-xs font-bold text-text-tertiary bg-bg-main/30 border-2 border-dashed border-border-medium flex items-center justify-center gap-2">
                    <CheckCircle2 className="w-4.5 h-4.5 text-indosat-teal" />
                    No multivariate anomalies detected in this range!
                  </div>
                )}
              </div>

              {/* Inspection Details */}
              {selectedAnomaly && (
                <div className="glass-panel p-5 bg-surface-hover/20 border border-border-medium shadow-sm animate-fade">
                  <div className="flex items-center justify-between border-b border-border-subtle pb-3 mb-4">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4.5 h-4.5 text-indosat-magenta" />
                      <span className="font-bold text-sm text-text-primary">
                        Anomaly Details
                      </span>
                    </div>
                    <span className="font-mono text-xs font-bold text-text-tertiary">
                      {selectedAnomaly.ts.replace('T', ' ').slice(0, 16)}
                    </span>
                  </div>

                  {/* Grid of Metric Values */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 mb-4">
                    {Object.entries(selectedAnomaly.metrics).map(([key, val]) => {
                      const isContr = selectedAnomaly.contributing.includes(key);
                      return (
                        <div key={key} className={`glass-panel p-3 ${isContr ? 'bg-status-error-bg/30 border-indosat-magenta/20' : 'bg-surface-card'}`}>
                          <span className="text-[9px] font-bold text-text-tertiary uppercase tracking-wider block mb-0.5 truncate" title={key.replace(/_/g, ' ')}>
                            {key.replace(/_/g, ' ')}
                          </span>
                          <span className={`text-base font-extrabold ${isContr ? 'text-indosat-magenta' : 'text-text-primary'}`}>
                            {val != null ? Number(val).toFixed(3) : '—'}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="text-xs font-semibold text-text-secondary leading-relaxed bg-surface-card p-3 rounded-lg border border-border-subtle">
                    <div className="font-bold text-text-tertiary uppercase tracking-widest text-[9px] mb-1">
                      Z-Score Attribution Analytics
                    </div>
                    The Isolation Forest flagged this window as anomalous because the combination of telemetry metrics was outside the historical 95% baseline covariance. 
                    {selectedAnomaly.contributing.length > 0 && (
                      <span className="block mt-1.5">
                        Primary anomalies traced back to: <strong className="text-indosat-magenta">{selectedAnomaly.contributing.map(c => c.replace(/_/g, ' ')).join(', ')}</strong>.
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            Click &quot;Run detection&quot; to analyse ClickHouse telemetry metrics with Isolation Forest.
          </div>
        )}
      </div>
    </div>
  );
}
