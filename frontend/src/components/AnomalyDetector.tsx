'use client';

import React, { useState, useRef } from 'react';
import { Bar, getElementAtEvent } from 'react-chartjs-2';
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
      backgroundColor: data.timeline.map(d => d.is_anomaly ? 'rgba(235,0,140,0.85)' : 'rgba(36,188,173,0.55)'),
      borderColor: data.timeline.map(d => d.is_anomaly ? 'rgba(235,0,140,0.85)' : 'rgba(36,188,173,0.55)'),
      borderWidth: 0,
      borderRadius: 2,
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
            { text: 'Normal', fillStyle: 'rgba(36,188,173,0.55)', strokeStyle: 'transparent', fontColor: '#8f8f8f', font: { family: 'Nunito', size: 12 } },
            { text: 'Anomaly', fillStyle: 'rgba(235,0,140,0.85)', strokeStyle: 'transparent', fontColor: '#8f8f8f', font: { family: 'Nunito', size: 12 } }
          ] as any
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
            return d.contributing.length ? 'Factors: ' + d.contributing.join(', ') : '';
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 10 }, maxTicksLimit: 12 },
        grid: { color: 'rgba(0,0,0,0.04)' }
      },
      y: {
        min: 0,
        max: data ? Math.max(...data.timeline.map(t => t.anomaly_score), 0.1) : 0.1,
        title: {
          display: true,
          text: 'Anomaly score',
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 11 }
        },
        ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 11 } },
        grid: { color: 'rgba(0,0,0,0.04)' }
      }
    }
  };

  const topAnomalies = data ? data.timeline.filter(t => t.is_anomaly).slice(0, 10) : [];

  return (
    <div className="panel">
      {/* Header */}
      <div className="panel-head">
        <div>
          <div className="panel-title">Multivariate anomaly detection</div>
          <div className="panel-meta">
            Isolation Forest — payment duration · failure rate · cache misses · order errors · node load
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="select-sm"
          >
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
            <option value="12">Last 12h</option>
            <option value="24">Last 24h</option>
          </select>
          <button
            onClick={fetchAnomalies}
            disabled={loading}
            className="btn-sm"
          >
            {loading ? 'Running...' : 'Run detection'}
          </button>
          <button
            onClick={() => onShowInfo('anomalies')}
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
            Running Isolation Forest on your metrics...
          </div>
        ) : data ? (
          <div>
            {/* Stats Summary Row */}
            <div className="stat-row" style={{ marginBottom: '1rem' }}>
              <div className="stat">
                <div className="stat-label">Windows analysed</div>
                <div className="stat-value">{data.total_windows}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Anomalies found</div>
                <div className="stat-value red">{data.anomaly_count}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Anomaly rate</div>
                <div className="stat-value amber">{data.anomaly_rate}%</div>
              </div>
            </div>

            {/* Bar Chart */}
            <div style={{ height: '220px', maxHeight: '220px', width: '100%', marginBottom: '1.25rem' }}>
              <Bar ref={chartRef} data={chartData as any} options={chartOptions as any} onClick={onChartClick} />
            </div>

            {/* Anomalies Table Wrap */}
            <div style={{ marginTop: '1.25rem', overflowX: 'auto' }}>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-3)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Top anomalous windows — click a bar to see details
              </div>
              {topAnomalies.length > 0 ? (
                <table className="det-table">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Score</th>
                      <th>Contributing factors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topAnomalies.map((item, idx) => (
                      <tr
                        key={idx}
                        style={{ cursor: 'pointer' }}
                        onClick={() => setSelectedAnomaly(item)}
                      >
                        <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
                          {item.ts.replace('T', ' ').slice(0, 16)}
                        </td>
                        <td>
                          <span className="tag red">{(item.anomaly_score * 100).toFixed(0)}</span>
                        </td>
                        <td>
                          {item.contributing.map((c, i) => (
                            <span key={i} className="tag amber">
                              {c}
                            </span>
                          ))}
                          {item.contributing.length === 0 && <span className="tag">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ color: 'var(--text-3)', fontSize: '0.85rem' }}>No anomalies detected.</div>
              )}
            </div>

            {/* Inspection Details */}
            {selectedAnomaly && (
              <div id="anomaly-detail" style={{ marginTop: '1rem', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: '10px', padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text)' }}>
                    Anomaly detail — <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{selectedAnomaly.ts.replace('T', ' ').slice(0, 16)}</span>
                  </div>
                  <button
                    onClick={() => setSelectedAnomaly(null)}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-3)', fontSize: '1rem' }}
                  >
                    ✕
                  </button>
                </div>

                {/* Grid of Metric Values */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem', marginBottom: '1rem' }}>
                  {Object.entries(selectedAnomaly.metrics).map(([key, val]) => {
                    const isContr = selectedAnomaly.contributing.includes(key);
                    return (
                      <div key={key} className="stat" style={{ padding: '0.85rem' }}>
                        <div className="stat-label">{key.replace(/_/g, ' ')}</div>
                        <div className={`stat-value ${isContr ? 'red' : ''}`} style={{ fontSize: '1.3rem' }}>
                          {val != null ? Number(val).toFixed(3) : '—'}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div style={{ fontSize: '0.82rem', color: 'var(--text-2)' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-3)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>
                    Contributing factors
                  </div>
                  {selectedAnomaly.contributing.map((c, i) => (
                    <span key={i} className="tag amber">
                      {c}
                    </span>
                  ))}
                  {selectedAnomaly.contributing.length === 0 && (
                    <span style={{ color: 'var(--text-3)' }}>No contributing factors identified.</span>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Click &quot;Run detection&quot; to analyse your metrics with Isolation Forest.
          </div>
        )}
      </div>
    </div>
  );
}
