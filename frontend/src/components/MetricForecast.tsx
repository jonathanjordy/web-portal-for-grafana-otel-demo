'use client';

import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
  ChartOptions
} from 'chart.js';
import 'chartjs-adapter-luxon';
import { ForecastResponse } from '../types/otel';

// Register Chart.js modules
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

interface MetricForecastProps {
  type: 'memory' | 'cpu' | 'traffic';
  title: string;
  apiBase: string;
  color: string;
  yAxisLabel: string;
  placeholderText: string;
  onShowInfo: (id: string) => void;
}

export default function MetricForecast({
  type,
  title,
  apiBase,
  color,
  yAxisLabel,
  placeholderText,
  onShowInfo
}: MetricForecastProps) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [statusText, setStatusText] = useState<string>('—');
  const [errorText, setErrorText] = useState<string | null>(null);

  const fetchForecast = async () => {
    setLoading(true);
    setErrorText(null);
    try {
      const res = await fetch(`${apiBase}/predictive/${type}`);
      const result = await res.json();
      
      if (result.detail) {
        setErrorText(result.detail);
        return;
      }
      
      setData(result);
      const peak = result.peak_predicted ? ` · peak: ${Number(result.peak_predicted.value).toFixed(2)}` : '';
      setStatusText(`${result.data_points} data points${peak}`);
    } catch (err: any) {
      setErrorText('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const chartData = data ? {
    datasets: [
      {
        label: 'Upper confidence',
        data: data.forecast.map(d => ({ x: d.ds, y: d.yhat_upper })),
        borderWidth: 0,
        backgroundColor: `${color}11`,
        fill: '+1', // Fill to next dataset
        pointRadius: 0,
        tension: 0.4
      },
      {
        label: 'Lower confidence',
        data: data.forecast.map(d => ({ x: d.ds, y: d.yhat_lower })),
        borderWidth: 0,
        backgroundColor: `${color}11`,
        fill: false,
        pointRadius: 0,
        tension: 0.4
      },
      {
        label: 'Forecast',
        data: data.forecast.map(d => ({ x: d.ds, y: d.yhat })),
        borderColor: color,
        borderWidth: 2,
        borderDash: [5, 5],
        backgroundColor: 'transparent',
        pointRadius: 0,
        tension: 0.4
      },
      {
        label: yAxisLabel,
        data: data.historical.map(d => ({ x: d.ds, y: d.y })),
        borderColor: '#2b2b2b',
        borderWidth: 2,
        backgroundColor: 'transparent',
        pointRadius: 0,
        tension: 0.4
      }
    ]
  } : null;

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          font: { family: 'Nunito', size: 11 },
          color: '#8f8f8f',
          boxWidth: 10
        }
      },
      tooltip: {
        titleFont: { family: 'Nunito', size: 12 },
        bodyFont: { family: 'Nunito', size: 12 },
        callbacks: {
          label: (c) => {
            if (c.dataset.label === 'Upper confidence' || c.dataset.label === 'Lower confidence') return '';
            return `${c.dataset.label}: ${Number(c.parsed.y).toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'hour',
          displayFormats: { hour: 'HH:mm' },
          tooltipFormat: 'dd LLL yyyy HH:mm'
        },
        ticks: {
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 10 },
          maxRotation: 0
        },
        grid: { color: 'rgba(0,0,0,0.04)' }
      },
      y: {
        title: {
          display: true,
          text: yAxisLabel,
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 11 }
        },
        ticks: {
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 11 }
        },
        grid: { color: 'rgba(0,0,0,0.04)' }
      }
    }
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">{title}</span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="panel-meta">{statusText}</span>
          <button 
            className="btn-sm" 
            onClick={fetchForecast} 
            disabled={loading}
          >
            {loading ? 'Running...' : 'Run forecast'}
          </button>
          <button 
            className="btn-info" 
            onClick={() => onShowInfo(type)} 
            title="How this works"
          >
            i
          </button>
        </div>
      </div>
      <div className="panel-body">
        {errorText ? (
          <div className="empty" style={{ padding: '1.5rem', border: 'none', color: 'var(--red)' }}>
            {errorText}
          </div>
        ) : loading ? (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <div 
                style={{ 
                  border: '4px solid rgba(36, 188, 173, 0.2)', 
                  borderTop: '4px solid var(--indosat-teal)', 
                  borderRadius: '50%', 
                  width: '32px', 
                  height: '32px',
                  animation: 'spin 1s linear infinite',
                  marginBottom: '12px'
                }} 
              />
              <style jsx global>{`
                @keyframes spin {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
              `}</style>
              <div>Fitting Prophet time-series model on ClickHouse history...</div>
            </div>
          </div>
        ) : data && chartData ? (
          <div style={{ height: '260px', maxHeight: '260px', width: '100%' }}>
            <Line data={chartData as any} options={chartOptions as any} />
          </div>
        ) : (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            {placeholderText}
          </div>
        )}
      </div>
    </div>
  );
}
