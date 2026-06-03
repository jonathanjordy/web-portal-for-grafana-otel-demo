'use client';

import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Info } from 'lucide-react';
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
          font: { family: 'Nunito', size: 11, weight: 'bold' },
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
          font: { family: 'Nunito', size: 11, weight: 'bold' },
          maxRotation: 0
        },
        grid: { color: 'rgba(0,0,0,0.03)' }
      },
      y: {
        title: {
          display: true,
          text: yAxisLabel,
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 11, weight: 'bold' }
        },
        ticks: {
          color: '#8f8f8f',
          font: { family: 'Nunito', size: 11, weight: 'bold' }
        },
        grid: { color: 'rgba(0,0,0,0.03)' }
      }
    }
  };

  return (
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <span className="font-bold text-sm text-text-primary">{title}</span>
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-text-tertiary">{statusText}</span>
          <button
            onClick={fetchForecast}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm active:translate-y-0"
          >
            {loading ? 'Running...' : 'Run forecast'}
          </button>
          <button
            onClick={() => onShowInfo(type)}
            className="w-6 h-6 rounded-full border border-border-medium bg-surface-card hover:bg-indosat-teal hover:border-indosat-teal hover:text-white flex items-center justify-center cursor-pointer transition-all duration-150"
            title="How this works"
          >
            <Info className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-5 min-h-[180px] flex flex-col justify-center">
        {errorText ? (
          <div className="text-center text-status-error font-semibold text-sm py-4">{errorText}</div>
        ) : loading ? (
          <div className="flex flex-col items-center justify-center py-10 select-none">
            <div className="w-8 h-8 border-4 border-indosat-teal border-t-transparent rounded-full animate-spin mb-3" />
            <div className="text-xs font-bold text-text-secondary">
              Fitting Prophet time-series model on ClickHouse history...
            </div>
          </div>
        ) : data && chartData ? (
          <div className="h-[220px] w-full">
            <Line data={chartData as any} options={chartOptions as any} />
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            {placeholderText}
          </div>
        )}
      </div>
    </div>
  );
}
