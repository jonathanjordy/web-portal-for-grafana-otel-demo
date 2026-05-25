import { useEffect, useRef, useState } from 'react';
import InfoButton from '../components/InfoButton.jsx';
import StatCard from '../components/StatCard.jsx';
import { Chart } from '../components/charts.js';
import { FilterSelect, useFilters } from '../hooks/useFilters.jsx';
import { apiGet } from '../services/api.js';

export default function PredictivePage({ onOpenInfo }) {
  const [summary, setSummary] = useState({});
  const [host, setHost] = useState('');
  const [service, setService] = useState('');
  const filters = useFilters();

  async function loadSummary() {
    try {
      const params = new URLSearchParams({ host, service });
      setSummary(await apiGet(`/predictive/summary?${params}`));
    } catch {
      setSummary({});
    }
  }

  useEffect(() => {
    loadSummary();
  }, [host, service]);

  useEffect(() => {
    if (host && !filters.hosts.some((option) => option.value === host)) setHost('');
  }, [filters.hosts, host]);

  const memColor = summary.memory?.used_pct > 85 ? 'red' : summary.memory?.used_pct > 70 ? 'amber' : 'green';
  const trend = summary.orders?.change_pct ?? 0;
  const trendColor = trend > 10 ? 'var(--amber)' : trend < -10 ? 'var(--red)' : 'var(--text-3)';

  return (
    <div className="page active">
      <div className="page-eyebrow">Page 1 - Predictive Analytics</div>
      <h1 className="page-title">Forecasting & Capacity</h1>
      <p className="page-desc">Predict resource saturation and traffic volume before they become incidents, using time-series models trained on your telemetry history.</p>

      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
        <FilterSelect label="Host" value={host} options={filters.hosts} onChange={setHost} allLabel="All hosts" disabledLabel="Host label unavailable" />
        <FilterSelect label="Service" value={service} options={filters.services} onChange={setService} allLabel="All services" />
      </div>

      <div className="stat-row">
        <StatCard label="Memory used" value={summary.memory ? `${summary.memory.used_pct}%` : '-'} sub="of total RAM" color={summary.memory ? memColor : ''} />
        <StatCard label="Load average (1m)" value={summary.load?.load1 ?? '-'} sub="system load" />
        <StatCard label="Orders last hour" value={summary.orders?.last_hour ?? '-'} sub={<span style={{ color: trendColor }}>{summary.orders ? `${trend >= 0 ? '+' : ''}${trend}% vs prev hour` : 'vs previous hour'}</span>} />
      </div>

      <ForecastPanel type="memory" title="Memory availability forecast" topic="memory" color="#24BCAD" label="Memory available (GB)" initialText='Click "Run forecast" to generate a 24-hour prediction using Facebook Prophet.' onOpenInfo={onOpenInfo} host={host} />
      <ForecastPanel type="cpu" title="CPU usage forecast" topic="cpu" color="#FFCA09" label="CPU usage (%)" initialText='Click "Run forecast" to generate a 24-hour CPU prediction.' onOpenInfo={onOpenInfo} host={host} />
      <ForecastPanel type="traffic" title="Order traffic forecast" topic="traffic" color="#EB008C" label="Orders per 5 min" initialText='Click "Run forecast" to predict order volume for the next 12 hours.' onOpenInfo={onOpenInfo} service={service} />
    </div>
  );
}

function ForecastPanel({ type, title, topic, color, label, initialText, onOpenInfo, host = '', service = '' }) {
  const [status, setStatus] = useState('-');
  const [loading, setLoading] = useState(initialText);
  const [data, setData] = useState(null);
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  async function loadForecast() {
    setLoading(`Fitting Prophet model on ${type} data...`);
    setData(null);
    try {
      const params = new URLSearchParams({ host, service });
      const nextData = await apiGet(`/predictive/${type}?${params}`);
      if (nextData.detail) {
        setLoading(nextData.detail);
        return;
      }
      const peak = nextData.peak_predicted ? ` - peak: ${nextData.peak_predicted.value}` : '';
      setStatus(`${nextData.data_points} data points${peak}`);
      setLoading('');
      setData(nextData);
    } catch (error) {
      setLoading(`Error: ${error.message}`);
    }
  }

  useEffect(() => {
    if (!data || !canvasRef.current) return undefined;
    if (chartRef.current) chartRef.current.destroy();

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        datasets: [
          { label: 'Confidence band', data: data.forecast.map((d) => ({ x: d.ds, y: d.yhat_upper })), borderWidth: 0, backgroundColor: `${color}22`, fill: '+1', pointRadius: 0, tension: 0.4 },
          { label: 'Lower band', data: data.forecast.map((d) => ({ x: d.ds, y: d.yhat_lower })), borderWidth: 0, backgroundColor: `${color}22`, fill: false, pointRadius: 0, tension: 0.4 },
          { label: 'Forecast', data: data.forecast.map((d) => ({ x: d.ds, y: d.yhat })), borderColor: color, borderWidth: 2.5, borderDash: [4, 4], backgroundColor: 'transparent', pointRadius: 0, tension: 0.4 },
          { label, data: data.historical.map((d) => ({ x: d.ds, y: d.y })), borderColor: '#2b2b2b', borderWidth: 2.5, backgroundColor: 'transparent', pointRadius: 0, tension: 0.4 },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: true, labels: { font: { family: 'Nunito', size: 12, weight: 600 }, color: '#8f8f8f', boxWidth: 12 } },
          tooltip: { titleFont: { family: 'Nunito' }, bodyFont: { family: 'Nunito' }, callbacks: { label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(2)}` } },
        },
        scales: {
          x: { type: 'time', time: { unit: 'hour', displayFormats: { hour: 'HH:mm' }, tooltipFormat: 'dd LLL yyyy HH:mm' }, adapters: { date: { locale: 'id' } }, ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 12, weight: 600 }, maxRotation: 0 }, grid: { color: 'rgba(0,0,0,0.04)' } },
          y: { ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 12, weight: 600 } }, grid: { color: 'rgba(0,0,0,0.04)' } },
        },
      },
    });

    return () => chartRef.current?.destroy();
  }, [color, data, label]);

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">{title}</span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="panel-meta">{status}</span>
          <button className="btn-sm" onClick={loadForecast} type="button">Run forecast</button>
          <InfoButton topic={topic} onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        <canvas ref={canvasRef} style={{ display: data ? 'block' : 'none', maxHeight: 260 }} />
      </div>
    </div>
  );
}
