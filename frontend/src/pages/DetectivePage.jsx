import { useEffect, useRef, useState } from 'react';
import InfoButton from '../components/InfoButton.jsx';
import StatCard from '../components/StatCard.jsx';
import { Chart } from '../components/charts.js';
import { FilterSelect, useFilters } from '../hooks/useFilters.jsx';
import { apiGet } from '../services/api.js';
import { copyText } from '../utils/copyText.js';
import { traceExploreUrl } from '../utils/traceUrl.js';

export default function DetectivePage({ onOpenInfo }) {
  const [host, setHost] = useState('');
  const [service, setService] = useState('');
  const filters = useFilters();

  useEffect(() => {
    if (host && !filters.hosts.some((option) => option.value === host)) setHost('');
  }, [filters.hosts, host]);

  return (
    <div className="page active">
      <div className="page-eyebrow">Page 2 - Detective Analytics</div>
      <h1 className="page-title">Smart Anomaly Detection</h1>
      <p className="page-desc">Context-aware anomaly detection across metrics, logs, and traces - moving beyond static thresholds to dynamic, multivariate intelligence.</p>
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
        <FilterSelect label="Host" value={host} options={filters.hosts} onChange={setHost} allLabel="All hosts" disabledLabel="Host label unavailable" />
        <FilterSelect label="Service" value={service} options={filters.services} onChange={setService} allLabel="All services" />
      </div>
      <AnomaliesPanel onOpenInfo={onOpenInfo} host={host} service={service} />
      <LogPatternsPanel onOpenInfo={onOpenInfo} service={service} />
      <TraceShapesPanel onOpenInfo={onOpenInfo} service={service} />
    </div>
  );
}

function AnomaliesPanel({ onOpenInfo, host, service }) {
  const [hours, setHours] = useState('6');
  const [loading, setLoading] = useState('Click "Run detection" to analyse your metrics with Isolation Forest.');
  const [data, setData] = useState(null);
  const [detail, setDetail] = useState(null);
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  async function loadAnomalies() {
    setLoading('Running Isolation Forest on your metrics...');
    setData(null);
    setDetail(null);
    try {
      const params = new URLSearchParams({ hours, host, service });
      const nextData = await apiGet(`/detective/anomalies?${params}`);
      if (nextData.detail) {
        setLoading(nextData.detail);
        return;
      }
      setLoading('');
      setData(nextData);
    } catch (error) {
      setLoading(`Error: ${error.message}`);
    }
  }

  useEffect(() => {
    if (!data || !canvasRef.current) return undefined;
    if (chartRef.current) chartRef.current.destroy();

    const allTs = data.timeline.map((d) => d.ts);
    const allScores = data.timeline.map((d) => d.anomaly_score);
    const bgColors = data.timeline.map((d) => (d.is_anomaly ? 'rgba(235,0,140,0.85)' : 'rgba(36,188,173,0.55)'));
    const maxScore = Math.max(...allScores, 0.1);

    chartRef.current = new Chart(canvasRef.current, {
      type: 'bar',
      data: {
        labels: allTs.map((t) => t.replace('T', ' ').slice(11, 16)),
        datasets: [{ label: 'Anomaly score', data: allScores, backgroundColor: bgColors, borderColor: bgColors, borderWidth: 0, borderRadius: 2, barPercentage: 0.9, categoryPercentage: 1.0 }],
      },
      options: {
        responsive: true,
        onClick: (event, elements) => {
          if (elements.length) setDetail(data.timeline[elements[0].index]);
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              generateLabels: () => [
                { text: 'Normal', fillStyle: 'rgba(36,188,173,0.55)', strokeStyle: 'transparent', fontColor: '#8f8f8f', font: { family: 'Nunito', size: 12 } },
                { text: 'Anomaly', fillStyle: 'rgba(235,0,140,0.85)', strokeStyle: 'transparent', fontColor: '#8f8f8f', font: { family: 'Nunito', size: 12 } },
              ],
            },
          },
          tooltip: {
            callbacks: {
              title: (items) => allTs[items[0].dataIndex].replace('T', ' ').slice(0, 16),
              label: (item) => {
                const row = data.timeline[item.dataIndex];
                return `Score: ${(row.anomaly_score * 100).toFixed(1)}${row.is_anomaly ? ' ANOMALY' : ''}`;
              },
              afterLabel: (item) => {
                const row = data.timeline[item.dataIndex];
                return row.contributing.length ? `Factors: ${row.contributing.join(', ')}` : '';
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 10 }, maxTicksLimit: 12 }, grid: { color: 'rgba(0,0,0,0.04)' } },
          y: { min: 0, max: Math.ceil(maxScore * 10) / 10, title: { display: true, text: 'Anomaly score', color: '#8f8f8f', font: { family: 'Nunito', size: 11 } }, ticks: { color: '#8f8f8f', font: { family: 'Nunito', size: 11 } }, grid: { color: 'rgba(0,0,0,0.04)' } },
        },
      },
    });

    return () => chartRef.current?.destroy();
  }, [data]);

  const anomalyRows = data?.timeline.filter((row) => row.is_anomaly).slice(0, 10) || [];

  return (
    <div className="panel">
      <div className="panel-head">
        <div><div className="panel-title">Multivariate anomaly detection</div><div className="panel-meta">Isolation Forest - order duration - failure rate - cache misses - order errors - node load</div></div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select className="select-sm" value={hours} onChange={(event) => setHours(event.target.value)}>
            <option value="3">Last 3h</option><option value="6">Last 6h</option><option value="12">Last 12h</option><option value="24">Last 24h</option>
          </select>
          <button className="btn-sm" onClick={loadAnomalies} type="button">Run detection</button>
          <InfoButton topic="anomalies" onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {data ? (
          <div className="stat-row" style={{ marginBottom: '1rem' }}>
            <StatCard label="Windows analysed" value={data.total_windows} />
            <StatCard label="Anomalies found" value={data.anomaly_count} color="red" />
            <StatCard label="Anomaly rate" value={`${data.anomaly_rate}%`} color="amber" />
          </div>
        ) : null}
        {data?.features_used ? (
          <div style={{ color: 'var(--text-3)', fontSize: '0.78rem', marginBottom: '0.85rem' }}>
            Features used: {data.features_used.join(', ')}
          </div>
        ) : null}
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        <canvas ref={canvasRef} style={{ display: data ? 'block' : 'none', maxHeight: 220 }} />
        {data ? (
          <div style={{ marginTop: '1.25rem', overflowX: 'auto' }}>
            {anomalyRows.length ? (
              <>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-3)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Top anomalous windows - click a bar to see details</div>
                <table className="det-table">
                  <thead><tr><th>Timestamp</th><th>Score</th><th>Contributing factors</th></tr></thead>
                  <tbody>{anomalyRows.map((row) => (
                    <tr key={row.ts} style={{ cursor: 'pointer' }} onClick={() => setDetail(row)}>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>{row.ts.replace('T', ' ').slice(0, 16)}</td>
                      <td><span className="tag red">{(row.anomaly_score * 100).toFixed(0)}</span></td>
                      <td>{row.contributing.length ? row.contributing.map((item) => <span className="tag amber" key={item}>{item}</span>) : <span className="tag">-</span>}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </>
            ) : <div style={{ color: 'var(--text-3)', fontSize: '0.85rem' }}>No anomalies detected.</div>}
          </div>
        ) : null}
        {detail ? <AnomalyDetail detail={detail} onClose={() => setDetail(null)} /> : null}
      </div>
    </div>
  );
}

function AnomalyDetail({ detail, onClose }) {
  return (
    <div style={{ marginTop: '1rem', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 10, padding: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text)' }}>Anomaly detail - <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{detail.ts.replace('T', ' ').slice(0, 16)}</span></div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-3)', fontSize: '1rem' }} type="button">x</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: '0.75rem', marginBottom: '1rem' }}>
        {Object.entries(detail.metrics || {}).map(([key, value]) => (
          <div className="stat" style={{ padding: '0.85rem' }} key={key}>
            <div className="stat-label">{key.replace(/_/g, ' ')}</div>
            <div className="stat-value" style={{ fontSize: '1.3rem', color: detail.contributing?.includes(key) ? 'var(--red)' : undefined }}>{value != null ? Number(value).toFixed(3) : '-'}</div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: '0.82rem', color: 'var(--text-2)' }}>
        {detail.contributing?.length ? (
          <>
            <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-3)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Contributing metrics</div>
            {detail.contributing.map((item) => <span className="tag amber" key={item}>{item}</span>)}
          </>
        ) : <span style={{ color: 'var(--text-3)' }}>No contributing factors identified.</span>}
      </div>
    </div>
  );
}

function LogPatternsPanel({ onOpenInfo, service }) {
  const [hours, setHours] = useState('2');
  const [loading, setLoading] = useState('Click "Analyse logs" to cluster log templates using Drain3.');
  const [data, setData] = useState(null);

  async function loadLogPatterns() {
    setLoading('Clustering log templates with Drain3...');
    setData(null);
    try {
      const params = new URLSearchParams({ hours, service });
      const nextData = await apiGet(`/detective/log-patterns?${params}`);
      if (nextData.detail) {
        setLoading(nextData.detail);
        return;
      }
      setLoading('');
      setData(nextData);
    } catch (error) {
      setLoading(`Error: ${error.message}`);
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div><div className="panel-title">Log pattern clustering</div><div className="panel-meta">Drain3 template mining - strips variables, groups logs by structure</div></div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select className="select-sm" value={hours} onChange={(event) => setHours(event.target.value)}><option value="1">Last 1h</option><option value="2">Last 2h</option><option value="6">Last 6h</option></select>
          <button className="btn-sm" onClick={loadLogPatterns} type="button">Analyse logs</button>
          <InfoButton topic="logs" onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {data ? (
          <div className="stat-row" style={{ marginBottom: '1rem' }}>
            <StatCard label="Total logs" value={data.total_logs} />
            <StatCard label="Unique patterns" value={data.unique_patterns} color="blue" />
            <StatCard label="New patterns" value={data.new_patterns} color="amber" />
            <StatCard label="Error patterns" value={data.error_patterns} color="red" />
          </div>
        ) : null}
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        {data ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="det-table">
              <thead><tr><th>Pattern template</th><th>Count</th><th>%</th><th>Service</th><th>Severity</th><th>Flags</th></tr></thead>
              <tbody>{data.patterns.map((pattern) => (
                <tr key={`${pattern.template}-${pattern.count}`}>
                  <td style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'monospace', fontSize: '0.75rem' }} title={pattern.template}>{pattern.template}</td>
                  <td>{pattern.count}</td>
                  <td style={{ color: 'var(--text-3)' }}>{pattern.pct_of_total}%</td>
                  <td><span className="tag">{pattern.top_service}</span></td>
                  <td><span className={`tag ${pattern.dominant_severity === 'ERROR' ? 'red' : pattern.dominant_severity === 'WARNING' ? 'amber' : 'blue'}`}>{pattern.dominant_severity}</span></td>
                  <td>{pattern.is_new ? <span className="tag amber">NEW</span> : null}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function TraceShapesPanel({ onOpenInfo, service }) {
  const [hours, setHours] = useState('2');
  const [loading, setLoading] = useState('Click "Analyse traces" to fingerprint trace shapes.');
  const [data, setData] = useState(null);
  const [copied, setCopied] = useState('');

  async function loadTraceShapes() {
    setLoading('Fingerprinting trace shapes...');
    setData(null);
    try {
      const params = new URLSearchParams({ hours, service });
      const nextData = await apiGet(`/detective/trace-shapes?${params}`);
      if (nextData.detail) {
        setLoading(nextData.detail);
        return;
      }
      setLoading('');
      setData(nextData);
    } catch (error) {
      setLoading(`Error: ${error.message}`);
    }
  }

  function copyTraceId(traceId) {
    copyText(traceId, () => {
      setCopied(traceId);
      setTimeout(() => setCopied(''), 1500);
    });
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div><div className="panel-title">Trace shape anomaly detection</div><div className="panel-meta">Fingerprints each trace's span structure - flags deviations from baseline</div></div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select className="select-sm" value={hours} onChange={(event) => setHours(event.target.value)}><option value="1">Last 1h</option><option value="2">Last 2h</option><option value="6">Last 6h</option></select>
          <button className="btn-sm" onClick={loadTraceShapes} type="button">Analyse traces</button>
          <InfoButton topic="traces" onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {data ? (
          <div className="stat-row" style={{ marginBottom: '1rem' }}>
            <StatCard label="Total traces" value={data.total_traces} />
            <StatCard label="Unique shapes" value={data.unique_shapes} color="blue" />
            <StatCard label="Anomalous traces" value={data.anomalous_count} color="red" />
            <StatCard label="Baseline coverage" value={`${data.baseline_pct}%`} color="green" />
          </div>
        ) : null}
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        {data ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="det-table">
              <thead><tr><th>Shape</th><th>Count</th><th>%</th><th>Deviation</th><th>Example trace IDs</th></tr></thead>
              <tbody>{data.shape_summary.slice(0, 10).map((shape) => (
                <tr key={shape.fingerprint}>
                  <td style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'monospace', fontSize: '0.72rem', cursor: 'help' }} title={shape.fingerprint}>{shape.fingerprint}</td>
                  <td>{shape.count}</td>
                  <td style={{ color: 'var(--text-3)' }}>{shape.pct_of_total}%</td>
                  <td>{shape.is_baseline ? <span className="tag green">baseline</span> : shape.deviation_type ? <span className="tag red">{shape.deviation_type.replace(/_/g, ' ')}</span> : <span className="tag">-</span>}</td>
                  <td>{shape.example_traces.length ? shape.example_traces.map((traceId) => (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: 3 }} key={traceId}>
                      <a href={traceExploreUrl(traceId)} target="_blank" rel="noopener noreferrer" title="Open trace in Grafana Tempo" className="trace-link" style={{ fontSize: '0.72rem' }}>{traceId}</a>
                      <button onClick={() => copyTraceId(traceId)} title="Copy trace ID" style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 4, padding: '1px 5px', fontSize: '0.65rem', cursor: 'pointer', color: copied === traceId ? 'var(--green)' : 'var(--text-3)', fontFamily: 'var(--sans)' }} type="button">{copied === traceId ? 'copied!' : 'copy'}</button>
                    </div>
                  )) : '-'}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : null}
      </div>
    </div>
  );
}
