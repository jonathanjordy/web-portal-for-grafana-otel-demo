import { useState } from 'react';
import InfoButton from '../components/InfoButton.jsx';
import { FilterSelect, useFilters } from '../hooks/useFilters.jsx';
import { apiGet, apiPost } from '../services/api.js';
import { copyText } from '../utils/copyText.js';
import { traceExploreUrl } from '../utils/traceUrl.js';

export default function DiagnosticPage({ onOpenInfo }) {
  const filters = useFilters();

  return (
    <div className="page active">
      <div className="page-eyebrow">Page 3 - Diagnostic Analytics</div>
      <h1 className="page-title">Root Cause Analysis</h1>
      <p className="page-desc">Automated RCA that connects the dots across metrics, traces, and logs - and hands you a human-readable incident summary.</p>
      <CausalGraphPanel onOpenInfo={onOpenInfo} services={filters.services} />
      <CorrelationPanel onOpenInfo={onOpenInfo} services={filters.services} />
      <SummaryPanel onOpenInfo={onOpenInfo} services={filters.services} />
    </div>
  );
}

function CausalGraphPanel({ onOpenInfo, services }) {
  const [hours, setHours] = useState('1');
  const [service, setService] = useState('');
  const [loading, setLoading] = useState('Click "Build graph" to generate the live service dependency map.');
  const [data, setData] = useState(null);

  async function loadCausalGraph() {
    setLoading('Building dependency graph from traces...');
    setData(null);
    try {
      const params = new URLSearchParams({ hours, service });
      const nextData = await apiGet(`/diagnostic/causal-graph?${params}`);
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
        <div><div className="panel-title">Service dependency graph</div><div className="panel-meta">Built from trace parent-child relationships - highlights root cause service</div></div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <ServiceSelect value={service} onChange={setService} services={services} />
          <select className="select-sm" value={hours} onChange={(event) => setHours(event.target.value)}><option value="1">Last 1h</option><option value="3">Last 3h</option><option value="6">Last 6h</option></select>
          <button className="btn-sm" onClick={loadCausalGraph} type="button">Build graph</button>
          <InfoButton topic="graph" onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        {data ? (
          <div>
            <DependencyGraph data={data} />
            <div style={{ overflowX: 'auto' }}>
              <table className="det-table" style={{ marginTop: '0.5rem' }}>
                <thead><tr><th>Service</th><th>Calls</th><th>Errors</th><th>Error rate</th><th>Avg latency</th><th>Status</th></tr></thead>
                <tbody>{data.nodes.map((node) => (
                  <tr key={node.id}>
                    <td style={{ fontWeight: 700 }}>{node.id}</td>
                    <td>{node.calls}</td>
                    <td>{node.errors}</td>
                    <td><span className={`tag ${node.error_rate > 10 ? 'red' : node.error_rate > 2 ? 'amber' : 'green'}`}>{node.error_rate}%</span></td>
                    <td>{node.avg_duration_ms}ms</td>
                    <td><span className={`tag ${node.status === 'error' ? 'red' : node.status === 'degraded' ? 'amber' : 'green'}`}>{node.status}</span>{node.id === data.root_cause ? ' ' : null}{node.id === data.root_cause ? <span className="tag red">root cause</span> : null}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function DependencyGraph({ data }) {
  const width = 700;
  const height = 260;
  const spacing = width / (data.nodes.length + 1 || 1);
  const positions = Object.fromEntries(data.nodes.map((node, index) => [node.id, { x: spacing * (index + 1), y: height / 2 }]));
  const colorFor = (status) => (status === 'error' ? '#EB008C' : status === 'degraded' ? '#d4a000' : '#24BCAD');

  return (
    <div style={{ background: 'var(--surface2)', borderRadius: 10, minHeight: 260, position: 'relative', overflow: 'hidden', marginBottom: '1rem' }}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} xmlns="http://www.w3.org/2000/svg">
        <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" strokeWidth="1.5" /></marker></defs>
        {data.edges.map((edge) => {
          const source = positions[edge.source];
          const target = positions[edge.target];
          if (!source || !target) return null;
          const color = edge.error_rate > 10 ? '#EB008C' : edge.error_rate > 2 ? '#d4a000' : '#24BCAD';
          const mx = (source.x + target.x) / 2;
          const my = (source.y + target.y) / 2 - 30;
          return (
            <g key={`${edge.source}-${edge.target}`}>
              <path d={`M${source.x},${source.y} Q${mx},${my} ${target.x},${target.y}`} fill="none" stroke={color} strokeWidth="2" strokeOpacity="0.6" markerEnd="url(#arr)" />
              <text x={mx} y={my - 6} textAnchor="middle" fontSize="10" fill={color} fontFamily="Nunito">{edge.avg_duration_ms}ms - {edge.error_rate}% err</text>
            </g>
          );
        })}
        {data.nodes.map((node) => {
          const pos = positions[node.id];
          const color = colorFor(node.status);
          const isRoot = node.id === data.root_cause;
          return (
            <g key={node.id}>
              <circle cx={pos.x} cy={pos.y} r={isRoot ? 36 : 28} fill={`${color}22`} stroke={color} strokeWidth={isRoot ? 3 : 1.5} />
              {isRoot ? <circle cx={pos.x} cy={pos.y} r="42" fill="none" stroke={color} strokeWidth="1" strokeDasharray="4 3" opacity="0.5" /> : null}
              <text x={pos.x} y={pos.y - 4} textAnchor="middle" fontSize="11" fontWeight="700" fill={color} fontFamily="Nunito">{node.id.replace('-service', '')}</text>
              <text x={pos.x} y={pos.y + 11} textAnchor="middle" fontSize="10" fill={color} fontFamily="Nunito">{node.error_rate}% err</text>
              {isRoot ? <text x={pos.x} y={pos.y + 58} textAnchor="middle" fontSize="10" fontWeight="700" fill={color} fontFamily="Nunito">root cause</text> : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function CorrelationPanel({ onOpenInfo, services }) {
  const [hours, setHours] = useState('1');
  const [service, setService] = useState('');
  const [loading, setLoading] = useState('Click "Correlate" to surface the slowest traces, error logs, and metric anomalies side by side.');
  const [data, setData] = useState(null);
  const [copied, setCopied] = useState('');

  async function loadCorrelation() {
    setLoading('Correlating telemetry signals...');
    setData(null);
    try {
      const params = new URLSearchParams({ hours, service });
      const nextData = await apiGet(`/diagnostic/correlate?${params}`);
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

  function handleCopy(value) {
    copyText(value, () => {
      setCopied(value);
      setTimeout(() => setCopied(''), 1500);
    });
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div><div className="panel-title">Telemetry correlation engine</div><div className="panel-meta">Slowest traces - error logs - metric spikes - all in one view</div></div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <ServiceSelect value={service} onChange={setService} services={services} />
          <select className="select-sm" value={hours} onChange={(event) => setHours(event.target.value)}><option value="1">Last 1h</option><option value="3">Last 3h</option><option value="6">Last 6h</option></select>
          <button className="btn-sm" onClick={loadCorrelation} type="button">Correlate</button>
          <InfoButton topic="correlate" onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        {data ? <CorrelationResults data={data} copied={copied} onCopy={handleCopy} /> : null}
      </div>
    </div>
  );
}

function CorrelationResults({ data, copied, onCopy }) {
  return (
    <div>
      <ResultSection title="Slowest traces" rows={data.slow_traces} headers={['Trace ID', 'Service', 'Span', 'Duration', 'Status']} renderRow={(row) => (
        <tr key={`${row.TraceId}-${row.SpanName}`}>
          <td style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}><TraceCopy traceId={row.TraceId} copied={copied} onCopy={onCopy} /></td>
          <td><span className="tag">{row.ServiceName}</span></td>
          <td style={{ fontSize: '0.78rem' }}>{row.SpanName}</td>
          <td><span className={`tag ${Number(row.Duration) / 1e6 > 1000 ? 'red' : 'amber'}`}>{(Number(row.Duration) / 1e6).toFixed(0)}ms</span></td>
          <td><span className={`tag ${row.StatusCode === 'STATUS_CODE_ERROR' ? 'red' : 'green'}`}>{row.StatusCode === 'STATUS_CODE_ERROR' ? 'error' : 'ok'}</span></td>
        </tr>
      )} />
      <ResultSection title="Error traces" rows={data.error_traces} headers={['Trace ID', 'Service', 'Span', 'Timestamp']} renderRow={(row) => (
        <tr key={`${row.TraceId}-${row.Timestamp}`}>
          <td style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}><TraceCopy traceId={row.TraceId} copied={copied} onCopy={onCopy} /></td>
          <td><span className="tag red">{row.ServiceName}</span></td>
          <td style={{ fontSize: '0.78rem' }}>{row.SpanName}</td>
          <td style={{ fontSize: '0.75rem', color: 'var(--text-3)' }}>{String(row.Timestamp).slice(0, 16)}</td>
        </tr>
      )} />
      <ResultSection title="Error logs" rows={data.error_logs} headers={['Trace ID', 'Service', 'Message', 'Time']} renderRow={(row) => (
        <tr key={`${row.TraceId}-${row.Timestamp}-${row.Body}`}>
          <td style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}>{row.TraceId ? <TraceCopy traceId={row.TraceId} copied={copied} onCopy={onCopy} /> : <span style={{ color: 'var(--text-3)' }}>-</span>}</td>
          <td><span className="tag red">{row.service_name || '-'}</span></td>
          <td style={{ fontSize: '0.78rem', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={row.Body}>{row.Body}</td>
          <td style={{ fontSize: '0.75rem', color: 'var(--text-3)', whiteSpace: 'nowrap' }}>{String(row.Timestamp).slice(0, 16)}</td>
        </tr>
      )} />
      <ResultSection title="Metric totals" rows={data.metric_summary} headers={['Metric', 'Total', 'Avg per min']} renderRow={(row) => (
        <tr key={row.MetricName}>
          <td><span className="tag">{row.MetricName}</span></td>
          <td style={{ fontWeight: 700 }}>{Number(row.total).toFixed(0)}</td>
          <td style={{ color: 'var(--text-3)' }}>{(Number(row.avg_val) || 0).toFixed(2)}</td>
        </tr>
      )} />
    </div>
  );
}

function ResultSection({ title, rows, headers, renderRow }) {
  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>{title} ({rows.length})</div>
      {rows.length ? (
        <table className="det-table">
          <thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead>
          <tbody>{rows.map(renderRow)}</tbody>
        </table>
      ) : <div style={{ color: 'var(--text-3)', fontSize: '0.82rem' }}>None found.</div>}
    </div>
  );
}

function TraceCopy({ traceId, copied, onCopy }) {
  return (
    <>
      <a
        href={traceExploreUrl(traceId)}
        target="_blank"
        rel="noopener noreferrer"
        title="Open trace in Grafana Tempo"
        className="trace-link"
      >
        {String(traceId).slice(0, 16)}...
      </a>
      <button onClick={() => onCopy(traceId)} style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 4, padding: '1px 4px', fontSize: '0.62rem', cursor: 'pointer', color: copied === traceId ? 'var(--green)' : 'var(--text-3)', marginLeft: 4 }} type="button">
        {copied === traceId ? 'copied!' : 'copy'}
      </button>
    </>
  );
}

function SummaryPanel({ onOpenInfo, services }) {
  const [hours, setHours] = useState('1');
  const [service, setService] = useState('');
  const [loading, setLoading] = useState('Click "Generate summary" to have Claude analyse your telemetry and produce an incident report.');
  const [summary, setSummary] = useState('');
  const [copied, setCopied] = useState(false);

  async function loadSummary() {
    setLoading('Claude is analysing your telemetry...');
    setSummary('');
    try {
      const data = await apiPost('/diagnostic/summarize', { hours: Number(hours), service });
      if (data.detail) {
        setLoading(data.detail);
        return;
      }
      setLoading('');
      setSummary(data.summary);
    } catch (error) {
      setLoading(`Error: ${error.message}`);
    }
  }

  function copySummary() {
    copyText(summary, () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div><div className="panel-title">AI incident summarization</div><div className="panel-meta">Powered by Claude - generates a plain-English RCA ready to post to Slack</div></div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <ServiceSelect value={service} onChange={setService} services={services} />
          <select className="select-sm" value={hours} onChange={(event) => setHours(event.target.value)}><option value="1">Last 1h</option><option value="3">Last 3h</option><option value="6">Last 6h</option></select>
          <button className="btn-sm" onClick={loadSummary} type="button">Generate summary</button>
          <InfoButton topic="llm" onOpen={onOpenInfo} />
        </div>
      </div>
      <div className="panel-body">
        {loading ? <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>{loading}</div> : null}
        {summary ? (
          <div>
            <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 10, padding: '1.25rem', fontSize: '0.9rem', lineHeight: 1.7, color: 'var(--text-2)', whiteSpace: 'pre-wrap', marginBottom: '1rem' }}>{summary}</div>
            <button className="btn-sm" onClick={copySummary} type="button">{copied ? 'Copied!' : 'Copy to clipboard'}</button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ServiceSelect({ value, onChange, services }) {
  return (
    <FilterSelect label="Service" value={value} options={services} onChange={onChange} allLabel="All services" />
  );
}
