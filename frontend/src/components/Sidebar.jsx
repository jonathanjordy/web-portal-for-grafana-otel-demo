import { useEffect, useState } from 'react';
import { apiGet } from '../services/api.js';

const navGroups = [
  {
    label: 'Analytics',
    items: [
      { id: 'predictive', label: 'Predictive', icon: <PredictiveIcon /> },
      { id: 'detective', label: 'Detective', icon: <DetectiveIcon /> },
      { id: 'diagnostic', label: 'Diagnostic', icon: <DiagnosticIcon /> },
    ],
  },
  {
    label: 'Assistant',
    items: [{ id: 'chatbot', label: 'AIOps Chat', icon: <ChatIcon /> }],
  },
  {
    label: 'Operations',
    items: [{ id: 'incidents', label: 'Incident Registry', icon: <IncidentIcon /> }],
  },
];

export default function Sidebar({ activePage, onNavigate }) {
  const [health, setHealth] = useState({ className: 'dot', text: 'Checking...' });

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const data = await apiGet('/health', { signal: AbortSignal.timeout(4000) });
        if (cancelled) return;
        setHealth(data.clickhouse === 'ok'
          ? { className: 'dot ok', text: 'ClickHouse connected' }
          : { className: 'dot error', text: 'ClickHouse error' });
      } catch {
        if (!cancelled) setHealth({ className: 'dot error', text: 'API unreachable' });
      }
    }

    checkHealth();
    const timer = setInterval(checkHealth, 30000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <aside>
      <div className="sidebar-top">
        <div className="wordmark">AIOps Portal<span>Observability Intelligence</span></div>
      </div>
      {navGroups.map((group) => (
        <div className="nav-group" key={group.label}>
          <div className="nav-label">{group.label}</div>
          {group.items.map((item) => (
            <button
              className={`nav-item ${activePage === item.id ? 'active' : ''}`}
              data-page={item.id}
              key={item.id}
              onClick={() => onNavigate(item.id)}
              type="button"
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      ))}
      <div className="sidebar-footer">
        <div className="health-row">
          <div className={health.className} />
          <span>{health.text}</span>
        </div>
      </div>
    </aside>
  );
}

function PredictiveIcon() {
  return <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1,12 5,7 8,9 12,4 15,5" /><line x1="15" y1="2" x2="15" y2="8" /><line x1="12" y1="8" x2="15" y2="8" /></svg>;
}

function DetectiveIcon() {
  return <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="6.5" cy="6.5" r="4.5" /><line x1="10" y1="10" x2="14" y2="14" /></svg>;
}

function DiagnosticIcon() {
  return <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="8" cy="8" r="6.5" /><line x1="8" y1="5" x2="8" y2="8" /><circle cx="8" cy="11" r="0.5" fill="currentColor" /></svg>;
}

function ChatIcon() {
  return <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 10.5c0 .8-.7 1.5-1.5 1.5H4l-2.5 2.5V3.5C1.5 2.7 2.2 2 3 2h9.5c.8 0 1.5.7 1.5 1.5v7z" /></svg>;
}

function IncidentIcon() {
  return <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="2" width="12" height="12" rx="2" /><line x1="5" y1="6" x2="11" y2="6" /><line x1="5" y1="9" x2="9" y2="9" /></svg>;
}
