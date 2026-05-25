import { useMemo, useState } from 'react';
import { buildIncidentSeed } from '../data/incidents.js';

const filters = ['all', 'active', 'investigating', 'resolved'];

export default function IncidentsPage() {
  const [currentFilter, setCurrentFilter] = useState('all');
  const [incidents, setIncidents] = useState(() => buildIncidentSeed());
  const [showModal, setShowModal] = useState(false);
  const [detailId, setDetailId] = useState('');

  const filtered = currentFilter === 'all' ? incidents : incidents.filter((incident) => incident.status === currentFilter);
  const selected = incidents.find((incident) => incident.id === detailId);

  function resolveIncident(id) {
    setIncidents((current) => current.map((incident) => {
      if (incident.id !== id) return incident;
      const resolvedAt = new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
      return { ...incident, status: 'resolved', opened: `${incident.opened} -> resolved ${resolvedAt}` };
    }));
  }

  function createIncident(newIncident) {
    setIncidents((current) => [
      { ...newIncident, id: `INC-${String(current.length + 1).padStart(3, '0')}`, status: 'active', activity: [] },
      ...current,
    ]);
    setCurrentFilter('all');
    setShowModal(false);
  }

  return (
    <div className="page active">
      <div className="page-eyebrow">Operations</div>
      <h1 className="page-title">Incident Registry</h1>
      <p className="page-desc">Track, manage, and resolve operational incidents. Create new incidents manually or let the Diagnostic engine raise them automatically.</p>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: '0.6rem' }}>
          {filters.map((filter) => (
            <button className={`filter-btn ${currentFilter === filter ? 'active' : ''}`} onClick={() => setCurrentFilter(filter)} key={filter} type="button">{labelFor(filter)}</button>
          ))}
        </div>
        <button className="btn" onClick={() => setShowModal(true)} type="button">+ New Incident</button>
      </div>
      <IncidentStats incidents={incidents} />
      <div className="panel">
        <div className="panel-head"><span className="panel-title">All incidents</span><span className="panel-meta">Showing {filtered.length} incident{filtered.length !== 1 ? 's' : ''}</span></div>
        <div style={{ overflowX: 'auto' }}>
          <table className="inc-table">
            <thead><tr><th>ID</th><th>Title</th><th>Service</th><th>Severity</th><th>Status</th><th>Opened</th><th>Assignee</th><th>Action</th></tr></thead>
            <tbody>{filtered.map((incident) => (
              <tr key={incident.id} style={{ cursor: 'pointer' }} onClick={() => setDetailId(incident.id)}>
                <td><span className="inc-id">{incident.id}</span></td>
                <td><span className="inc-title">{incident.title}</span></td>
                <td>{incident.service}</td>
                <td><span className={`sev-badge sev-${incident.severity}`}>{incident.severity}</span></td>
                <td><span className={`status-badge-sm status-${incident.status}`}>{incident.status}</span></td>
                <td style={{ whiteSpace: 'nowrap', color: 'var(--text-3)', fontSize: '0.8rem' }}>{incident.opened}</td>
                <td style={{ color: incident.assignee === 'Unassigned' ? 'var(--text-3)' : 'var(--text-2)' }}>{incident.assignee}</td>
                <td>{incident.status !== 'resolved' ? <button className="inc-action" onClick={(event) => { event.stopPropagation(); resolveIncident(incident.id); }} type="button">Resolve</button> : <span style={{ color: 'var(--text-3)', fontSize: '0.8rem' }}>Closed</span>}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>
      {selected ? <IncidentDetail incident={selected} onClose={() => setDetailId('')} onResolve={(id) => { resolveIncident(id); setDetailId(''); }} /> : null}
      {showModal ? <NewIncidentModal onClose={() => setShowModal(false)} onCreate={createIncident} /> : null}
    </div>
  );
}

function IncidentStats({ incidents }) {
  const counts = useMemo(() => ({
    active: incidents.filter((incident) => incident.status === 'active').length,
    investigating: incidents.filter((incident) => incident.status === 'investigating').length,
    resolved: incidents.filter((incident) => incident.status === 'resolved').length,
  }), [incidents]);

  return (
    <div className="stat-row" style={{ marginBottom: '1.5rem' }}>
      <div className="stat"><div className="stat-label">Active</div><div className="stat-value red">{counts.active}</div><div className="stat-sub">require attention</div></div>
      <div className="stat"><div className="stat-label">Investigating</div><div className="stat-value amber">{counts.investigating}</div><div className="stat-sub">in progress</div></div>
      <div className="stat"><div className="stat-label">Resolved today</div><div className="stat-value green">{counts.resolved}</div><div className="stat-sub">closed incidents</div></div>
      <div className="stat"><div className="stat-label">Avg resolution time</div><div className="stat-value">42m</div><div className="stat-sub">last 7 days</div></div>
    </div>
  );
}

function IncidentDetail({ incident, onClose, onResolve }) {
  return (
    <div className="modal-overlay open" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 580 }} onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}><span className="inc-id" style={{ fontSize: '0.85rem' }}>{incident.id}</span><span className="modal-title">{incident.title}</span></div>
          <button className="modal-close" onClick={onClose} type="button">x</button>
        </div>
        <div className="modal-body" style={{ gap: '1.4rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="detail-field"><div className="detail-label">Service</div><div className="detail-value">{incident.service}</div></div>
            <div className="detail-field"><div className="detail-label">Severity</div><span className={`sev-badge sev-${incident.severity}`}>{incident.severity}</span></div>
            <div className="detail-field"><div className="detail-label">Status</div><span className={`status-badge-sm status-${incident.status}`}>{incident.status}</span></div>
            <div className="detail-field"><div className="detail-label">Assignee</div><div className="detail-value">{incident.assignee}</div></div>
            <div className="detail-field" style={{ gridColumn: 'span 2' }}><div className="detail-label">Opened</div><div className="detail-value">{incident.opened}</div></div>
          </div>
          <div className="detail-field"><div className="detail-label">Description</div><div className="detail-desc">{incident.desc || 'No description provided.'}</div></div>
          <div className="detail-field"><div className="detail-label">Activity log</div><div className="activity-log">{incident.activity?.length ? incident.activity.map((item) => <div className="activity-item" key={`${item.time}-${item.text}`}><span className="activity-time">{item.time}</span><div className={`activity-dot ${item.dot}`} /><span>{item.text}</span></div>) : <span style={{ color: 'var(--text-3)', fontSize: '0.85rem' }}>No activity yet.</span>}</div></div>
        </div>
        <div className="modal-foot">
          {incident.status !== 'resolved' ? (
            <>
              <button className="btn btn-ghost" onClick={onClose} type="button">Close</button>
              <button className="btn" onClick={() => onResolve(incident.id)} type="button">Mark as resolved</button>
            </>
          ) : <button className="btn" onClick={onClose} type="button">Close</button>}
        </div>
      </div>
    </div>
  );
}

function NewIncidentModal({ onClose, onCreate }) {
  const [title, setTitle] = useState('');
  const [service, setService] = useState('order-service');
  const [severity, setSeverity] = useState('medium');
  const [assignee, setAssignee] = useState('');
  const [desc, setDesc] = useState('');
  const [titleError, setTitleError] = useState(false);

  function handleCreate() {
    if (!title.trim()) {
      setTitleError(true);
      return;
    }
    const now = new Date();
    const opened = `${now.toLocaleDateString('id-ID', { year: 'numeric', month: '2-digit', day: '2-digit' }).split('/').reverse().join('-')} ${now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })}`;
    onCreate({ title: title.trim(), service, severity, opened, assignee: assignee.trim() || 'Unassigned', desc: desc.trim() });
  }

  return (
    <div className="modal-overlay open" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head"><span className="modal-title">New Incident</span><button className="modal-close" onClick={onClose} type="button">x</button></div>
        <div className="modal-body">
          <div className="form-group"><label>Title</label><input type="text" value={title} placeholder="Brief description of the incident" onChange={(event) => { setTitle(event.target.value); setTitleError(false); }} style={{ borderColor: titleError ? 'var(--red)' : undefined }} /></div>
          <div className="form-row">
            <div className="form-group"><label>Service</label><select value={service} onChange={(event) => setService(event.target.value)}><option>order-service</option><option>inventory-service</option><option>payment-service</option><option>node / infrastructure</option><option>multiple services</option></select></div>
            <div className="form-group"><label>Severity</label><select value={severity} onChange={(event) => setSeverity(event.target.value)}><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></div>
          </div>
          <div className="form-group"><label>Assignee</label><input type="text" value={assignee} placeholder="Who is handling this?" onChange={(event) => setAssignee(event.target.value)} /></div>
          <div className="form-group"><label>Description</label><textarea value={desc} rows="3" placeholder="What is happening? Any initial findings?" onChange={(event) => setDesc(event.target.value)} /></div>
        </div>
        <div className="modal-foot">
          <button className="btn btn-ghost" onClick={onClose} type="button">Cancel</button>
          <button className="btn" onClick={handleCreate} type="button">Create incident</button>
        </div>
      </div>
    </div>
  );
}

function labelFor(filter) {
  if (filter === 'all') return 'All';
  if (filter === 'active') return 'Active';
  if (filter === 'investigating') return 'Investigating';
  return 'Resolved';
}
