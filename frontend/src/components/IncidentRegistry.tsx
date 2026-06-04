'use client';

import React, { useState } from 'react';
import { Incident } from '../types/otel';

const INITIAL_INCIDENTS: Incident[] = [
  { 
    id: 'INC-001', 
    title: 'Payment gateway rejection rate elevated', 
    service: 'payment-service', 
    severity: 'critical', 
    status: 'active', 
    opened: '2026-04-29 08:14', 
    assignee: 'Ahmad R.', 
    desc: 'Payment failures spiked to 45% at 08:12. Gateway returning 402 consistently.',
    activity: [
      { time: '08:12', dot: 'red', text: 'Anomaly detected — failure rate 45%' },
      { time: '08:14', dot: 'red', text: 'Incident raised automatically' },
      { time: '08:17', dot: 'amber', text: 'Ahmad R. assigned and started investigation' },
      { time: '08:22', dot: 'blue', text: 'Checked payment-service logs — gateway returning 402 consistently' }
    ]
  },
  { 
    id: 'INC-002', 
    title: 'Inventory DB slow query detected', 
    service: 'inventory-service', 
    severity: 'high', 
    status: 'investigating', 
    opened: '2026-04-29 07:55', 
    assignee: 'Siti N.', 
    desc: 'db-stock-lookup span averaging 2.1s. Suspected missing index on stock table.',
    activity: [
      { time: '07:50', dot: 'amber', text: 'Slow span alert fired — db-stock-lookup > 2s' },
      { time: '07:55', dot: 'red', text: 'Incident raised' },
      { time: '08:01', dot: 'amber', text: 'Siti N. assigned' },
      { time: '08:10', dot: 'blue', text: 'Reviewing EXPLAIN output on stock table query' }
    ]
  },
  { 
    id: 'INC-003', 
    title: 'Memory usage trending upward on node', 
    service: 'node / infrastructure', 
    severity: 'medium', 
    status: 'investigating', 
    opened: '2026-04-29 06:30', 
    assignee: 'Budi S.', 
    desc: 'Prophet forecast predicts memory saturation in ~18 hours if trend continues.',
    activity: [
      { time: '06:28', dot: 'amber', text: 'Prophet forecast predicted saturation in 18h' },
      { time: '06:30', dot: 'red', text: 'Incident raised' },
      { time: '06:45', dot: 'amber', text: 'Budi S. assigned — monitoring trend' }
    ]
  },
  { 
    id: 'INC-004', 
    title: 'Order service response time p99 > 3s', 
    service: 'order-service', 
    severity: 'high', 
    status: 'active', 
    opened: '2026-04-29 09:01', 
    assignee: 'Unassigned', 
    desc: 'p99 latency crossed 3s threshold. Correlated with INC-002 slow inventory lookup.',
    activity: [
      { time: '09:01', dot: 'red', text: 'p99 latency crossed 3s threshold' },
      { time: '09:03', dot: 'blue', text: 'Correlated with INC-002 inventory slow query' }
    ]
  },
  { 
    id: 'INC-005', 
    title: 'Redis cache hit rate dropped below 50%', 
    service: 'inventory-service', 
    severity: 'medium', 
    status: 'active', 
    opened: '2026-04-29 09:15', 
    assignee: 'Siti N.', 
    desc: 'Cache hit rate fell from 85% to 43%. Possible cache eviction or TTL issue.',
    activity: [
      { time: '09:12', dot: 'amber', text: 'Cache hit rate monitoring alert fired' },
      { time: '09:15', dot: 'red', text: 'Incident raised' },
      { time: '09:18', dot: 'amber', text: 'Siti N. assigned — checking Redis TTL config' }
    ]
  },
  { 
    id: 'INC-006', 
    title: 'Intermittent connection timeout to ClickHouse', 
    service: 'multiple services', 
    severity: 'low', 
    status: 'resolved', 
    opened: '2026-04-28 22:10', 
    assignee: 'Ahmad R.', 
    desc: 'OTel collector reported retry errors. Resolved after ClickHouse restart.',
    activity: [
      { time: '22:10', dot: 'red', text: 'OTel collector retry errors detected' },
      { time: '22:15', dot: 'amber', text: 'Ahmad R. investigated — ClickHouse connection dropping' },
      { time: '22:31', dot: 'blue', text: 'ClickHouse restarted' },
      { time: '22:33', dot: 'green', text: 'Connections restored — incident resolved' }
    ]
  },
  { 
    id: 'INC-007', 
    title: 'Deployment caused 2 min downtime on order-service', 
    service: 'order-service', 
    severity: 'high', 
    status: 'resolved', 
    opened: '2026-04-28 18:00', 
    assignee: 'Budi S.', 
    desc: 'Rolling restart during peak hours caused brief unavailability. Post-mortem filed.',
    activity: [
      { time: '18:00', dot: 'red', text: 'Deployment started on order-service' },
      { time: '18:02', dot: 'red', text: 'Health check failures detected' },
      { time: '18:04', dot: 'amber', text: 'Budi S. monitoring — all pods restarting' },
      { time: '18:06', dot: 'green', text: 'All pods healthy — service restored' },
      { time: '18:10', dot: 'green', text: 'Post-mortem filed — resolved' }
    ]
  },
  { 
    id: 'INC-008', 
    title: 'Node CPU load spike during batch job', 
    service: 'node / infrastructure', 
    severity: 'medium', 
    status: 'resolved', 
    opened: '2026-04-28 14:22', 
    assignee: 'Siti N.', 
    desc: 'Load average hit 4.2 during scheduled analytics job. Resolved after job completed.',
    activity: [
      { time: '14:20', dot: 'amber', text: 'CPU load average spike — load1 = 4.2' },
      { time: '14:22', dot: 'red', text: 'Incident raised' },
      { time: '14:25', dot: 'blue', text: 'Siti N. identified scheduled analytics batch job' },
      { time: '14:55', dot: 'green', text: 'Batch job completed — load normalised — resolved' }
    ]
  },
  { 
    id: 'INC-009', 
    title: 'Payment amount histogram showing outliers > $5000', 
    service: 'payment-service', 
    severity: 'low', 
    status: 'resolved', 
    opened: '2026-04-28 11:05', 
    assignee: 'Ahmad R.', 
    desc: 'Unusually large payment amounts detected. Confirmed as load test data, not real.',
    activity: [
      { time: '11:05', dot: 'amber', text: 'Histogram outliers detected — payments > $5000' },
      { time: '11:08', dot: 'blue', text: 'Ahmad R. investigated — confirmed load test data' },
      { time: '11:15', dot: 'green', text: 'No real impact — resolved' }
    ]
  }
];

export default function IncidentRegistry() {
  const [incidents, setIncidents] = useState<Incident[]>(INITIAL_INCIDENTS);
  const [currentFilter, setCurrentFilter] = useState('all');
  const [selectedInc, setSelectedInc] = useState<Incident | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  // Form states
  const [formTitle, setFormTitle] = useState('');
  const [formService, setFormService] = useState('order-service');
  const [formSeverity, setFormSeverity] = useState<'critical' | 'high' | 'medium' | 'low'>('medium');
  const [formAssignee, setFormAssignee] = useState('');
  const [formDesc, setFormDesc] = useState('');
  const [formTitleError, setFormTitleError] = useState(false);

  const filteredIncidents = currentFilter === 'all' 
    ? incidents 
    : incidents.filter(i => i.status === currentFilter);

  const activeCount = incidents.filter(i => i.status === 'active').length;
  const investigatingCount = incidents.filter(i => i.status === 'investigating').length;
  const resolvedCount = incidents.filter(i => i.status === 'resolved').length;

  const handleResolve = (id: string) => {
    setIncidents(prev => prev.map(inc => {
      if (inc.id === id) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
        return {
          ...inc,
          status: 'resolved',
          opened: inc.opened.includes('→ resolved') ? inc.opened : `${inc.opened} → resolved ${timeStr}`
        };
      }
      return inc;
    }));
  };

  const handleResolveFromDetail = (id: string) => {
    handleResolve(id);
    setSelectedInc(null);
  };

  const handleCreateIncident = () => {
    if (!formTitle.trim()) {
      setFormTitleError(true);
      return;
    }

    const now = new Date();
    const dateStr = now.toLocaleDateString('id-ID', { year: 'numeric', month: '2-digit', day: '2-digit' }).split('/').reverse().join('-');
    const timeStr = now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
    const ts = `${dateStr} ${timeStr}`;

    const newId = `INC-${String(incidents.length + 1).padStart(3, '0')}`;
    const newInc: Incident = {
      id: newId,
      title: formTitle.trim(),
      service: formService,
      severity: formSeverity,
      status: 'active',
      opened: ts,
      assignee: formAssignee.trim() || 'Unassigned',
      desc: formDesc.trim(),
      activity: []
    };

    setIncidents(prev => [newInc, ...prev]);
    setIsCreateOpen(false);
    setCurrentFilter('all');

    // Reset Form
    setFormTitle('');
    setFormService('order-service');
    setFormSeverity('medium');
    setFormAssignee('');
    setFormDesc('');
    setFormTitleError(false);
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.5rem', gap: '1rem', flexWrap: 'wrap', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', gap: '0.6rem' }}>
          <button 
            className={`filter-btn ${currentFilter === 'all' ? 'active' : ''}`}
            onClick={() => setCurrentFilter('all')}
          >
            All
          </button>
          <button 
            className={`filter-btn ${currentFilter === 'active' ? 'active' : ''}`}
            onClick={() => setCurrentFilter('active')}
          >
            Active
          </button>
          <button 
            className={`filter-btn ${currentFilter === 'investigating' ? 'active' : ''}`}
            onClick={() => setCurrentFilter('investigating')}
          >
            Investigating
          </button>
          <button 
            className={`filter-btn ${currentFilter === 'resolved' ? 'active' : ''}`}
            onClick={() => setCurrentFilter('resolved')}
          >
            Resolved
          </button>
        </div>
        <button className="btn" onClick={() => setIsCreateOpen(true)}>+ New Incident</button>
      </div>

      <div className="stat-row" style={{ marginBottom: '1.5rem' }}>
        <div className="stat">
          <div className="stat-label">Active</div>
          <div className="stat-value red">{activeCount}</div>
          <div className="stat-sub">require attention</div>
        </div>
        <div className="stat">
          <div className="stat-label">Investigating</div>
          <div className="stat-value amber">{investigatingCount}</div>
          <div className="stat-sub">in progress</div>
        </div>
        <div className="stat">
          <div className="stat-label">Resolved today</div>
          <div className="stat-value green">{resolvedCount}</div>
          <div className="stat-sub">closed incidents</div>
        </div>
        <div className="stat">
          <div className="stat-label">Avg resolution time</div>
          <div className="stat-value">42m</div>
          <div className="stat-sub">last 7 days</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">All incidents</span>
          <span className="panel-meta">Showing {filteredIncidents.length} incident{filteredIncidents.length !== 1 ? 's' : ''}</span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="inc-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Title</th>
                <th>Service</th>
                <th>Severity</th>
                <th>Status</th>
                <th>Opened</th>
                <th>Assignee</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredIncidents.map(inc => (
                <tr key={inc.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedInc(inc)}>
                  <td><span className="inc-id">{inc.id}</span></td>
                  <td><span className="inc-title">{inc.title}</span></td>
                  <td>{inc.service}</td>
                  <td><span className={`sev-badge sev-${inc.severity}`}>{inc.severity}</span></td>
                  <td><span className={`status-badge-sm status-${inc.status}`}>{inc.status}</span></td>
                  <td style={{ whiteSpace: 'nowrap', color: 'var(--text-3)', fontSize: '0.8rem' }}>{inc.opened}</td>
                  <td style={{ color: inc.assignee === 'Unassigned' ? 'var(--text-3)' : 'var(--text-2)' }}>{inc.assignee}</td>
                  <td>
                    {inc.status !== 'resolved' ? (
                      <button 
                        className="inc-action" 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleResolve(inc.id);
                        }}
                      >
                        Resolve
                      </button>
                    ) : (
                      <span style={{ color: 'var(--text-3)', fontSize: '0.8rem' }}>Closed</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* modal detail */}
      {selectedInc && (
        <div className="modal-overlay open" onClick={() => setSelectedInc(null)}>
          <div className="modal" style={{ maxWidth: '580px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span className="inc-id" style={{ fontSize: '0.85rem' }}>{selectedInc.id}</span>
                <span className="modal-title">{selectedInc.title}</span>
              </div>
              <button className="modal-close" onClick={() => setSelectedInc(null)}>✕</button>
            </div>
            <div className="modal-body" style={{ gap: '1.4rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="detail-field">
                  <div className="detail-label">Service</div>
                  <div className="detail-value">{selectedInc.service}</div>
                </div>
                <div className="detail-field">
                  <div className="detail-label">Severity</div>
                  <div><span className={`sev-badge sev-${selectedInc.severity}`}>{selectedInc.severity}</span></div>
                </div>
                <div className="detail-field">
                  <div className="detail-label">Status</div>
                  <div><span className={`status-badge-sm status-${selectedInc.status}`}>{selectedInc.status}</span></div>
                </div>
                <div className="detail-field">
                  <div className="detail-label">Assignee</div>
                  <div className="detail-value">{selectedInc.assignee}</div>
                </div>
                <div className="detail-field" style={{ gridColumn: 'span 2' }}>
                  <div className="detail-label">Opened</div>
                  <div className="detail-value">{selectedInc.opened}</div>
                </div>
              </div>
              <div className="detail-field">
                <div className="detail-label">Description</div>
                <div className="detail-desc">{selectedInc.desc || 'No description provided.'}</div>
              </div>
              <div className="detail-field">
                <div className="detail-label">Activity log</div>
                <div className="activity-log">
                  {selectedInc.activity && selectedInc.activity.length > 0 ? (
                    selectedInc.activity.map((a, i) => (
                      <div className="activity-item" key={i}>
                        <span className="activity-time">{a.time}</span>
                        <div className={`activity-dot ${a.dot}`}></div>
                        <span>{a.text}</span>
                      </div>
                    ))
                  ) : (
                    <span style={{ color: 'var(--text-3)', fontSize: '0.85rem' }}>No activity yet.</span>
                  )}
                </div>
              </div>
            </div>
            <div className="modal-foot">
              {selectedInc.status !== 'resolved' ? (
                <>
                  <button className="btn btn-ghost" onClick={() => setSelectedInc(null)}>Close</button>
                  <button className="btn" onClick={() => handleResolveFromDetail(selectedInc.id)}>Mark as resolved</button>
                </>
              ) : (
                <button className="btn" onClick={() => setSelectedInc(null)}>Close</button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* modal create */}
      {isCreateOpen && (
        <div className="modal-overlay open" onClick={() => setIsCreateOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <span className="modal-title">New Incident</span>
              <button className="modal-close" onClick={() => setIsCreateOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label>Title</label>
                <input 
                  type="text" 
                  value={formTitle}
                  onChange={(e) => {
                    setFormTitle(e.target.value);
                    if (e.target.value) setFormTitleError(false);
                  }}
                  style={{ borderColor: formTitleError ? 'var(--red)' : '' }}
                  placeholder="Brief description of the incident"
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Service</label>
                  <select 
                    value={formService}
                    onChange={(e) => setFormService(e.target.value)}
                  >
                    <option value="order-service">order-service</option>
                    <option value="inventory-service">inventory-service</option>
                    <option value="payment-service">payment-service</option>
                    <option value="node / infrastructure">node / infrastructure</option>
                    <option value="multiple services">multiple services</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Severity</label>
                  <select 
                    value={formSeverity}
                    onChange={(e) => setFormSeverity(e.target.value as 'critical' | 'high' | 'medium' | 'low')}
                  >
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label>Assignee</label>
                <input 
                  type="text" 
                  value={formAssignee}
                  onChange={(e) => setFormAssignee(e.target.value)}
                  placeholder="Who is handling this?"
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea 
                  rows={3} 
                  value={formDesc}
                  onChange={(e) => setFormDesc(e.target.value)}
                  placeholder="What is happening? Any initial findings?"
                />
              </div>
            </div>
            <div className="modal-foot">
              <button className="btn btn-ghost" onClick={() => setIsCreateOpen(false)}>Cancel</button>
              <button className="btn" onClick={handleCreateIncident}>Create incident</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
