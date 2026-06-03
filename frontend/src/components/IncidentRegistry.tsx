'use client';

import React, { useState } from 'react';
import { 
  ShieldAlert, 
  Plus, 
  User, 
  Clock, 
  FileText, 
  X, 
  CheckCircle2, 
  AlertTriangle 
} from 'lucide-react';
import { Incident, ActivityLog } from '../types/otel';

// Original static incident seeds from the codebase
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

interface IncidentRegistryProps {
  apiBase: string;
}

export default function IncidentRegistry({ apiBase }: IncidentRegistryProps) {
  const [incidents, setIncidents] = useState<Incident[]>(INITIAL_INCIDENTS);
  const [filter, setFilter] = useState<'all' | 'active' | 'investigating' | 'resolved'>('all');
  
  // Modals visibility state
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [selectedInc, setSelectedInc] = useState<Incident | null>(null);

  // Form Fields State
  const [formTitle, setFormTitle] = useState('');
  const [formService, setFormService] = useState('order-service');
  const [formSeverity, setFormSeverity] = useState<'critical' | 'high' | 'medium' | 'low'>('medium');
  const [formAssignee, setFormAssignee] = useState('');
  const [formDesc, setFormDesc] = useState('');
  const [formTitleError, setFormTitleError] = useState(false);

  // Calculation Stats
  const activeCount = incidents.filter(i => i.status === 'active').length;
  const investigatingCount = incidents.filter(i => i.status === 'investigating').length;
  const resolvedCount = incidents.filter(i => i.status === 'resolved').length;

  const handleResolve = (id: string) => {
    setIncidents(prev => prev.map(inc => {
      if (inc.id === id) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
        const updatedActivity = [
          ...inc.activity,
          { time: timeStr, dot: 'green' as const, text: 'Marked as resolved' }
        ];
        return {
          ...inc,
          status: 'resolved' as const,
          opened: `${inc.opened} → resolved ${timeStr}`,
          activity: updatedActivity
        };
      }
      return inc;
    }));
    
    // Auto-update selected incident detail if currently open
    if (selectedInc?.id === id) {
      setSelectedInc(prev => prev ? { ...prev, status: 'resolved', activity: [...prev.activity, { time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }), dot: 'green', text: 'Marked as resolved' }] } : null);
    }
  };

  const handleCreateIncident = () => {
    if (!formTitle.trim()) {
      setFormTitleError(true);
      return;
    }
    
    const now = new Date();
    const dateStr = now.toLocaleDateString('id-ID', { year: 'numeric', month: '2-digit', day: '2-digit' }).split('/').reverse().join('-');
    const timeStr = now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
    const timestamp = `${dateStr} ${timeStr}`;

    const newId = `INC-${String(incidents.length + 1).padStart(3, '0')}`;
    const newInc: Incident = {
      id: newId,
      title: formTitle.trim(),
      service: formService,
      severity: formSeverity,
      status: 'active',
      opened: timestamp,
      assignee: formAssignee.trim() || 'Unassigned',
      desc: formDesc.trim(),
      activity: [
        { time: timeStr, dot: 'red', text: 'Incident opened manually' }
      ]
    };

    setIncidents(prev => [newInc, ...prev]);
    
    // Clear Form & Close
    setFormTitle('');
    setFormAssignee('');
    setFormDesc('');
    setFormTitleError(false);
    setIsCreateOpen(false);
    setFilter('all');
  };

  const filteredIncidents = filter === 'all' 
    ? incidents 
    : incidents.filter(i => i.status === filter);

  const getSeverityBadgeColor = (sev: string) => {
    switch (sev) {
      case 'critical': return 'bg-status-error-bg text-indosat-magenta border border-indosat-magenta/20';
      case 'high': return 'bg-status-warning-bg text-status-warning border border-status-warning/40';
      case 'medium': return 'bg-status-ok-bg text-indosat-teal';
      case 'low': default: return 'bg-surface-hover text-text-secondary';
    }
  };

  const getStatusBadgeColor = (stat: string) => {
    switch (stat) {
      case 'active': return 'bg-status-error-bg text-indosat-magenta';
      case 'investigating': return 'bg-status-warning-bg text-status-warning';
      case 'resolved': default: return 'bg-status-ok-bg text-indosat-teal';
    }
  };

  const getTimelineDotColor = (dot: string) => {
    switch (dot) {
      case 'red': return 'bg-indosat-magenta shadow-[0_0_6px_var(--color-indosat-magenta)]';
      case 'amber': return 'bg-status-warning';
      case 'green': return 'bg-indosat-teal shadow-[0_0_6px_var(--color-indosat-teal)]';
      case 'blue': default: return 'bg-indosat-teal';
    }
  };

  return (
    <div className="space-y-5 animate-fade select-none">
      {/* Filters & Trigger */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex gap-2">
          {(['all', 'active', 'investigating', 'resolved'] as const).map((opt) => (
            <button
              key={opt}
              onClick={() => setFilter(opt)}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-all cursor-pointer capitalize ${
                filter === opt
                  ? 'bg-indosat-magenta border-indosat-magenta text-white shadow-sm shadow-status-error/15'
                  : 'bg-surface-card border-border-subtle text-text-secondary hover:bg-surface-hover hover:text-text-primary'
              }`}
            >
              {opt}
            </button>
          ))}
        </div>
        <button
          onClick={() => setIsCreateOpen(true)}
          className="px-4 py-2 bg-indosat-teal text-white hover:bg-indosat-teal/90 rounded-lg text-xs font-bold flex items-center gap-2 cursor-pointer hover:-translate-y-[1px] shadow-sm transition-all"
        >
          <Plus className="w-4 h-4" />
          New Incident
        </button>
      </div>

      {/* Summary Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 select-none">
        <div className="glass-panel p-4.5 bg-surface-card">
          <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Active</span>
          <span className="text-2xl font-extrabold text-indosat-magenta">{activeCount}</span>
          <span className="text-[11px] font-semibold text-text-tertiary block mt-1">require SRE triage</span>
        </div>
        <div className="glass-panel p-4.5 bg-surface-card">
          <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Investigating</span>
          <span className="text-2xl font-extrabold text-status-warning">{investigatingCount}</span>
          <span className="text-[11px] font-semibold text-text-tertiary block mt-1">active taskforce</span>
        </div>
        <div className="glass-panel p-4.5 bg-surface-card">
          <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Resolved Today</span>
          <span className="text-2xl font-extrabold text-indosat-teal">{resolvedCount}</span>
          <span className="text-[11px] font-semibold text-text-tertiary block mt-1">closed cases</span>
        </div>
        <div className="glass-panel p-4.5 bg-surface-card">
          <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider block">Avg MTTR</span>
          <span className="text-2xl font-extrabold text-text-primary">42m</span>
          <span className="text-[11px] font-semibold text-text-tertiary block mt-1">last 7 days SLA</span>
        </div>
      </div>

      {/* Registry Incidents Panel */}
      <div className="glass-panel">
        <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between bg-surface-hover/20 select-none">
          <span className="font-bold text-sm text-text-primary">All Registered Incidents</span>
          <span className="text-xs font-bold text-text-tertiary">
            Showing {filteredIncidents.length} incident{filteredIncidents.length !== 1 ? 's' : ''}
          </span>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
            <thead className="bg-surface-hover/30 text-text-tertiary uppercase select-none">
              <tr>
                <th className="px-4 py-2.5 text-left tracking-wider">ID</th>
                <th className="px-4 py-2.5 text-left tracking-wider">Incident Title</th>
                <th className="px-4 py-2.5 text-left tracking-wider">Service Scope</th>
                <th className="px-4 py-2.5 text-left tracking-wider">Severity</th>
                <th className="px-4 py-2.5 text-left tracking-wider">Status</th>
                <th className="px-4 py-2.5 text-left tracking-wider">Opened</th>
                <th className="px-4 py-2.5 text-left tracking-wider">Assignee</th>
                <th className="px-4 py-2.5 text-right tracking-wider">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
              {filteredIncidents.map((inc) => (
                <tr
                  key={inc.id}
                  onClick={() => setSelectedInc(inc)}
                  className="hover:bg-surface-hover/40 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3.5 font-mono text-text-tertiary">{inc.id}</td>
                  <td className="px-4 py-3.5 font-bold text-text-primary text-[12.5px] truncate max-w-[220px]">
                    {inc.title}
                  </td>
                  <td className="px-4 py-3.5">Scope: {inc.service}</td>
                  <td className="px-4 py-3.5 select-none">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase ${getSeverityBadgeColor(inc.severity)}`}>
                      {inc.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 select-none">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase ${getStatusBadgeColor(inc.status)}`}>
                      {inc.status}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-text-tertiary whitespace-nowrap font-mono">{inc.opened}</td>
                  <td className={`px-4 py-3.5 ${inc.assignee === 'Unassigned' ? 'text-text-tertiary' : 'text-text-secondary'}`}>
                    {inc.assignee}
                  </td>
                  <td className="px-4 py-3.5 text-right select-none" onClick={(e) => e.stopPropagation()}>
                    {inc.status !== 'resolved' ? (
                      <button
                        onClick={() => handleResolve(inc.id)}
                        className="text-indosat-teal hover:text-indosat-teal/80 font-extrabold underline underline-offset-4 cursor-pointer text-xs"
                      >
                        Resolve
                      </button>
                    ) : (
                      <span className="text-text-tertiary">Closed</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* MODAL 1: CREATE NEW INCIDENT */}
      {isCreateOpen && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[200] flex items-center justify-center p-4">
          <div className="bg-surface-card border border-border-subtle rounded-2xl shadow-xl w-full max-w-[500px] overflow-hidden animate-fade">
            {/* Modal Header */}
            <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between">
              <span className="font-extrabold text-sm text-text-primary flex items-center gap-1.5">
                <ShieldAlert className="w-5 h-5 text-indosat-magenta" />
                Report New Operational Incident
              </span>
              <button
                onClick={() => setIsCreateOpen(false)}
                className="w-7 h-7 rounded-full bg-surface-hover hover:bg-border-medium text-text-secondary hover:text-text-primary flex items-center justify-center cursor-pointer transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 space-y-4 text-xs font-semibold">
              <div className="flex flex-col gap-1.5">
                <label className="text-text-secondary">Incident Title</label>
                <input
                  type="text"
                  value={formTitle}
                  onChange={(e) => {
                    setFormTitle(e.target.value);
                    if (e.target.value) setFormTitleError(false);
                  }}
                  placeholder="e.g. Stripe checkout gateway rejecting API requests"
                  className={`px-3 py-2 border rounded-lg bg-bg-main outline-none focus:border-indosat-teal transition-all text-text-primary ${
                    formTitleError ? 'border-indosat-magenta' : 'border-border-medium'
                  }`}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-text-secondary">Service Scope</label>
                  <select
                    value={formService}
                    onChange={(e) => setFormService(e.target.value)}
                    className="px-3 py-2 border border-border-medium rounded-lg bg-bg-main outline-none focus:border-indosat-teal text-text-secondary"
                  >
                    <option value="order-service">order-service</option>
                    <option value="inventory-service">inventory-service</option>
                    <option value="payment-service">payment-service</option>
                    <option value="node / infrastructure">node / infrastructure Scope</option>
                    <option value="multiple services">multiple services</option>
                  </select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-text-secondary">Severity Code</label>
                  <select
                    value={formSeverity}
                    onChange={(e) => setFormSeverity(e.target.value as any)}
                    className="px-3 py-2 border border-border-medium rounded-lg bg-bg-main outline-none focus:border-indosat-teal text-text-secondary"
                  >
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-text-secondary">Assignee Operator</label>
                <input
                  type="text"
                  value={formAssignee}
                  onChange={(e) => setFormAssignee(e.target.value)}
                  placeholder="e.g. Siti N."
                  className="px-3 py-2 border border-border-medium rounded-lg bg-bg-main outline-none focus:border-indosat-teal text-text-primary"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-text-secondary">Brief Description</label>
                <textarea
                  value={formDesc}
                  onChange={(e) => setFormDesc(e.target.value)}
                  rows={3}
                  placeholder="Identify telemetry findings, metric anomalies, or error payloads..."
                  className="px-3 py-2 border border-border-medium rounded-lg bg-bg-main outline-none focus:border-indosat-teal text-text-primary resize-none font-medium"
                />
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-5 py-4 border-t border-border-subtle bg-surface-hover/20 flex justify-end gap-3 select-none">
              <button
                onClick={() => setIsCreateOpen(false)}
                className="px-4 py-2 border border-border-medium rounded-lg hover:bg-surface-hover text-text-secondary cursor-pointer font-bold"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateIncident}
                className="px-4 py-2 bg-indosat-teal text-white hover:bg-indosat-teal/90 rounded-lg cursor-pointer font-bold"
              >
                Create Incident
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 2: VIEW INCIDENT DETAIL & TIMELINE */}
      {selectedInc && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[200] flex items-center justify-center p-4">
          <div className="bg-surface-card border border-border-subtle rounded-2xl shadow-xl w-full max-w-[580px] overflow-hidden animate-fade">
            {/* Modal Header */}
            <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-xs font-bold text-text-tertiary bg-bg-main px-2 py-0.5 border border-border-subtle rounded">
                  {selectedInc.id}
                </span>
                <span className="font-extrabold text-sm text-text-primary leading-tight max-w-[340px] truncate" title={selectedInc.title}>
                  {selectedInc.title}
                </span>
              </div>
              <button
                onClick={() => setSelectedInc(null)}
                className="w-7 h-7 rounded-full bg-surface-hover hover:bg-border-medium text-text-secondary hover:text-text-primary flex items-center justify-center cursor-pointer transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 space-y-4 max-h-[65vh] overflow-y-auto text-xs font-semibold select-text">
              {/* Grid Meta Details */}
              <div className="grid grid-cols-2 gap-x-4 gap-y-3 pb-3 border-b border-border-subtle">
                <div className="flex flex-col gap-0.5">
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider">Service Scope</span>
                  <span className="text-text-secondary text-xs">{selectedInc.service}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider">Assignee Operator</span>
                  <span className="text-text-secondary text-xs flex items-center gap-1">
                    <User className="w-3.5 h-3.5 text-text-tertiary" />
                    {selectedInc.assignee}
                  </span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider">Severity Code</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase w-fit select-none ${getSeverityBadgeColor(selectedInc.severity)}`}>
                    {selectedInc.severity}
                  </span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider">Status Scope</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase w-fit select-none ${getStatusBadgeColor(selectedInc.status)}`}>
                    {selectedInc.status}
                  </span>
                </div>
                <div className="flex flex-col gap-0.5 col-span-2">
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider">Opened Lifecycle</span>
                  <span className="text-text-secondary font-mono text-xs flex items-center gap-1 select-none">
                    <Clock className="w-3.5 h-3.5 text-text-tertiary" />
                    {selectedInc.opened}
                  </span>
                </div>
              </div>

              {/* Description Panel */}
              <div className="flex flex-col gap-1 bg-surface-hover/35 border border-border-subtle p-3.5 rounded-xl">
                <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider flex items-center gap-1 select-none">
                  <FileText className="w-3.5 h-3.5 text-text-tertiary" />
                  Incident Log Description
                </span>
                <span className="text-text-secondary text-xs font-medium leading-relaxed">
                  {selectedInc.desc || 'No description logged.'}
                </span>
              </div>

              {/* Activity Timeline */}
              <div className="flex flex-col gap-2.5">
                <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider select-none">
                  SRE Activity Audit Log
                </span>
                
                {selectedInc.activity && selectedInc.activity.length > 0 ? (
                  <div className="relative border-l border-border-medium pl-4 ml-2.5 py-1.5 space-y-4">
                    {selectedInc.activity.map((act, idx) => (
                      <div key={idx} className="relative flex items-start gap-3">
                        {/* Timeline Node Dot */}
                        <span className={`absolute -left-[20.5px] top-1 w-2.5 h-2.5 rounded-full select-none ${getTimelineDotColor(act.dot)}`} />
                        <span className="text-[10px] font-extrabold text-text-tertiary font-mono pt-0.5 select-none">{act.time}</span>
                        <span className="text-text-secondary font-medium leading-relaxed">{act.text}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className="text-text-tertiary text-xs italic select-none">No timeline audit events logged yet.</span>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-5 py-4 border-t border-border-subtle bg-surface-hover/20 flex justify-end gap-3 select-none">
              <button
                onClick={() => setSelectedInc(null)}
                className="px-4 py-2 border border-border-medium rounded-lg hover:bg-surface-hover text-text-secondary cursor-pointer font-bold"
              >
                Close Window
              </button>
              {selectedInc.status !== 'resolved' && (
                <button
                  onClick={() => {
                    handleResolve(selectedInc.id);
                  }}
                  className="px-4 py-2 bg-indosat-teal hover:bg-indosat-teal/90 text-white rounded-lg cursor-pointer font-bold flex items-center gap-1.5"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  Mark Resolved
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
