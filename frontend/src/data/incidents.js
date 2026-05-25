export const DUMMY_INCIDENTS = [
  { id: 'INC-001', title: 'Payment gateway rejection rate elevated', service: 'payment-service', severity: 'critical', status: 'active', opened: '2026-04-29 08:14', assignee: 'Ahmad R.', desc: 'Payment failures spiked to 45% at 08:12. Gateway returning 402 consistently.' },
  { id: 'INC-002', title: 'Inventory DB slow query detected', service: 'inventory-service', severity: 'high', status: 'investigating', opened: '2026-04-29 07:55', assignee: 'Siti N.', desc: 'db-stock-lookup span averaging 2.1s. Suspected missing index on stock table.' },
  { id: 'INC-003', title: 'Memory usage trending upward on node', service: 'node / infrastructure', severity: 'medium', status: 'investigating', opened: '2026-04-29 06:30', assignee: 'Budi S.', desc: 'Prophet forecast predicts memory saturation in ~18 hours if trend continues.' },
  { id: 'INC-004', title: 'Order service response time p99 > 3s', service: 'order-service', severity: 'high', status: 'active', opened: '2026-04-29 09:01', assignee: 'Unassigned', desc: 'p99 latency crossed 3s threshold. Correlated with INC-002 slow inventory lookup.' },
  { id: 'INC-005', title: 'Redis cache hit rate dropped below 50%', service: 'inventory-service', severity: 'medium', status: 'active', opened: '2026-04-29 09:15', assignee: 'Siti N.', desc: 'Cache hit rate fell from 85% to 43%. Possible cache eviction or TTL issue.' },
  { id: 'INC-006', title: 'Intermittent connection timeout to ClickHouse', service: 'multiple services', severity: 'low', status: 'resolved', opened: '2026-04-28 22:10', assignee: 'Ahmad R.', desc: 'OTel collector reported retry errors. Resolved after ClickHouse restart.' },
  { id: 'INC-007', title: 'Deployment caused 2 min downtime on order-service', service: 'order-service', severity: 'high', status: 'resolved', opened: '2026-04-28 18:00', assignee: 'Budi S.', desc: 'Rolling restart during peak hours caused brief unavailability. Post-mortem filed.' },
  { id: 'INC-008', title: 'Node CPU load spike during batch job', service: 'node / infrastructure', severity: 'medium', status: 'resolved', opened: '2026-04-28 14:22', assignee: 'Siti N.', desc: 'Load average hit 4.2 during scheduled analytics job. Resolved after job completed.' },
  { id: 'INC-009', title: 'Payment amount histogram showing outliers > $5000', service: 'payment-service', severity: 'low', status: 'resolved', opened: '2026-04-28 11:05', assignee: 'Ahmad R.', desc: 'Unusually large payment amounts detected. Confirmed as load test data, not real.' },
];

const ACTIVITY_SEEDS = {
  'INC-001': [{ time: '08:12', dot: 'red', text: 'Anomaly detected - failure rate 45%' }, { time: '08:14', dot: 'red', text: 'Incident raised automatically' }, { time: '08:17', dot: 'amber', text: 'Ahmad R. assigned and started investigation' }, { time: '08:22', dot: 'blue', text: 'Checked payment-service logs - gateway returning 402 consistently' }],
  'INC-002': [{ time: '07:50', dot: 'amber', text: 'Slow span alert fired - db-stock-lookup > 2s' }, { time: '07:55', dot: 'red', text: 'Incident raised' }, { time: '08:01', dot: 'amber', text: 'Siti N. assigned' }, { time: '08:10', dot: 'blue', text: 'Reviewing EXPLAIN output on stock table query' }],
  'INC-003': [{ time: '06:28', dot: 'amber', text: 'Prophet forecast predicted saturation in 18h' }, { time: '06:30', dot: 'red', text: 'Incident raised' }, { time: '06:45', dot: 'amber', text: 'Budi S. assigned - monitoring trend' }],
  'INC-004': [{ time: '09:01', dot: 'red', text: 'p99 latency crossed 3s threshold' }, { time: '09:03', dot: 'blue', text: 'Correlated with INC-002 inventory slow query' }],
  'INC-005': [{ time: '09:12', dot: 'amber', text: 'Cache hit rate monitoring alert fired' }, { time: '09:15', dot: 'red', text: 'Incident raised' }, { time: '09:18', dot: 'amber', text: 'Siti N. assigned - checking Redis TTL config' }],
  'INC-006': [{ time: '22:10', dot: 'red', text: 'OTel collector retry errors detected' }, { time: '22:15', dot: 'amber', text: 'Ahmad R. investigated - ClickHouse connection dropping' }, { time: '22:31', dot: 'blue', text: 'ClickHouse restarted' }, { time: '22:33', dot: 'green', text: 'Connections restored - incident resolved' }],
  'INC-007': [{ time: '18:00', dot: 'red', text: 'Deployment started on order-service' }, { time: '18:02', dot: 'red', text: 'Health check failures detected' }, { time: '18:04', dot: 'amber', text: 'Budi S. monitoring - all pods restarting' }, { time: '18:06', dot: 'green', text: 'All pods healthy - service restored' }, { time: '18:10', dot: 'green', text: 'Post-mortem filed - resolved' }],
  'INC-008': [{ time: '14:20', dot: 'amber', text: 'CPU load average spike - load1 = 4.2' }, { time: '14:22', dot: 'red', text: 'Incident raised' }, { time: '14:25', dot: 'blue', text: 'Siti N. identified scheduled analytics batch job' }, { time: '14:55', dot: 'green', text: 'Batch job completed - load normalised - resolved' }],
  'INC-009': [{ time: '11:05', dot: 'amber', text: 'Histogram outliers detected - payments > $5000' }, { time: '11:08', dot: 'blue', text: 'Ahmad R. investigated - confirmed load test data' }, { time: '11:15', dot: 'green', text: 'No real impact - resolved' }],
};

export function buildIncidentSeed() {
  return DUMMY_INCIDENTS.map((incident) => ({
    ...incident,
    activity: ACTIVITY_SEEDS[incident.id] || [],
  }));
}
