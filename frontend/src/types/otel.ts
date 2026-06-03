// OTel Signals and Dashboard TypeScript Types

export interface HealthStatus {
  clickhouse: 'ok' | 'error';
  timestamp: string;
}

// Predictive Analytics Types
export interface ForecastDataPoint {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

export interface HistoricalDataPoint {
  ds: string;
  y: number | null;
}

export interface ForecastResponse {
  data_points: number;
  peak_predicted?: {
    value: number;
    timestamp: string;
  };
  forecast: ForecastDataPoint[];
  historical: HistoricalDataPoint[];
}

export interface PredictiveSummary {
  memory?: {
    used_pct: number;
  };
  load?: {
    load1: number;
  };
  orders?: {
    last_hour: number;
    change_pct: number;
  };
}

// Detective / Anomaly Detection Types
export interface AnomalyTimelineItem {
  ts: string;
  anomaly_score: number;
  is_anomaly: boolean;
  contributing: string[];
  metrics: Record<string, number | null>;
}

export interface AnomalyResponse {
  total_windows: number;
  anomaly_count: number;
  anomaly_rate: number;
  timeline: AnomalyTimelineItem[];
}

export interface LogPatternItem {
  template: string;
  count: number;
  pct_of_total: number;
  top_service: string;
  dominant_severity: 'INFO' | 'WARNING' | 'ERROR';
  is_new: boolean;
}

export interface LogPatternResponse {
  total_logs: number;
  unique_patterns: number;
  new_patterns: number;
  error_patterns: number;
  patterns: LogPatternItem[];
}

export interface TraceShapeItem {
  fingerprint: string;
  count: number;
  pct_of_total: number;
  is_baseline: boolean;
  deviation_type?: string;
  example_traces: string[];
}

export interface TraceShapeResponse {
  total_traces: number;
  unique_shapes: number;
  anomalous_count: number;
  baseline_pct: number;
  shape_summary: TraceShapeItem[];
}

// Diagnostic / Root Cause Analysis Types
export interface DependencyNode {
  id: string;
  calls: number;
  errors: number;
  error_rate: number;
  avg_duration_ms: number;
  status: 'ok' | 'degraded' | 'error';
}

export interface DependencyEdge {
  source: string;
  target: string;
  error_rate: number;
  avg_duration_ms: number;
}

export interface CausalGraphResponse {
  nodes: DependencyNode[];
  edges: DependencyEdge[];
  root_cause: string;
}

export interface SlowTrace {
  TraceId: string;
  ServiceName: string;
  SpanName: string;
  Duration: number;
  StatusCode: string;
}

export interface ErrorTrace {
  TraceId: string;
  ServiceName: string;
  SpanName: string;
  Timestamp: string | number;
}

export interface ErrorLog {
  TraceId?: string;
  service_name?: string;
  Body: string;
  Timestamp: string | number;
}

export interface MetricSummary {
  MetricName: string;
  total: number;
  avg_val: number;
}

export interface TelemetryCorrelationResponse {
  slow_traces: SlowTrace[];
  error_traces: ErrorTrace[];
  error_logs: ErrorLog[];
  metric_summary: MetricSummary[];
}

export interface LLMIncidentSummaryResponse {
  summary: string;
}

// Chatbot Assistant Types
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  executed?: boolean;
  row_count?: number;
}

export interface ChatResponse {
  response: string;
  executed?: boolean;
  row_count?: number;
}

// Incident Management Types
export interface ActivityLog {
  time: string;
  dot: 'red' | 'amber' | 'blue' | 'green';
  text: string;
}

export interface Incident {
  id: string;
  title: string;
  service: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'all' | 'active' | 'investigating' | 'resolved';
  opened: string;
  assignee: string;
  desc: string;
  activity: ActivityLog[];
}
