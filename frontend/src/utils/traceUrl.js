// Grafana Tempo "Explore" deep link for a single trace.
// Clicking a trace ID opens this URL in a new tab with the trace ID injected
// into the TraceQL query parameter.
const GRAFANA_BASE = 'http://35.219.90.43:3000';

export function traceExploreUrl(traceId) {
  const panes = {
    VYN: {
      datasource: 'tempo',
      queries: [
        {
          query: String(traceId),
          queryType: 'traceql',
          refId: 'A',
          datasource: { type: 'tempo', uid: 'tempo' },
          limit: 20,
          tableType: 'traces',
        },
      ],
      range: { from: 'now-30m', to: 'now' },
    },
  };

  const params = new URLSearchParams({
    schemaVersion: '1',
    panes: JSON.stringify(panes),
    orgId: '1',
  });

  return `${GRAFANA_BASE}/explore?${params.toString()}`;
}
