'use client';

import React, { useState, useEffect, useRef } from 'react';
import { CausalGraphResponse } from '../types/otel';

interface DependencyGraphProps {
  apiBase: string;
  onShowInfo: (id: string) => void;
}

export default function DependencyGraph({ apiBase, onShowInfo }: DependencyGraphProps) {
  const [hours, setHours] = useState('1');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<CausalGraphResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 260 });

  // Update SVG dimensions on resize
  useEffect(() => {
    if (!containerRef.current || !data) return;
    
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth || 700,
          height: 260
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [data]);

  const fetchGraph = async () => {
    setLoading(true);
    setErrorText(null);
    try {
      const res = await fetch(`${apiBase}/diagnostic/causal-graph?hours=${hours}`);
      const result = await res.json();
      
      if (result.detail) {
        setErrorText(result.detail);
        return;
      }
      
      setData(result);
    } catch (err: any) {
      setErrorText('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Node position dictionary mapping
  const nodePos: Record<string, { x: number; y: number }> = {};
  if (data) {
    const spacing = dimensions.width / (data.nodes.length + 1);
    data.nodes.forEach((node, i) => {
      nodePos[node.id] = {
        x: spacing * (i + 1),
        y: dimensions.height / 2
      };
    });
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'error': return '#EB008C';
      case 'degraded': return '#d4a000';
      case 'ok': default: return '#24BCAD';
    }
  };

  const getEdgeColor = (errorRate: number) => {
    if (errorRate > 10) return '#EB008C';
    if (errorRate > 2) return '#d4a000';
    return '#24BCAD';
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="panel-title">Service dependency graph</div>
          <div className="panel-meta">Built from trace parent-child relationships — highlights root cause service</div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="select-sm"
          >
            <option value="1">Last 1h</option>
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchGraph}
            disabled={loading}
            className="btn-sm"
          >
            {loading ? 'Building map...' : 'Build graph'}
          </button>
          <button
            onClick={() => onShowInfo('graph')}
            className="btn-info"
            title="How this works"
          >
            i
          </button>
        </div>
      </div>

      <div className="panel-body">
        {loading && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Building dependency graph from traces...
          </div>
        )}

        {!loading && errorText && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none', color: 'var(--red)' }}>
            {errorText}
          </div>
        )}

        {!loading && !errorText && !data && (
          <div className="empty" style={{ padding: '1.5rem', border: 'none' }}>
            Click &quot;Build graph&quot; to generate the live service dependency map.
          </div>
        )}

        {!loading && !errorText && data && (
          <div id="graph-wrap">
            <div
              id="graph-canvas"
              style={{
                background: 'var(--surface2)',
                borderRadius: '10px',
                minHeight: '260px',
                position: 'relative',
                overflow: 'hidden',
                marginBottom: '1rem'
              }}
              ref={containerRef}
            >
              <svg
                width="100%"
                height={dimensions.height}
                viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                xmlns="http://www.w3.org/2000/svg"
              >
                {/* Defs for arrow markers */}
                <defs>
                  <marker
                    id="arr"
                    viewBox="0 0 10 10"
                    refX="8"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" strokeWidth="1.5" />
                  </marker>
                </defs>

                {/* Draw Curves / Edges */}
                {data.edges.map((edge, idx) => {
                  const s = nodePos[edge.source];
                  const t = nodePos[edge.target];
                  if (!s || !t) return null;
                  
                  const strokeColor = getEdgeColor(edge.error_rate);
                  const mx = (s.x + t.x) / 2;
                  const my = (s.y + t.y) / 2 - 30; // Curve height

                  return (
                    <g key={`edge-${idx}`}>
                      <path
                        d={`M${s.x},${s.y} Q${mx},${my} ${t.x},${t.y}`}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth="2"
                        strokeOpacity="0.6"
                        markerEnd="url(#arr)"
                      />
                      <text
                        x={mx}
                        y={my - 6}
                        textAnchor="middle"
                        fontSize="10"
                        fill={strokeColor}
                        fontFamily="Nunito"
                      >
                        {edge.avg_duration_ms.toFixed(0)}ms · {edge.error_rate}% err
                      </text>
                    </g>
                  );
                })}

                {/* Draw Node Circles */}
                {data.nodes.map((node) => {
                  const p = nodePos[node.id];
                  if (!p) return null;

                  const color = getStatusColor(node.status);
                  const isRoot = node.id === data.root_cause;
                  const radius = isRoot ? 36 : 28;

                  return (
                    <g key={node.id} className="select-none pointer-events-none">
                      {/* Outer concentric pulsing ring for root cause */}
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={radius}
                        fill={`${color}22`}
                        stroke={color}
                        strokeWidth={isRoot ? 3 : 1.5}
                      />
                      {isRoot && (
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={42}
                          fill="none"
                          stroke={color}
                          strokeWidth="1"
                          strokeDasharray="4 3"
                          opacity="0.5"
                        />
                      )}

                      {/* Text details */}
                      <text
                        x={p.x}
                        y={p.y - 4}
                        textAnchor="middle"
                        fontSize="11"
                        fontWeight="700"
                        fill={color}
                        fontFamily="Nunito"
                      >
                        {node.id.replace('-service', '')}
                      </text>
                      <text
                        x={p.x}
                        y={p.y + 11}
                        textAnchor="middle"
                        fontSize="10"
                        fill={color}
                        fontFamily="Nunito"
                      >
                        {node.error_rate}% err
                      </text>

                      {/* Root cause indicator badge */}
                      {isRoot && (
                        <text
                          x={p.x}
                          y={p.y + 58}
                          textAnchor="middle"
                          fontSize="10"
                          fontWeight="700"
                          fill={color}
                          fontFamily="Nunito"
                        >
                          ⚠ root cause
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Nodes metrics table */}
            <div id="graph-nodes-wrap" style={{ overflowX: 'auto' }}>
              <table className="det-table" style={{ marginTop: '0.5rem' }}>
                <thead>
                  <tr>
                    <th>Service</th>
                    <th>Calls</th>
                    <th>Errors</th>
                    <th>Error rate</th>
                    <th>Avg latency</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.nodes.map((node) => {
                    const isRoot = node.id === data.root_cause;
                    return (
                      <tr key={node.id}>
                        <td style={{ fontWeight: 700 }}>{node.id}</td>
                        <td>{node.calls}</td>
                        <td>{node.errors}</td>
                        <td>
                          <span className={`tag ${node.error_rate > 10 ? 'red' : node.error_rate > 2 ? 'amber' : 'green'}`}>
                            {node.error_rate}%
                          </span>
                        </td>
                        <td>{node.avg_duration_ms.toFixed(0)}ms</td>
                        <td>
                          <span className={`tag ${node.status === 'error' ? 'red' : node.status === 'degraded' ? 'amber' : 'green'}`}>
                            {node.status}
                          </span>
                          {isRoot && (
                            <span className="tag red">root cause</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
