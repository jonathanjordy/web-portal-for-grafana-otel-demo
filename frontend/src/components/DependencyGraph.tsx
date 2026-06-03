'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Info, Network, AlertCircle, ShieldAlert } from 'lucide-react';
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
    <div className="glass-panel mb-5">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex flex-wrap items-center justify-between gap-4 bg-surface-hover/20 select-none">
        <div>
          <span className="font-bold text-sm text-text-primary block">Service Dependency Graph</span>
          <span className="text-[11px] font-semibold text-text-tertiary">
            Constructed from distributed trace parenting topologies — pinpointing root cause propagation origins
          </span>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            className="text-xs px-2.5 py-1.5 border border-border-medium rounded-md font-semibold bg-bg-main text-text-secondary outline-none focus:border-indosat-teal transition-all"
          >
            <option value="1">Last 1h</option>
            <option value="3">Last 3h</option>
            <option value="6">Last 6h</option>
          </select>
          <button
            onClick={fetchGraph}
            disabled={loading}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 hover:-translate-y-[1px] transition-all cursor-pointer shadow-sm"
          >
            {loading ? 'Building map...' : 'Build graph'}
          </button>
          <button
            onClick={() => onShowInfo('graph')}
            className="w-6 h-6 rounded-full border border-border-medium bg-surface-card hover:bg-indosat-teal hover:border-indosat-teal hover:text-white flex items-center justify-center cursor-pointer transition-all duration-150"
            title="How this works"
          >
            <Info className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-5">
        {errorText ? (
          <div className="text-center text-status-error font-semibold text-sm py-4">{errorText}</div>
        ) : loading ? (
          <div className="flex flex-col items-center justify-center py-12 select-none">
            <div className="w-8 h-8 border-4 border-indosat-teal border-t-transparent rounded-full animate-spin mb-3" />
            <div className="text-xs font-bold text-text-secondary">
              Querying distributed trace graphs & assembling causal dependency matrices...
            </div>
          </div>
        ) : data ? (
          <div className="animate-fade">
            {/* SVG Visualizer */}
            <div ref={containerRef} className="w-full bg-surface-hover/30 rounded-xl mb-5 overflow-hidden border border-border-subtle relative h-[260px]">
              <svg width="100%" height={dimensions.height} viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} xmlns="http://www.w3.org/2000/svg" className="absolute top-0 left-0">
                {/* Defs for arrow markers */}
                <defs>
                  <marker
                    id="arr"
                    viewBox="0 0 10 10"
                    refX="22" // Offset slightly to stop on edge of circle
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto-start-reverse"
                  >
                    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" strokeWidth="2" />
                  </marker>
                </defs>

                {/* Draw Curves / Edges */}
                {data.edges.map((edge, idx) => {
                  const s = nodePos[edge.source];
                  const t = nodePos[edge.target];
                  if (!s || !t) return null;
                  
                  const strokeColor = getEdgeColor(edge.error_rate);
                  const mx = (s.x + t.x) / 2;
                  const my = (s.y + t.y) / 2 - 35; // Curve height

                  return (
                    <g key={`edge-${idx}`}>
                      <path
                        d={`M${s.x},${s.y} Q${mx},${my} ${t.x},${t.y}`}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth="2"
                        strokeOpacity="0.75"
                        markerEnd="url(#arr)"
                      />
                      <text
                        x={mx}
                        y={my - 8}
                        textAnchor="middle"
                        fontSize="10"
                        fontWeight="bold"
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
                  const radius = isRoot ? 34 : 26;

                  return (
                    <g key={node.id} className="select-none pointer-events-none">
                      {/* Outer concentric pulsing ring for root cause */}
                      {isRoot && (
                        <>
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={40}
                            fill="none"
                            stroke={color}
                            strokeWidth="1.5"
                            strokeDasharray="4 3"
                            className="animate-pulse"
                            opacity="0.6"
                          />
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={46}
                            fill="none"
                            stroke={color}
                            strokeWidth="0.75"
                            strokeDasharray="2 4"
                            opacity="0.3"
                          />
                        </>
                      )}

                      {/* Main Circle */}
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={radius}
                        fill={`${color}12`}
                        stroke={color}
                        strokeWidth={isRoot ? 3 : 1.5}
                      />

                      {/* Text details */}
                      <text
                        x={p.x}
                        y={p.y - 3}
                        textAnchor="middle"
                        fontSize="10.5"
                        fontWeight="800"
                        fill={color}
                        fontFamily="Nunito"
                      >
                        {node.id.replace('-service', '')}
                      </text>
                      <text
                        x={p.x}
                        y={p.y + 10}
                        textAnchor="middle"
                        fontSize="9.5"
                        fontWeight="bold"
                        fill={color}
                        fontFamily="Nunito"
                      >
                        {node.error_rate}% err
                      </text>

                      {/* Root cause indicator badge */}
                      {isRoot && (
                        <text
                          x={p.x}
                          y={p.y + 55}
                          textAnchor="middle"
                          fontSize="9.5"
                          fontWeight="800"
                          fill={color}
                          fontFamily="Nunito"
                          className="uppercase tracking-widest animate-pulse"
                        >
                          ⚠ Root Cause
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Nodes metrics table */}
            <div className="overflow-x-auto border border-border-subtle rounded-lg">
              <table className="min-w-full divide-y divide-border-subtle text-xs font-semibold">
                <thead className="bg-surface-hover/30 text-text-tertiary select-none">
                  <tr>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Service Node</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Calls</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Errors</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Error Rate</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">Avg Latency</th>
                    <th className="px-4 py-2.5 text-left uppercase tracking-wider">System Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle bg-surface-card text-text-secondary">
                  {data.nodes.map((node) => {
                    const isRoot = node.id === data.root_cause;
                    return (
                      <tr key={node.id} className="hover:bg-surface-hover/30 transition-colors">
                        <td className="px-4 py-3 font-bold text-text-primary flex items-center gap-1.5">
                          {isRoot && <ShieldAlert className="w-4 h-4 text-indosat-magenta animate-pulse" />}
                          {node.id}
                        </td>
                        <td className="px-4 py-3 select-none">{node.calls}</td>
                        <td className="px-4 py-3 select-none">{node.errors}</td>
                        <td className="px-4 py-3 select-none">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${node.error_rate > 10 ? 'bg-status-error-bg text-indosat-magenta' : node.error_rate > 2 ? 'bg-status-warning-bg text-status-warning' : 'bg-status-ok-bg text-indosat-teal'}`}>
                            {node.error_rate}%
                          </span>
                        </td>
                        <td className="px-4 py-3 select-none">{node.avg_duration_ms.toFixed(1)}ms</td>
                        <td className="px-4 py-3 flex items-center gap-1.5 select-none">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${node.status === 'error' ? 'bg-status-error-bg text-indosat-magenta' : node.status === 'degraded' ? 'bg-status-warning-bg text-status-warning' : 'bg-status-ok-bg text-indosat-teal'}`}>
                            {node.status}
                          </span>
                          {isRoot && (
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-status-error-bg text-indosat-magenta border border-indosat-magenta/10 font-extrabold uppercase animate-pulse">
                              root cause
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-border-medium rounded-xl py-10 text-center text-text-tertiary font-semibold text-xs select-none">
            Click &quot;Build graph&quot; to compile and generate the live distributed service dependency map.
          </div>
        )}
      </div>
    </div>
  );
}
