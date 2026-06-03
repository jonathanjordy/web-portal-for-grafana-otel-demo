import React from 'react';
import { 
  TrendingUp, 
  Search, 
  Activity, 
  MessageSquareCode, 
  ShieldAlert 
} from 'lucide-react';
import { HealthState } from '../hooks/useHealthCheck';

interface SidebarProps {
  currentPage: string;
  onPageChange: (page: string) => void;
  health: HealthState;
}

export default function Sidebar({ currentPage, onPageChange, health }: SidebarProps) {
  const navItems = [
    { id: 'predictive', label: 'Predictive', group: 'Analytics', icon: TrendingUp },
    { id: 'detective', label: 'Detective', group: 'Analytics', icon: Search },
    { id: 'diagnostic', label: 'Diagnostic', group: 'Analytics', icon: Activity },
    { id: 'chatbot', label: 'AIOps Chat', group: 'Assistant', icon: MessageSquareCode },
    { id: 'incidents', label: 'Incident Registry', group: 'Operations', icon: ShieldAlert },
  ];

  // Group items
  const groups = Array.from(new Set(navItems.map(item => item.group)));

  const getHealthText = () => {
    switch (health) {
      case 'ok': return 'ClickHouse connected';
      case 'error': return 'ClickHouse error';
      case 'unreachable': return 'API unreachable';
      case 'checking': default: return 'Checking...';
    }
  };

  const getHealthDotColor = () => {
    switch (health) {
      case 'ok': return 'bg-status-ok shadow-[0_0_8px_var(--color-indosat-teal)]';
      case 'error': return 'bg-status-error shadow-[0_0_8px_var(--color-indosat-magenta)]';
      case 'unreachable': return 'bg-status-error shadow-[0_0_8px_var(--color-indosat-magenta)]';
      case 'checking': default: return 'bg-text-tertiary animate-pulse';
    }
  };

  return (
    <aside className="w-[260px] bg-surface-card border-r border-border-subtle flex flex-col h-full flex-shrink-0">
      {/* Brand Wordmark */}
      <div className="p-7 border-b border-border-subtle">
        <div className="font-sans text-2xl font-extrabold text-indosat-magenta tracking-tight leading-none select-none">
          AIOps Portal
          <span className="block text-xs font-semibold text-text-tertiary tracking-wider mt-1.5 uppercase">
            Observability Intelligence
          </span>
        </div>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 overflow-y-auto py-4">
        {groups.map(group => (
          <div key={group} className="mb-6 px-4">
            <div className="text-[10px] font-bold text-text-tertiary tracking-widest uppercase px-3 mb-2">
              {group}
            </div>
            <nav className="space-y-1">
              {navItems
                .filter(item => item.group === group)
                .map(item => {
                  const Icon = item.icon;
                  const isActive = currentPage === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => onPageChange(item.id)}
                      className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg font-semibold text-sm transition-all duration-150 text-left cursor-pointer select-none ${
                        isActive
                          ? 'bg-indosat-magenta text-white shadow-md shadow-status-error/15'
                          : 'text-text-secondary hover:bg-surface-hover hover:text-text-primary'
                      }`}
                    >
                      <Icon className={`w-4.5 h-4.5 flex-shrink-0 transition-opacity ${isActive ? 'opacity-100' : 'opacity-70'}`} />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
            </nav>
          </div>
        ))}
      </div>

      {/* Health Status Indicator */}
      <div className="p-5 border-t border-border-subtle bg-surface-hover/30">
        <div className="flex items-center gap-2.5 font-semibold text-xs text-text-secondary select-none">
          <span className={`w-2.5 h-2.5 rounded-full ${getHealthDotColor()}`} />
          <span>{getHealthText()}</span>
        </div>
      </div>
    </aside>
  );
}
