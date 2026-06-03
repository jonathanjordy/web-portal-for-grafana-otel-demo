import React from 'react';

export type StatCardColor = 'default' | 'teal' | 'yellow' | 'magenta' | 'green' | 'amber' | 'red';

interface StatCardProps {
  label: string;
  value: string | number;
  subText: string;
  color?: StatCardColor;
}

export default function StatCard({ label, value, subText, color = 'default' }: StatCardProps) {
  const getColorClass = () => {
    switch (color) {
      case 'teal':
      case 'green':
        return 'text-indosat-teal';
      case 'magenta':
      case 'red':
        return 'text-indosat-magenta';
      case 'yellow':
      case 'amber':
        return 'text-status-warning';
      case 'default':
      default:
        return 'text-text-primary';
    }
  };

  return (
    <div className="glass-panel p-5 hover:border-border-medium hover:shadow-md transition-all duration-150 flex flex-col justify-between select-none">
      <div>
        <div className="text-[10px] font-bold text-text-tertiary tracking-wider uppercase mb-1">
          {label}
        </div>
        <div className={`font-sans text-3xl font-extrabold tracking-tight ${getColorClass()}`}>
          {value}
        </div>
      </div>
      <div className="text-xs font-semibold text-text-tertiary mt-2">
        {subText}
      </div>
    </div>
  );
}
