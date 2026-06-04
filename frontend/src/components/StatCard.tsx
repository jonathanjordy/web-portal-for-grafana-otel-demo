import React from 'react';

export type StatCardColor = 'default' | 'teal' | 'yellow' | 'magenta' | 'green' | 'amber' | 'red' | 'blue';

interface StatCardProps {
  label: string;
  value: string | number;
  subText: string | React.ReactNode;
  color?: StatCardColor;
}

export default function StatCard({ label, value, subText, color = 'default' }: StatCardProps) {
  const getColorClass = () => {
    switch (color) {
      case 'teal':
      case 'green':
      case 'blue':
        return 'green'; // both map to var(--green)
      case 'magenta':
      case 'red':
        return 'red'; // maps to var(--red)
      case 'yellow':
      case 'amber':
        return 'amber'; // maps to var(--amber)
      case 'default':
      default:
        return '';
    }
  };

  const valClass = getColorClass();

  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${valClass}`}>{value}</div>
      <div className="stat-sub">{subText}</div>
    </div>
  );
}
