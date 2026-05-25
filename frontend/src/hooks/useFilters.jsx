import { useEffect, useState } from 'react';
import { apiGet } from '../services/api.js';

const fallbackServices = [
  { value: 'order-service' },
  { value: 'inventory-service' },
  { value: 'payment-service' },
];

export function useFilters() {
  const [filters, setFilters] = useState({ services: fallbackServices, hosts: [] });

  useEffect(() => {
    let cancelled = false;

    async function loadFilters() {
      try {
        const data = await apiGet('/filters');
        if (!cancelled) {
          setFilters({
            services: data.services?.length ? data.services : fallbackServices,
            hosts: data.hosts || [],
          });
        }
      } catch {
        if (!cancelled) setFilters({ services: fallbackServices, hosts: [] });
      }
    }

    loadFilters();
    return () => {
      cancelled = true;
    };
  }, []);

  return filters;
}

export function FilterSelect({ label, value, options, onChange, allLabel, disabledLabel = 'No values available' }) {
  const disabled = !options.length;

  return (
    <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-3)', fontSize: '0.75rem', fontWeight: 700 }}>
      {label}
      <select className="select-sm" value={disabled ? '' : value} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
        <option value="">{disabled ? disabledLabel : allLabel}</option>
        {options.map((option) => (
          <option value={option.value} key={option.value}>{option.value}</option>
        ))}
      </select>
    </label>
  );
}
