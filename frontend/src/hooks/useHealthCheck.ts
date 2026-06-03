import { useState, useEffect } from 'react';

export type HealthState = 'checking' | 'ok' | 'error' | 'unreachable';

export function useHealthCheck(apiBase: string, intervalMs: number = 30000) {
  const [health, setHealth] = useState<HealthState>('checking');

  useEffect(() => {
    let active = true;
    
    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), 4000);
        
        const res = await fetch(`${apiBase}/health`, { signal: controller.signal });
        clearTimeout(id);
        
        if (!active) return;

        if (res.ok) {
          const data = await res.json();
          if (data.clickhouse === 'ok') {
            setHealth('ok');
          } else {
            setHealth('error');
          }
        } else {
          setHealth('error');
        }
      } catch (err) {
        if (!active) return;
        setHealth('unreachable');
      }
    };

    checkHealth();
    const timer = setInterval(checkHealth, intervalMs);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [apiBase, intervalMs]);

  return health;
}
