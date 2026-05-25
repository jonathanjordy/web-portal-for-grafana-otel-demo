export const API_BASE = (() => {
  if (import.meta.env.VITE_API_BASE) {
    return import.meta.env.VITE_API_BASE.replace(/\/$/, '');
  }

  if (window.API_BASE) {
    return String(window.API_BASE).replace(/\/$/, '');
  }

  const params = new URLSearchParams(window.location.search);
  const override = params.get('api');
  if (override) return override.replace(/\/$/, '');

  const host = window.location.hostname || 'localhost';
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  return `${protocol}//${host}:8080/api`;
})();

export async function apiGet(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  return res.json();
}

export async function apiPost(path, body, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    ...options,
  });
  return res.json();
}
