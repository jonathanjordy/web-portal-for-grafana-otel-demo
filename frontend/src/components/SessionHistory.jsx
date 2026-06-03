import { useState } from 'react';

const GROUP_ORDER = ['Today', 'Yesterday', 'This Week', 'Older'];

function startOfDay(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

function groupFor(epochSeconds) {
  const then = new Date(epochSeconds * 1000);
  const today = startOfDay(new Date());
  const dayDiff = Math.round((today - startOfDay(then)) / 86400000);
  if (dayDiff <= 0) return 'Today';
  if (dayDiff === 1) return 'Yesterday';
  if (dayDiff < 7) return 'This Week';
  return 'Older';
}

function relativeTime(epochSeconds) {
  const diff = Math.floor(Date.now() / 1000 - epochSeconds);
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (groupFor(epochSeconds) === 'Today') return `${Math.floor(diff / 3600)} hr ago`;
  const d = new Date(epochSeconds * 1000);
  const date = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  if (groupFor(epochSeconds) === 'Older') return date;
  const time = d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  return `${date}, ${time}`;
}

export default function SessionHistory({ sessions, activeSessionId, onNewChat, onSelect, onRename, onDelete }) {
  const [editingId, setEditingId] = useState(null);
  const [draft, setDraft] = useState('');

  function beginEdit(session, event) {
    event.stopPropagation();
    setEditingId(session.id);
    setDraft(session.title);
  }

  function commitEdit(id) {
    const title = draft.trim();
    setEditingId(null);
    if (title) onRename(id, title);
  }

  function confirmDelete(session, event) {
    event.stopPropagation();
    if (window.confirm(`Delete session "${session.title}"?`)) onDelete(session.id);
  }

  // Bucket sessions into time groups, preserving the newest-first order they arrive in.
  const grouped = {};
  for (const session of sessions) {
    const group = groupFor(session.updated_at);
    (grouped[group] ||= []).push(session);
  }

  return (
    <aside className="session-panel">
      <button className="new-chat-btn" type="button" onClick={onNewChat}>
        <span className="new-chat-plus">+</span> New Chat
      </button>

      <div className="session-scroll">
        <div className="session-heading">Session History</div>

        {sessions.length === 0 ? (
          <div className="session-empty">No sessions yet.</div>
        ) : (
          GROUP_ORDER.filter((g) => grouped[g]).map((group) => (
            <div className="session-group" key={group}>
              <div className="session-group-label">{group}</div>
              {grouped[group].map((session) => (
                <div
                  className={`session-row ${session.id === activeSessionId ? 'active' : ''}`}
                  key={session.id}
                  onClick={() => onSelect(session.id)}
                >
                  <div className="session-row-main">
                    {editingId === session.id ? (
                      <input
                        className="session-rename-input"
                        value={draft}
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setDraft(e.target.value)}
                        onBlur={() => setEditingId(null)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitEdit(session.id);
                          if (e.key === 'Escape') setEditingId(null);
                        }}
                      />
                    ) : (
                      <div className="session-title">{session.title}</div>
                    )}
                    <div className="session-time">{relativeTime(session.updated_at)}</div>
                  </div>

                  <span className="session-badge">{session.message_count}</span>

                  <div className="session-actions">
                    <button className="session-icon" type="button" title="Rename" onClick={(e) => beginEdit(session, e)}>
                      <PencilIcon />
                    </button>
                    <button className="session-icon" type="button" title="Delete" onClick={(e) => confirmDelete(session, e)}>
                      <TrashIcon />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      <div className="session-footer">
        <ClockIcon /> {sessions.length} session{sessions.length !== 1 ? 's' : ''} saved
      </div>
    </aside>
  );
}

function PencilIcon() {
  return <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M11.5 2.5l2 2L6 12l-2.5.5L4 10z" /></svg>;
}

function TrashIcon() {
  return <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M3 4h10M6 4V2.5h4V4M4.5 4l.5 9h6l.5-9" /></svg>;
}

function ClockIcon() {
  return <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="6" /><path d="M8 4.5V8l2.5 1.5" /></svg>;
}
