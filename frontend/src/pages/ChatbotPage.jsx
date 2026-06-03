import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { apiGet, apiPost, apiPatch, apiDelete } from '../services/api.js';
import SessionHistory from '../components/SessionHistory.jsx';

const suggestions = [
  'How many orders in the last hour?',
  'Show payment failure rate by item',
  'Which service had the most errors today?',
  'What is the p95 order duration?',
  'Show node memory usage trend',
];

const GREETING = {
  role: 'assistant',
  html: 'Hello. I have access to your ClickHouse <code>otel</code> database — traces, logs, and metrics from all three services plus node exporter data. What would you like to know?',
};

export default function ChatbotPage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([GREETING]);
  const [sending, setSending] = useState(false);
  const messagesRef = useRef(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    });
  }

  async function refreshSessions() {
    try {
      const data = await apiGet('/chatbot/sessions');
      const list = data.sessions || [];
      setSessions(list);
      return list;
    } catch {
      return [];
    }
  }

  function turnsToMessages(turns) {
    return turns.map((turn) =>
      turn.role === 'user'
        ? { role: 'user', html: escapeHtml(turn.content) }
        : { role: 'assistant', html: turn.html || escapeHtml(turn.content) },
    );
  }

  // The URL is the source of truth. Load the session in the URL, or redirect to
  // a sensible default (newest session, or a freshly created one).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await refreshSessions();
      if (cancelled) return;

      if (sessionId) {
        const data = await apiGet(`/chatbot/sessions/${sessionId}`);
        if (cancelled) return;
        if (data && Array.isArray(data.turns)) {
          setMessages([GREETING, ...turnsToMessages(data.turns)]);
          scrollToBottom();
        } else {
          navigate('/chat', { replace: true }); // unknown / stale id
        }
      } else if (list.length > 0) {
        navigate(`/chat/${list[0].id}`, { replace: true });
      } else {
        const created = await apiPost('/chatbot/sessions', {});
        if (cancelled) return;
        navigate(`/chat/${created.id}`, { replace: true });
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  async function newChat() {
    const session = await apiPost('/chatbot/sessions', {});
    await refreshSessions();
    navigate(`/chat/${session.id}`);
  }

  function selectSession(id) {
    navigate(`/chat/${id}`);
  }

  async function renameSession(id, title) {
    await apiPatch(`/chatbot/sessions/${id}`, { title });
    refreshSessions();
  }

  async function deleteSession(id) {
    await apiDelete(`/chatbot/sessions/${id}`);
    const list = await refreshSessions();
    if (id === sessionId) {
      if (list.length > 0) navigate(`/chat/${list[0].id}`);
      else navigate('/chat');
    }
  }

  async function sendMessage(message = input.trim()) {
    if (!message || sending || !sessionId) return;
    setInput('');
    setSending(true);

    const userMsg = { role: 'user', html: escapeHtml(message) };
    const thinking = { role: 'assistant', html: '<em style="color:var(--text-3)">Thinking...</em>' };
    setMessages((current) => [...current, userMsg, thinking]);
    scrollToBottom();

    try {
      const data = await apiPost('/chatbot/chat', { message, session_id: sessionId });
      const reply = data.response || 'Sorry, I could not process that.';
      setMessages((current) => {
        const updated = current.slice(0, -1);
        updated.push({
          role: 'assistant',
          html: reply,
          badge: data.executed && data.row_count !== undefined ? `${data.row_count} row${data.row_count !== 1 ? 's' : ''} returned` : '',
        });
        return updated;
      });
      refreshSessions();
    } catch {
      setMessages((current) => {
        const updated = current.slice(0, -1);
        updated.push({ role: 'assistant', html: 'Backend unreachable.' });
        return updated;
      });
    } finally {
      setSending(false);
      scrollToBottom();
    }
  }

  return (
    <div className="page active page-wide">
      <div className="page-eyebrow">Page 4 - AIOps Assistant</div>
      <h1 className="page-title">Talk to Your Data</h1>
      <p className="page-desc">Ask questions in plain English. The assistant translates them into ClickHouse SQL, runs the query, and returns results as tables or charts.</p>
      <div className="chat-layout">
        <div className="chat-wrap">
          <div className="chat-messages" ref={messagesRef}>
            {messages.map((message, index) => (
              <div className={`msg ${message.role}`} key={`${message.role}-${index}`}>
                <div className="msg-role">{message.role === 'user' ? 'You' : 'Assistant'}</div>
                <div className="msg-bubble" dangerouslySetInnerHTML={{ __html: message.html }} />
                {message.badge ? <div style={{ fontSize: '0.7rem', color: 'var(--green)', fontWeight: 700, marginTop: '0.25rem', fontFamily: 'monospace' }}>{message.badge}</div> : null}
              </div>
            ))}
          </div>
          <div className="chat-bottom">
            <div className="suggestions">
              {suggestions.map((item) => (
                <button className="suggestion" onClick={() => sendMessage(item)} key={item} type="button">{item}</button>
              ))}
            </div>
            <div className="chat-input-row">
              <input
                type="text"
                value={input}
                placeholder="Ask anything about your telemetry data..."
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') sendMessage();
                }}
              />
              <button className="btn" onClick={() => sendMessage()} disabled={sending} type="button">Send</button>
            </div>
          </div>
        </div>

        <SessionHistory
          sessions={sessions}
          activeSessionId={sessionId}
          onNewChat={newChat}
          onSelect={selectSession}
          onRename={renameSession}
          onDelete={deleteSession}
        />
      </div>
    </div>
  );
}

function escapeHtml(text) {
  const element = document.createElement('div');
  element.textContent = text;
  return element.innerHTML;
}
