import { useRef, useState } from 'react';
import { apiPost } from '../services/api.js';

const suggestions = [
  'How many orders in the last hour?',
  'Show payment failure rate by item',
  'Which service had the most errors today?',
  'What is the p95 order duration?',
  'Show node memory usage trend',
];

export default function ChatbotPage() {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      html: 'Hello. I have access to your ClickHouse <code>otel</code> database - traces, logs, and metrics from all three services plus node exporter data. What would you like to know?',
    },
  ]);
  const [sending, setSending] = useState(false);
  const messagesRef = useRef(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    });
  }

  async function sendMessage(message = input.trim()) {
    if (!message || sending) return;
    setInput('');
    setSending(true);

    const userMsg = { role: 'user', html: escapeHtml(message) };
    const thinking = { role: 'assistant', html: '<em style="color:var(--text-3)">Thinking...</em>' };
    const nextHistory = [...history, { role: 'user', content: message }];
    setMessages((current) => [...current, userMsg, thinking]);
    scrollToBottom();

    try {
      const data = await apiPost('/chatbot/chat', { message, history: nextHistory });
      const reply = data.response || 'Sorry, I could not process that.';
      const plainReply = stripHtml(reply).slice(0, 300);
      setHistory([...nextHistory, { role: 'assistant', content: plainReply }]);
      setMessages((current) => {
        const updated = current.slice(0, -1);
        updated.push({
          role: 'assistant',
          html: reply,
          badge: data.executed && data.row_count !== undefined ? `${data.row_count} row${data.row_count !== 1 ? 's' : ''} returned` : '',
        });
        return updated;
      });
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
    <div className="page active">
      <div className="page-eyebrow">Page 4 - AIOps Assistant</div>
      <h1 className="page-title">Talk to Your Data</h1>
      <p className="page-desc">Ask questions in plain English. The assistant translates them into ClickHouse SQL, runs the query, and returns results as tables or charts.</p>
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
    </div>
  );
}

function stripHtml(html) {
  const element = document.createElement('div');
  element.innerHTML = html;
  return element.textContent || element.innerText || '';
}

function escapeHtml(text) {
  const element = document.createElement('div');
  element.textContent = text;
  return element.innerHTML;
}
