'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types/otel';

interface AIOpsChatProps {
  apiBase: string;
}

export default function AIOpsChat({ apiBase }: AIOpsChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Hello. I have access to your ClickHouse <code>otel</code> database — traces, logs, and metrics from all three services plus node exporter data. What would you like to know?'
    }
  ]);
  const [inputVal, setInputVal] = useState('');
  const [loading, setLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestions = [
    'How many orders in the last hour?',
    'Show payment failure rate by item',
    'Which service had the most errors today?',
    'What is the p95 order duration?',
    'Show node memory usage trend'
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputVal).trim();
    if (!text || loading) return;

    setInputVal('');
    setLoading(true);

    const userMsg: ChatMessage = { role: 'user', content: text };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);

    // Prepare assistant thinking message placeholder
    const thinkingMsg: ChatMessage = { 
      role: 'assistant', 
      content: '<em style="color: var(--text-3)">Thinking... Analyzing schema and drafting ClickHouse SQL...</em>' 
    };
    setMessages(prev => [...prev, thinkingMsg]);

    try {
      const historyPayload = nextMessages.map(m => ({
        role: m.role,
        content: m.content
      }));

      const res = await fetch(`${apiBase}/chatbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: historyPayload })
      });
      const data = await res.json();
      
      const reply = data.response || 'Sorry, I could not process that request.';
      
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        updated[lastIdx] = {
          role: 'assistant',
          content: reply,
          executed: data.executed,
          row_count: data.row_count
        };
        return updated;
      });
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        updated[lastIdx] = {
          role: 'assistant',
          content: '<span style="color: var(--red)">Backend assistant service is currently unreachable.</span>'
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-wrap">
      <div className="chat-messages" id="chat-messages">
        {messages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          return (
            <div key={idx} className={`msg ${isUser ? 'user' : 'assistant'}`}>
              <div className="msg-role">{isUser ? 'You' : 'Assistant'}</div>
              <div
                className="msg-bubble"
                dangerouslySetInnerHTML={{ __html: msg.content }}
              />
              {msg.executed && msg.row_count !== undefined && (
                <div style={{ fontSize: '0.72rem', color: 'var(--green)', marginTop: '4px', fontWeight: 'bold' }}>
                  ✓ {msg.row_count} row{msg.row_count !== 1 ? 's' : ''} returned
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-bottom">
        <div className="suggestions" id="suggestions">
          {suggestions.map((sug, i) => (
            <div
              key={i}
              className="suggestion"
              onClick={() => !loading && handleSend(sug)}
              style={{ cursor: loading ? 'not-allowed' : 'pointer' }}
            >
              {sug}
            </div>
          ))}
        </div>
        <div className="chat-input-row">
          <input
            type="text"
            id="chat-input"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSend();
            }}
            disabled={loading}
            placeholder="Ask anything about your telemetry data..."
          />
          <button
            className="btn"
            onClick={() => handleSend()}
            disabled={loading}
            id="send-btn"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
