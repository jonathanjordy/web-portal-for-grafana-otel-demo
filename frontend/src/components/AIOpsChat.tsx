'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Terminal, CheckCircle } from 'lucide-react';
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

  // Auto scroll to bottom
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

    // Append user message
    const userMsg: ChatMessage = { role: 'user', content: text };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);

    // Prepare assistant thinking message placeholder
    const thinkingMsg: ChatMessage = { 
      role: 'assistant', 
      content: '<em class="text-text-tertiary select-none">Thinking... Analyzing schema and drafting ClickHouse SQL...</em>' 
    };
    setMessages(prev => [...prev, thinkingMsg]);

    try {
      // Map frontend roles to match what the backend expects or handles
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
      
      // Update thinking message with actual response
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
          content: '<span class="text-status-error">Backend assistant service is currently unreachable.</span>'
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel flex flex-col h-[600px] shadow-md animate-fade select-none">
      {/* Messages Window */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-bg-main/40">
        {messages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={idx}
              className={`flex flex-col gap-1 max-w-[80%] ${isUser ? 'ml-auto items-end animate-fade' : 'mr-auto items-start animate-fade'}`}
            >
              <div className="text-[9px] font-bold text-text-tertiary tracking-widest uppercase px-1">
                {isUser ? 'You' : 'AIOps Assistant'}
              </div>
              <div
                className={`chat-bubble px-4 py-3 rounded-2xl text-[13px] leading-relaxed select-text ${
                  isUser
                    ? 'bg-indosat-teal text-white rounded-tr-none font-semibold'
                    : 'bg-surface-card border border-border-subtle text-text-primary rounded-tl-none font-medium'
                }`}
                dangerouslySetInnerHTML={{ __html: msg.content }}
              />
              {msg.executed && msg.row_count !== undefined && (
                <div className="flex items-center gap-1 text-[10px] font-bold text-indosat-teal font-mono mt-0.5 select-none">
                  <CheckCircle className="w-3 h-3" />
                  <span>✓ {msg.row_count} row{msg.row_count !== 1 ? 's' : ''} returned</span>
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Tray */}
      <div className="bg-surface-card border-t border-border-subtle flex flex-col">
        {/* Suggestion Chips */}
        <div className="flex flex-wrap gap-1.5 p-3.5 pb-2.5">
          {suggestions.map((sug, i) => (
            <button
              key={i}
              onClick={() => handleSend(sug)}
              disabled={loading}
              className="text-xs font-semibold px-3 py-1.5 rounded-full border border-border-subtle bg-bg-main text-text-secondary hover:bg-indosat-teal hover:border-indosat-teal hover:text-white transition-all duration-150 cursor-pointer disabled:opacity-50 disabled:pointer-events-none select-none"
            >
              {sug}
            </button>
          ))}
        </div>

        {/* TextInput row */}
        <div className="flex items-center gap-3 p-4 border-t border-border-subtle/50">
          <div className="relative flex-1">
            <input
              type="text"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSend();
              }}
              disabled={loading}
              placeholder="Ask anything about your OTel telemetry database schemas..."
              className="w-full text-sm font-semibold px-4 py-2.5 border border-border-medium rounded-lg bg-bg-main text-text-primary outline-none focus:border-indosat-teal focus:ring-4 focus:ring-status-ok-bg/50 transition-all placeholder:text-text-tertiary select-text"
            />
            <Terminal className="absolute right-3.5 top-3.5 w-4 h-4 text-text-tertiary select-none" />
          </div>
          <button
            onClick={() => handleSend()}
            disabled={loading || !inputVal.trim()}
            className="p-2.5 rounded-lg bg-indosat-teal text-white hover:bg-indosat-teal/90 disabled:opacity-50 cursor-pointer shadow-sm transition-transform active:scale-95"
            title="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
