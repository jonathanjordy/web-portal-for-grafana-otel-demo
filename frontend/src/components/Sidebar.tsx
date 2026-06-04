"use client";

import React from 'react';
import { HealthState } from '../hooks/useHealthCheck';

interface SidebarProps {
  currentPage: string;
  onPageChange: (page: string) => void;
  health: HealthState;
}

export default function Sidebar({ currentPage, onPageChange, health }: SidebarProps) {
  const getHealthText = () => {
    switch (health) {
      case 'ok': return 'ClickHouse connected';
      case 'error': return 'ClickHouse error';
      case 'unreachable': return 'API unreachable';
      case 'checking': default: return 'Checking...';
    }
  };

  const getHealthDotClass = () => {
    switch (health) {
      case 'ok': return 'dot ok';
      case 'error': return 'dot error';
      case 'unreachable': return 'dot error';
      case 'checking': default: return 'dot';
    }
  };

  return (
    <aside>
      <div className="sidebar-top">
        <div className="wordmark">
          AIOps Portal
          <span>Observability Intelligence</span>
        </div>
      </div>
      
      <div className="nav-group">
        <div className="nav-label">Analytics</div>
        <button 
          className={`nav-item ${currentPage === 'predictive' ? 'active' : ''}`}
          onClick={() => onPageChange('predictive')}
          data-page="predictive"
        >
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="1,12 5,7 8,9 12,4 15,5" />
            <line x1="15" y1="2" x2="15" y2="8" />
            <line x1="12" y1="8" x2="15" y2="8" />
          </svg>
          Predictive
        </button>
        <button 
          className={`nav-item ${currentPage === 'detective' ? 'active' : ''}`}
          onClick={() => onPageChange('detective')}
          data-page="detective"
        >
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="6.5" cy="6.5" r="4.5" />
            <line x1="10" y1="10" x2="14" y2="14" />
          </svg>
          Detective
        </button>
        <button 
          className={`nav-item ${currentPage === 'diagnostic' ? 'active' : ''}`}
          onClick={() => onPageChange('diagnostic')}
          data-page="diagnostic"
        >
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="8" cy="8" r="6.5" />
            <line x1="8" y1="5" x2="8" y2="8" />
            <circle cx="8" cy="11" r="0.5" fill="currentColor" />
          </svg>
          Diagnostic
        </button>
      </div>

      <div className="nav-group">
        <div className="nav-label">Assistant</div>
        <button 
          className={`nav-item ${currentPage === 'chatbot' ? 'active' : ''}`}
          onClick={() => onPageChange('chatbot')}
          data-page="chatbot"
        >
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 10.5c0 .8-.7 1.5-1.5 1.5H4l-2.5 2.5V3.5C1.5 2.7 2.2 2 3 2h9.5c.8 0 1.5.7 1.5 1.5v7z" />
          </svg>
          AIOps Chat
        </button>
      </div>

      <div className="nav-group">
        <div className="nav-label">Operations</div>
        <button 
          className={`nav-item ${currentPage === 'incidents' ? 'active' : ''}`}
          onClick={() => onPageChange('incidents')}
          data-page="incidents"
        >
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="2" width="12" height="12" rx="2" />
            <line x1="5" y1="6" x2="11" y2="6" />
            <line x1="5" y1="9" x2="9" y2="9" />
          </svg>
          Incident Registry
        </button>
      </div>

      <div className="sidebar-footer">
        <div className="health-row">
          <div className={getHealthDotClass()} id="health-dot" />
          <span id="health-text">{getHealthText()}</span>
        </div>
      </div>
    </aside>
  );
}
