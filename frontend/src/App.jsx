import { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import InfoModal from './components/InfoModal.jsx';
import Sidebar from './components/Sidebar.jsx';
import ChatbotPage from './pages/ChatbotPage.jsx';
import DetectivePage from './pages/DetectivePage.jsx';
import DiagnosticPage from './pages/DiagnosticPage.jsx';
import IncidentsPage from './pages/IncidentsPage.jsx';
import PredictivePage from './pages/PredictivePage.jsx';

export default function App() {
  const [infoTopic, setInfoTopic] = useState('');

  return (
    <BrowserRouter>
      <div className="shell">
        <Sidebar />
        <main>
          <Routes>
            <Route path="/" element={<Navigate to="/predictive" replace />} />
            <Route path="/predictive" element={<PredictivePage onOpenInfo={setInfoTopic} />} />
            <Route path="/detective" element={<DetectivePage onOpenInfo={setInfoTopic} />} />
            <Route path="/diagnostic" element={<DiagnosticPage onOpenInfo={setInfoTopic} />} />
            <Route path="/chat" element={<ChatbotPage />} />
            <Route path="/chat/:sessionId" element={<ChatbotPage />} />
            <Route path="/incidents" element={<IncidentsPage />} />
            <Route path="*" element={<Navigate to="/predictive" replace />} />
          </Routes>
          <InfoModal topic={infoTopic} onClose={() => setInfoTopic('')} />
        </main>
      </div>
    </BrowserRouter>
  );
}
