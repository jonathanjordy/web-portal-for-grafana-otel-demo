import { useState } from 'react';
import InfoModal from './components/InfoModal.jsx';
import Sidebar from './components/Sidebar.jsx';
import ChatbotPage from './pages/ChatbotPage.jsx';
import DetectivePage from './pages/DetectivePage.jsx';
import DiagnosticPage from './pages/DiagnosticPage.jsx';
import IncidentsPage from './pages/IncidentsPage.jsx';
import PredictivePage from './pages/PredictivePage.jsx';

export default function App() {
  const [activePage, setActivePage] = useState('predictive');
  const [infoTopic, setInfoTopic] = useState('');

  return (
    <div className="shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main>
        {activePage === 'predictive' ? <PredictivePage onOpenInfo={setInfoTopic} /> : null}
        {activePage === 'detective' ? <DetectivePage onOpenInfo={setInfoTopic} /> : null}
        {activePage === 'diagnostic' ? <DiagnosticPage onOpenInfo={setInfoTopic} /> : null}
        {activePage === 'chatbot' ? <ChatbotPage /> : null}
        {activePage === 'incidents' ? <IncidentsPage /> : null}
        <InfoModal topic={infoTopic} onClose={() => setInfoTopic('')} />
      </main>
    </div>
  );
}
