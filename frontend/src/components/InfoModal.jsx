import { INFO_CONTENT } from '../data/infoContent.js';

export default function InfoModal({ topic, onClose }) {
  if (!topic) return null;
  const content = INFO_CONTENT[topic] || {
    title: 'Module Info',
    desc: '-',
    read: '-',
    sample: '-',
  };

  return (
    <div className="modal-overlay open" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 650 }} onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <span className="modal-title">{content.title}</span>
          <button className="modal-close" onClick={onClose} type="button">x</button>
        </div>
        <div className="modal-body" style={{ gap: '1.25rem', maxHeight: '75vh', overflowY: 'auto' }}>
          <div className="detail-field"><div className="detail-label">How it works</div><div className="detail-desc">{content.desc}</div></div>
          <div className="detail-field"><div className="detail-label">How to read the data</div><div className="detail-desc">{content.read}</div></div>
          <div className="detail-field"><div className="detail-label">Sample Data (JSON / Format)</div><div className="code-block">{content.sample}</div></div>
        </div>
      </div>
    </div>
  );
}
