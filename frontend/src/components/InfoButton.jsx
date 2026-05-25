export default function InfoButton({ topic, onOpen }) {
  return (
    <button className="btn-info" onClick={() => onOpen(topic)} title="How this works" type="button">
      i
    </button>
  );
}
