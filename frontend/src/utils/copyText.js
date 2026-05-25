export function copyText(text, onDone) {
  navigator.clipboard.writeText(text).then(() => {
    if (onDone) onDone();
  });
}
