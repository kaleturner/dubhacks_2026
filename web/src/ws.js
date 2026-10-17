const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8787";

export function connectJob(text, handlers) {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "prompt", text }));
  };
  ws.onerror = () => {
    handlers.onStatus?.({ type: "status", stage: "ERROR", detail: `cannot reach ${WS_URL}` });
  };
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "status") handlers.onStatus?.(message);
    if (message.type === "result") handlers.onResult?.(message);
  };
  return () => ws.close();
}
