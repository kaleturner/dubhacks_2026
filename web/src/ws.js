function socketUrl() {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

const WS_URL = socketUrl();

export function connectJob(text, handlers) {
  const ws = new WebSocket(WS_URL);
  let finished = false;
  let closedByClient = false;

  function fail(detail) {
    if (finished || closedByClient) return;
    finished = true;
    handlers.onStatus?.({ type: "status", stage: "ERROR", detail });
  }

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "prompt", text }));
  };
  ws.onerror = () => {
    fail(`cannot reach ${WS_URL}`);
  };
  ws.onclose = () => {
    fail("connection closed");
  };
  ws.onmessage = (event) => {
    if (closedByClient) return;
    const message = JSON.parse(event.data);
    if (message.type === "status") {
      if (message.stage === "ERROR" || message.stage === "DONE") finished = true;
      handlers.onStatus?.(message);
    }
    if (message.type === "result") {
      finished = true;
      handlers.onResult?.(message);
    }
  };
  return () => {
    closedByClient = true;
    finished = true;
    ws.close();
  };
}
