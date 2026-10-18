import { wsUrl } from "./server.js";

export function connectJob(text, handlers) {
  const url = wsUrl();
  const ws = new WebSocket(url);
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
    fail(`cannot reach ${url}`);
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
