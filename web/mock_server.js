import { WebSocketServer } from "ws";

const port = Number(process.env.MOCK_WS_PORT || 8787);
const wss = new WebSocketServer({ port });

const script = [
  { delay: 400, msg: { type: "status", stage: "PARSE", detail: "apple, 15s, orbit" } },
  { delay: 500, msg: { type: "status", stage: "LOCATE", detail: "apple found" } },
  { delay: 500, msg: { type: "status", stage: "SCOUT", detail: "scout_0" } },
  { delay: 300, msg: { type: "capture_still", job_id: "j1", pose_id: 0 } },
  { delay: 500, msg: { type: "status", stage: "SCOUT", detail: "scout_1" } },
  { delay: 300, msg: { type: "capture_still", job_id: "j1", pose_id: 1 } },
  { delay: 500, msg: { type: "status", stage: "PLAN", detail: "best angle 1, move orbit" } },
  { delay: 400, msg: { type: "start_recording", job_id: "j1" } },
  { delay: 1200, msg: { type: "status", stage: "RECORD", detail: "orbit" } },
  { delay: 400, msg: { type: "stop_recording", job_id: "j1" } },
  { delay: 600, msg: { type: "status", stage: "POST", detail: "trimming to 15s" } },
  { delay: 400, msg: { type: "result", job_id: "j1", url: "/media/j1_final.mp4", duration_s: 15 } },
  { delay: 200, msg: { type: "status", stage: "DONE", detail: "clip ready" } },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

wss.on("connection", (ws) => {
  ws.on("message", async (raw) => {
    const data = JSON.parse(raw.toString());
    if (data.type !== "prompt") {
      ws.send(JSON.stringify({ type: "status", stage: "ERROR", detail: "expected a prompt" }));
      return;
    }
    for (const step of script) {
      await sleep(step.delay);
      if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(step.msg));
    }
  });
});

console.log(`mock ws on ws://localhost:${port}`);
