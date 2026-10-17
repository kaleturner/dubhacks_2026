import { useState } from "react";
import PromptScreen from "./screens/Prompt.jsx";
import StatusScreen from "./screens/Status.jsx";
import ResultScreen from "./screens/Result.jsx";
import { connectJob } from "./ws.js";

export default function App() {
  const [screen, setScreen] = useState("prompt");
  const [stage, setStage] = useState("PARSE");
  const [detail, setDetail] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  function start(text) {
    setScreen("status");
    setStage("PARSE");
    setDetail("connecting");
    setResult(null);
    setError("");
    connectJob(text, {
      onStatus(message) {
        setStage(message.stage);
        setDetail(message.detail);
        if (message.stage === "ERROR") setError(message.detail);
      },
      onCaptureStill(message) {
        setDetail(`snap pose ${message.pose_id}`);
        // TODO: grab a still from the rear camera and POST /upload/still.
      },
      onStartRecording() {
        setDetail("recording");
        // TODO: MediaRecorder start.
      },
      onStopRecording() {
        setDetail("stopping");
        // TODO: MediaRecorder stop, then POST /upload/video.
      },
      onResult(message) {
        setResult(message);
        setScreen("result");
      },
    });
  }

  return (
    <main>
      <p className="eyebrow">DubHacks recorder</p>
      {screen === "prompt" && <PromptScreen onSubmit={start} />}
      {screen === "status" && <StatusScreen stage={stage} detail={detail} error={error} />}
      {screen === "result" && <ResultScreen result={result} onAgain={() => setScreen("prompt")} />}
    </main>
  );
}
