import { useRef, useState } from "react";
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
  const [stills, setStills] = useState([]);
  const [bestIdx, setBestIdx] = useState(null);
  const closeJob = useRef(null);

  function stopJob() {
    closeJob.current?.();
    closeJob.current = null;
  }

  function backToPrompt() {
    stopJob();
    setScreen("prompt");
  }

  function start(text) {
    stopJob();
    setScreen("status");
    setStage("PARSE");
    setDetail("connecting");
    setResult(null);
    setError("");
    setStills([]);
    setBestIdx(null);
    closeJob.current = connectJob(text, {
      onStatus(message) {
        setStage(message.stage);
        setDetail(message.detail);
        if (message.stills) setStills(message.stills);
        if (message.best_idx !== undefined && message.best_idx !== null) setBestIdx(message.best_idx);
        if (message.stage === "ERROR") setError(message.detail);
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
      {screen === "status" && (
        <StatusScreen
          stage={stage}
          detail={detail}
          error={error}
          onBack={backToPrompt}
          stills={stills}
          bestIdx={bestIdx}
        />
      )}
      {screen === "result" && <ResultScreen result={result} onAgain={backToPrompt} />}
    </main>
  );
}
