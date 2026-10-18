import { useState } from "react";
import { transcribe } from "../api.js";
import { beginRecording, hasSpeechRecognition, listenOnce, stopRecording } from "../voice.js";

const EMPTY_TRANSCRIPT = "no words came back; type the prompt instead";

export default function PromptScreen({ onSubmit }) {
  const [text, setText] = useState("record a 15 second video of this apple");
  const [hint, setHint] = useState("");
  const [recording, setRecording] = useState(false);

  async function onMic() {
    if (!hasSpeechRecognition()) {
      if (recording) {
        setRecording(false);
        setHint("transcribing");
        try {
          const blob = await stopRecording();
          const heard = await transcribe(blob);
          if (!heard) {
            setHint(EMPTY_TRANSCRIPT);
            return;
          }
          setText(heard);
          setHint("");
        } catch (error) {
          setHint(error.message);
        }
        return;
      }
      setHint("recording");
      try {
        await beginRecording();
        setRecording(true);
      } catch (error) {
        setHint(error.message);
      }
      return;
    }

    setHint("listening");
    try {
      const heard = await listenOnce();
      setText(heard);
      setHint("");
    } catch (error) {
      setHint(error.message);
    }
  }

  return (
    <section>
      <h1>What should we film?</h1>
      <textarea value={text} onChange={(event) => setText(event.target.value)} rows={4} />
      <div className="row">
        <button type="button" className="ghost" onClick={onMic}>
          {recording ? "Stop" : "Mic"}
        </button>
        <button type="button" onClick={() => onSubmit(text.trim())} disabled={!text.trim() || recording}>
          Start
        </button>
      </div>
      {hint && <p className="hint">{hint}</p>}
    </section>
  );
}
