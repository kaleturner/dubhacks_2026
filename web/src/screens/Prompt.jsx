import { useRef, useState } from "react";
import { transcribe } from "../api.js";
import { listenForPrompt } from "../voice.js";

const EMPTY_TRANSCRIPT = "no words came back; type the prompt instead";

export default function PromptScreen({ onSubmit }) {
  const [text, setText] = useState("record a 15 second video of this apple");
  const [hint, setHint] = useState("");
  const [listening, setListening] = useState(false);
  const [busy, setBusy] = useState(false);
  const session = useRef(null);

  async function onMic() {
    if (session.current) {
      session.current.stop();
      return;
    }
    setListening(true);
    setHint("listening");
    const run = listenForPrompt();
    session.current = run;
    try {
      const heard = await run.done;
      session.current = null;
      setListening(false);
      let words = heard.text || "";
      if (!words && heard.blob && heard.blob.size) {
        setBusy(true);
        setHint("transcribing");
        words = await transcribe(heard.blob);
      }
      if (!words) {
        setHint(EMPTY_TRANSCRIPT);
        return;
      }
      setText(words);
      setHint("");
    } catch (error) {
      setHint(error.message);
    } finally {
      session.current = null;
      setListening(false);
      setBusy(false);
    }
  }

  return (
    <section>
      <h1>What should we film?</h1>
      <textarea value={text} onChange={(event) => setText(event.target.value)} rows={4} />
      <div className="row">
        <button type="button" className="ghost" onClick={onMic} disabled={busy}>
          {listening ? "Stop" : "Mic"}
        </button>
        <button type="button" onClick={() => onSubmit(text.trim())} disabled={!text.trim() || listening || busy}>
          Start
        </button>
      </div>
      {hint && <p className="hint">{hint}</p>}
    </section>
  );
}
