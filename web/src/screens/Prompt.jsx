import { useState } from "react";
import { listenOnce } from "../voice.js";

export default function PromptScreen({ onSubmit }) {
  const [text, setText] = useState("record a 15 second video of this apple");
  const [hint, setHint] = useState("");

  async function onMic() {
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
          Mic
        </button>
        <button type="button" onClick={() => onSubmit(text.trim())} disabled={!text.trim()}>
          Start
        </button>
      </div>
      {hint && <p className="hint">{hint}</p>}
    </section>
  );
}
