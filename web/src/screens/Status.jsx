import { mediaUrl } from "../api.js";

const STAGES = ["PARSE", "LOCATE", "SCOUT", "PLAN", "RECORD", "POST", "DONE"];

export default function StatusScreen({ stage, detail, error, onBack, stills = [], bestIdx = null }) {
  const active = STAGES.indexOf(stage);
  return (
    <section>
      <h1>{error ? "Something stalled" : "Filming"}</h1>
      <ol className="steps">
        {STAGES.map((name, index) => (
          <li key={name} className={index === active ? "on" : index < active ? "done" : ""}>
            {name}
          </li>
        ))}
      </ol>
      {stills.length > 0 && (
        <div className="stills">
          {stills.map((url, index) => (
            <img
              key={url}
              src={mediaUrl(url)}
              alt={`scout ${index}`}
              className={index === bestIdx ? "best" : ""}
            />
          ))}
        </div>
      )}
      <p className="hint">{detail}</p>
      {error && (
        <div className="row">
          <button type="button" onClick={onBack}>
            Back
          </button>
        </div>
      )}
    </section>
  );
}
