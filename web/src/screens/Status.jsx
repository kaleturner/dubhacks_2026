const STAGES = ["PARSE", "LOCATE", "SCOUT", "PLAN", "RECORD", "POST", "DONE"];

export default function StatusScreen({ stage, detail, error }) {
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
      <p className="hint">{detail}</p>
    </section>
  );
}
