import { mediaUrl } from "../api.js";

export default function ResultScreen({ result, onAgain }) {
  const raw = mediaUrl(result?.url);
  const edit = mediaUrl(result?.edit_url);
  const shareUrl = edit || raw;
  const canShare = typeof navigator.share === "function";

  async function onShare() {
    try {
      await navigator.share({ title: "Clip", url: shareUrl });
    } catch (error) {
      if (error?.name === "AbortError") return;
    }
  }

  return (
    <section>
      <h1>{result?.duration_s ? `${result.duration_s}s clip` : "Clip"}</h1>
      <h2>Raw</h2>
      <video src={raw} controls playsInline />
      <h2>Edit</h2>
      {edit ? (
        <video src={edit} controls playsInline />
      ) : (
        <p className="hint">The music edit did not finish. The raw clip is ready.</p>
      )}
      <div className="row">
        <a className="button" href={raw} download>
          Raw
        </a>
        {edit && (
          <a className="button" href={edit} download>
            Edit
          </a>
        )}
        {canShare && (
          <button type="button" onClick={onShare}>
            Share
          </button>
        )}
        <button type="button" className="ghost" onClick={onAgain}>
          Again
        </button>
      </div>
    </section>
  );
}
