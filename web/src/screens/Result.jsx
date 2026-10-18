import { mediaUrl } from "../api.js";

export default function ResultScreen({ result, onAgain }) {
  const src = mediaUrl(result?.url);
  const canShare = typeof navigator.share === "function";

  async function onShare() {
    try {
      await navigator.share({ title: "Clip", url: src });
    } catch (error) {
      if (error?.name === "AbortError") return;
    }
  }

  return (
    <section>
      <h1>{result?.duration_s ? `${result.duration_s}s clip` : "Clip"}</h1>
      <video src={src} controls playsInline />
      <div className="row">
        <a className="button" href={src} download>
          Download
        </a>
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
