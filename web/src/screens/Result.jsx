import { useState } from "react";
import { mediaUrl } from "../api.js";

export default function ResultScreen({ result, onAgain }) {
  const raw = mediaUrl(result?.url);
  const edit = mediaUrl(result?.edit_url);
  const [showRaw, setShowRaw] = useState(false);
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
      <h1>{edit ? "Edit" : result?.duration_s ? `${result.duration_s}s clip` : "Clip"}</h1>
      {edit ? (
        <video src={edit} controls playsInline />
      ) : (
        <>
          <p className="hint">The music edit did not finish. The raw clip is ready.</p>
          <video src={raw} controls playsInline />
        </>
      )}
      <div className="row wrap">
        {edit ? (
          <a className="button" href={edit} download>
            Save
          </a>
        ) : (
          raw && (
            <a className="button" href={raw} download>
              Save
            </a>
          )
        )}
        {edit && raw && (
          <button type="button" className="ghost" aria-expanded={showRaw} onClick={() => setShowRaw((open) => !open)}>
            {showRaw ? "Hide raw" : "Raw"}
          </button>
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
      {edit && showRaw && raw && (
        <div className="raw">
          <h2>Raw</h2>
          <video src={raw} controls playsInline />
          <div className="row">
            <a className="button ghost" href={raw} download>
              Save raw
            </a>
          </div>
        </div>
      )}
    </section>
  );
}
