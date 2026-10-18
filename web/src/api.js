const API_URL = import.meta.env.VITE_API_URL || "";

export function mediaUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_URL}${path}`;
}

function audioName(type) {
  if (type.includes("mp4") || type.includes("m4a") || type.includes("aac")) return "speech.m4a";
  if (type.includes("ogg")) return "speech.ogg";
  if (type.includes("wav")) return "speech.wav";
  return "speech.webm";
}

export async function transcribe(blob) {
  const body = new FormData();
  body.append("file", blob, audioName(blob.type || ""));
  const response = await fetch(`${API_URL}/transcribe`, { method: "POST", body });
  if (!response.ok) {
    let detail = "transcription failed";
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string" && payload.detail) detail = payload.detail;
    } catch {
      /* the status line is enough */
    }
    throw new Error(detail);
  }
  const payload = await response.json();
  return payload.text || "";
}
