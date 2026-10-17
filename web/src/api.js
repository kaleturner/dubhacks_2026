const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function mediaUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_URL}${path}`;
}

export async function transcribe(blob) {
  const body = new FormData();
  body.append("file", blob, "speech.webm");
  const response = await fetch(`${API_URL}/transcribe`, { method: "POST", body });
  const payload = await response.json();
  return payload.text || "";
}
