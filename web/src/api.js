const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function mediaUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_URL}${path}`;
}

export async function uploadStill(jobId, poseId, blob) {
  const body = new FormData();
  body.append("file", blob, "still.jpg");
  const response = await fetch(`${API_URL}/upload/still?job_id=${jobId}&pose_id=${poseId}`, {
    method: "POST",
    body,
  });
  if (!response.ok) throw new Error("still upload failed");
  return response.json();
}

export async function uploadVideo(jobId, blob, filename) {
  const body = new FormData();
  body.append("file", blob, filename);
  const response = await fetch(`${API_URL}/upload/video?job_id=${jobId}`, { method: "POST", body });
  if (!response.ok) throw new Error("video upload failed");
  return response.json();
}

export async function transcribe(blob) {
  const body = new FormData();
  body.append("file", blob, "speech.webm");
  const response = await fetch(`${API_URL}/transcribe`, { method: "POST", body });
  const payload = await response.json();
  return payload.text || "";
}
