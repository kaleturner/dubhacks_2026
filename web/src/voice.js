/** webkitSpeechRecognition, with a MediaRecorder take when that API is missing. */

const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];

let active = null;

export function hasSpeechRecognition() {
  return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
}

export function listenOnce() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    return Promise.reject(new Error("no speech recognition"));
  }
  return new Promise((resolve, reject) => {
    const recognition = new Recognition();
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      resolve(event.results[0][0].transcript);
    };
    recognition.onerror = () => reject(new Error("speech recognition failed"));
    recognition.start();
  });
}

export function preferredMime() {
  if (typeof MediaRecorder === "undefined") return "";
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

export async function beginRecording() {
  if (active) return;
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    throw new Error("this browser cannot record audio");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = preferredMime();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) chunks.push(event.data);
  });
  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      const type = recorder.mimeType || mimeType || "audio/webm";
      resolve(new Blob(chunks, { type }));
    });
    recorder.addEventListener("error", () => {
      stream.getTracks().forEach((track) => track.stop());
      reject(new Error("recording failed"));
    });
  });
  recorder.start();
  active = { recorder, stopped };
}

export function stopRecording() {
  if (!active) return Promise.reject(new Error("not recording"));
  const { recorder, stopped } = active;
  active = null;
  if (recorder.state !== "inactive") recorder.stop();
  return stopped;
}
