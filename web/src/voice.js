/** One tap listens. The browser's recognizer fills the prompt when it can;
 *  otherwise the recording is posted to /transcribe. */

const SPEECH_RMS = 0.02;
const SILENCE_MS = 800;
const MAX_MS = 12000;
const START_GRACE_MS = 4000;
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];

export function hasSpeechRecognition() {
  return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
}

function preferredMime() {
  if (typeof MediaRecorder === "undefined") return "";
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

export function listenForPrompt() {
  let halt = () => {};
  const done = new Promise((resolve, reject) => {
    let settled = false;
    let stopping = false;
    let recorder = null;
    let stream = null;
    let audioCtx = null;
    let timer = null;
    let recognition = null;
    let micFailed = null;
    const chunks = [];

    function blob() {
      const type = recorder?.mimeType || preferredMime() || "audio/webm";
      return new Blob(chunks, { type });
    }

    function settle(value, error) {
      if (settled) return;
      settled = true;
      if (timer) clearInterval(timer);
      if (recognition) {
        recognition.onresult = null;
        recognition.onerror = null;
        recognition.onend = null;
        try {
          recognition.abort();
        } catch {
          /* already stopped */
        }
      }
      if (audioCtx) audioCtx.close().catch(() => {});
      stream?.getTracks().forEach((track) => track.stop());
      if (error) reject(error);
      else resolve(value);
    }

    function stopAndResolveBlob() {
      if (settled || stopping) return;
      stopping = true;
      if (!recorder || recorder.state === "inactive") {
        settle({ blob: blob() });
        return;
      }
      recorder.addEventListener("stop", () => settle({ blob: blob() }), { once: true });
      try {
        recorder.stop();
      } catch {
        settle({ blob: blob() });
      }
    }

    halt = () => stopAndResolveBlob();

    const canRecord =
      Boolean(navigator.mediaDevices?.getUserMedia) && typeof MediaRecorder !== "undefined";
    if (!canRecord && !hasSpeechRecognition()) {
      settle(null, new Error("this browser cannot record audio"));
      return;
    }

    if (hasSpeechRecognition()) {
      const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new Recognition();
      recognition.lang = "en-US";
      recognition.onresult = (event) => {
        const text = event.results?.[0]?.[0]?.transcript?.trim() || "";
        if (!text) return;
        stopping = true;
        if (recorder && recorder.state !== "inactive") {
          try {
            recorder.stop();
          } catch {
            /* the transcript is enough */
          }
        }
        settle({ text });
      };
      recognition.onerror = () => {
        recognition = null;
        if (micFailed) settle(null, micFailed);
      };
      recognition.onend = () => {
        recognition = null;
        if (micFailed) settle(null, micFailed);
      };
      try {
        recognition.start();
      } catch {
        recognition = null;
      }
    }

    if (!canRecord) return;

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((mic) => {
        if (settled) {
          mic.getTracks().forEach((track) => track.stop());
          return;
        }
        stream = mic;
        const mimeType = preferredMime();
        try {
          recorder = new MediaRecorder(mic, mimeType ? { mimeType } : undefined);
        } catch {
          recorder = new MediaRecorder(mic);
        }
        recorder.addEventListener("dataavailable", (event) => {
          if (event.data.size) chunks.push(event.data);
        });
        recorder.start();
        watchSilence(mic);
      })
      .catch((error) => {
        micFailed = new Error(error?.message || "microphone blocked");
        if (!recognition) settle(null, micFailed);
      });

    function watchSilence(mic) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      audioCtx = new Ctx();
      audioCtx.resume().catch(() => {});
      const source = audioCtx.createMediaStreamSource(mic);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      const samples = new Float32Array(analyser.fftSize);
      const started = performance.now();
      let heard = false;
      let quietAt = 0;
      timer = setInterval(() => {
        if (settled) return;
        analyser.getFloatTimeDomainData(samples);
        let sum = 0;
        for (const sample of samples) sum += sample * sample;
        const rms = Math.sqrt(sum / samples.length);
        const now = performance.now();
        if (rms >= SPEECH_RMS) {
          heard = true;
          quietAt = 0;
        } else if (heard) {
          quietAt = quietAt || now;
        }
        const quietFor = quietAt ? now - quietAt : 0;
        const waitingOnBrowser = Boolean(recognition);
        if (!waitingOnBrowser && heard && quietFor >= SILENCE_MS) halt();
        if (!waitingOnBrowser && !heard && now - started >= START_GRACE_MS) halt();
        if (now - started >= MAX_MS) halt();
      }, 100);
    }
  });

  return { done, stop: () => halt() };
}
