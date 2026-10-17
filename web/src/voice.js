/** webkitSpeechRecognition, with POST /transcribe as the fallback. */

export function listenOnce() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    return Promise.reject(new Error("no speech recognition; use the text box or record audio"));
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
