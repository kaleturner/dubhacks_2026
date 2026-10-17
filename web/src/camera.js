/** Rear camera stills and recording. Frontend fills these in. */

export async function openRearCamera() {
  // getUserMedia: rear camera, 1920x1080, 30fps. Lock exposure and focus where supported.
  // Request a Wake Lock so the phone stays awake for the whole job.
  throw new Error("camera is not wired yet");
}

export function snapshot() {
  // Draw the video element to a canvas and return a jpeg Blob.
  throw new Error("snapshot is not wired yet");
}
