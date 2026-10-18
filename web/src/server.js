/** Where the phone sends prompts. Dev and the FastAPI build use their own host.
 *  The GitHub Pages build asks for the laptop's public URL and remembers it. */

const KEY = "recorder-server";

export function pagesMode() {
  return import.meta.env.VITE_PAGES === "1";
}

export function parseOrigin(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  try {
    const url = new URL(/^https?:\/\//i.test(text) ? text : `https://${text}`);
    if (url.protocol !== "http:" && url.protocol !== "https:") return "";
    return url.origin;
  } catch {
    return "";
  }
}

export function savedServer() {
  try {
    return localStorage.getItem(KEY) || "";
  } catch {
    return "";
  }
}

export function saveServer(value) {
  const origin = parseOrigin(value);
  try {
    if (origin) localStorage.setItem(KEY, origin);
    else if (!String(value || "").trim()) localStorage.removeItem(KEY);
  } catch {
    /* private mode */
  }
  return origin;
}

export function apiOrigin() {
  const fromEnv = import.meta.env.VITE_API_URL;
  if (fromEnv) return String(fromEnv).replace(/\/$/, "");
  if (pagesMode()) return savedServer();
  return "";
}

export function wsUrl() {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  const origin = apiOrigin();
  if (origin) {
    const url = new URL(origin);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = "/ws";
    url.search = "";
    url.hash = "";
    return url.toString();
  }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}
