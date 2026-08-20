// Central place to point the frontend at the Flask backend.
// Override by setting `window.PRESSIQ_API_BASE` before this script loads
// (e.g. in production, inject the real API URL).
const API_BASE = window.PRESSIQ_API_BASE || "http://localhost:5000";

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Very small markdown-ish renderer for the LLM's **bold** / bullet output,
// so answers look right without pulling in a full markdown dependency.
function renderAnswer(text) {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n- /g, "\n• ")
    .replace(/\n/g, "<br>");
}

function statusLabel(slug) {
  return { allowed: "Allowed", needs_caution: "Needs caution", refused: "Refused" }[slug] || slug;
}