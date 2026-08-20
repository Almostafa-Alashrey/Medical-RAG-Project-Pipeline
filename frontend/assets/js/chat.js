const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const newChatBtn = document.getElementById("newChatBtn");

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="msg-user-bubble">${escapeHtml(text)}</div>`;
  chatWindow.appendChild(row);
  scrollToBottom();
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.id = "typingRow";
  row.innerHTML = `
    <div class="bot-block">
      <div class="bot-card"><span class="typing-dots"><span></span><span></span><span></span></span></div>
    </div>`;
  chatWindow.appendChild(row);
  scrollToBottom();
}

function removeTypingIndicator() {
  const row = document.getElementById("typingRow");
  if (row) row.remove();
}

function addBotMessage(result) {
  const slug = result.status_slug || (result.status || "").toLowerCase().replace(" ", "_");
  const label = statusLabel(slug);
  const confidence = result.confidence && typeof result.confidence.score === "number"
    ? ` · confidence ${(result.confidence.score / 100).toFixed(2)}`
    : "";

  const row = document.createElement("div");
  row.className = "msg-row bot";

  if (slug === "refused") {
    row.innerHTML = `
      <div class="bot-block">
        <span class="status-badge refused">⚠ Refused · out of guideline scope</span>
        <div class="bot-card refused-card">
          <p class="bot-text">${escapeHtml(result.answer)}</p>
        </div>
      </div>`;
  } else {
    const sectionTitle = result.section_title || "Guideline reference";
    const chips = (result.sources || [])
      .map(s => `<span class="citation-chip">${escapeHtml(s.chunk_id)} · p.${escapeHtml((s.page_numbers || []).join("-"))}</span>`)
      .join("");
    row.innerHTML = `
      <div class="bot-block">
        <span class="status-badge ${slug}">✓ ${label}${confidence}</span>
        <div class="bot-card">
          <h5>${escapeHtml(sectionTitle)}</h5>
          <p class="bot-text">${renderAnswer(result.answer)}</p>
          <div class="citation-chips">${chips}</div>
        </div>
      </div>`;
  }
  chatWindow.appendChild(row);
  scrollToBottom();
}

async function askQuestion(question) {
  addUserMessage(question);
  addTypingIndicator();
  chatInput.value = "";
  try {
    const result = await apiPost("/api/chat", { question });
    removeTypingIndicator();
    addBotMessage(result);
  } catch (err) {
    removeTypingIndicator();
    addBotMessage({
      status: "Error",
      status_slug: "refused",
      answer: "Sorry — I couldn't reach the PressIQ backend. Make sure the Flask API is running on " + API_BASE + ".",
    });
    console.error(err);
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if (q) askQuestion(q);
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => askQuestion(chip.dataset.q));
});

newChatBtn.addEventListener("click", () => {
  chatWindow.innerHTML = "";
});

// Seed with a friendly opening prompt, mirroring the product screenshot.
addBotMessage({
  status: "Allowed",
  status_slug: "allowed",
  confidence: { score: 100 },
  section_title: "Welcome",
  answer: "Hi! Ask me anything about the WHO guideline for the pharmacological treatment of hypertension in adults — I'll answer with page-level citations, or tell you honestly when the evidence isn't strong enough.",
  sources: [],
});