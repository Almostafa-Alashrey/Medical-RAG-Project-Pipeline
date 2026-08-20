const historyList = document.getElementById("historyList");
const filterTabs = document.getElementById("filterTabs");
const searchInput = document.getElementById("searchInput");

let currentStatus = "all";
let searchDebounce = null;

function statusBadgeHtml(slug) {
  const label = statusLabel(slug);
  return `<span class="status-badge ${slug}">${label}</span>`;
}

function historyIcon() {
  return `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>`;
}

function renderRows(items) {
  if (!items.length) {
    historyList.innerHTML = `<div class="history-empty">No questions yet — ask something on the Chat page.</div>`;
    return;
  }
  historyList.innerHTML = items.map(item => `
    <div class="history-row">
      <div class="history-icon">${historyIcon()}</div>
      <div class="history-main">
        <div class="q">${escapeHtml(item.question)}</div>
        <div class="meta">${escapeHtml(item.day_label || "")} · ${item.sources_count || 0} sources cited</div>
      </div>
      <div class="history-status">${statusBadgeHtml(item.status_slug)}</div>
    </div>
  `).join("");
}

async function loadHistory() {
  historyList.innerHTML = `<div class="history-empty">Loading…</div>`;
  const params = new URLSearchParams();
  if (currentStatus !== "all") params.set("status", currentStatus);
  const q = searchInput.value.trim();
  if (q) params.set("q", q);

  try {
    const data = await apiGet(`/api/history?${params.toString()}`);
    renderRows(data.items || []);
  } catch (err) {
    historyList.innerHTML = `<div class="history-empty">Couldn't load history — is the backend running on ${API_BASE}?</div>`;
    console.error(err);
  }
}

filterTabs.addEventListener("click", (e) => {
  const btn = e.target.closest(".filter-tab");
  if (!btn) return;
  filterTabs.querySelectorAll(".filter-tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentStatus = btn.dataset.status;
  loadHistory();
});

searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(loadHistory, 300);
});

loadHistory();