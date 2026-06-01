import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HG = {
  sessions: new Map(),
  current: null,
  selected: new Set(),
  pollTimer: null,
};

function ensureStyles() {
  if (document.getElementById("humangate-style-link")) return;
  const link = document.createElement("link");
  link.id = "humangate-style-link";
  link.rel = "stylesheet";
  link.href = "extensions/ComfyUI-naginagi-vibe-lab/humangate.css";
  document.head.appendChild(link);
}

function ensureOverlay() {
  if (document.getElementById("humangate-overlay")) return;
  const root = document.createElement("div");
  root.id = "humangate-overlay";
  root.className = "hg-hidden";
  root.innerHTML = `
    <div class="hg-panel">
      <header class="hg-header">
        <div>
          <h2 id="hg-title">HumanGate</h2>
          <p id="hg-message">Waiting for input.</p>
        </div>
        <button id="hg-close" type="button" title="Close overlay">x</button>
      </header>
      <div id="hg-body" class="hg-body"></div>
      <footer class="hg-footer">
        <span id="hg-status"></span>
        <button id="hg-clear" type="button">Clear</button>
        <button id="hg-all" type="button">All</button>
        <button id="hg-stop" type="button">Stop</button>
        <button id="hg-resume" type="button" class="hg-primary">Resume</button>
      </footer>
    </div>`;
  document.body.appendChild(root);
  document.getElementById("hg-close").addEventListener("click", () => closeOverlay(false));
  document.getElementById("hg-clear").addEventListener("click", () => {
    HG.selected.clear();
    renderCurrent();
  });
  document.getElementById("hg-all").addEventListener("click", () => {
    const session = HG.current;
    if (!session) return;
    const count = session.payload?.batch_size ?? 0;
    HG.selected = new Set(Array.from({ length: count }, (_, i) => i));
    renderCurrent();
  });
  document.getElementById("hg-stop").addEventListener("click", () => sendDecision("stop"));
  document.getElementById("hg-resume").addEventListener("click", () => sendDecision("resume"));
  window.addEventListener("keydown", onKeyDown);
}

function onKeyDown(event) {
  if (!HG.current) return;
  if (event.key === "Enter") {
    event.preventDefault();
    sendDecision("resume");
  } else if (event.key === "Escape") {
    event.preventDefault();
    sendDecision("stop");
  } else if (event.key.toLowerCase() === "a") {
    const count = HG.current.payload?.batch_size ?? 0;
    HG.selected = new Set(Array.from({ length: count }, (_, i) => i));
    renderCurrent();
  } else if (event.key.toLowerCase() === "c") {
    HG.selected.clear();
    renderCurrent();
  } else if (/^[1-9]$/.test(event.key)) {
    const idx = Number(event.key) - 1;
    toggleIndex(idx);
  }
}

function openOverlay(session) {
  HG.current = session;
  HG.selected = new Set();
  const mode = session.payload?.selection_mode || "single";
  if (mode === "single") HG.selected.add(0);
  document.getElementById("humangate-overlay").classList.remove("hg-hidden");
  renderCurrent();
}

function closeOverlay(clearCurrent = true) {
  const root = document.getElementById("humangate-overlay");
  if (root) root.classList.add("hg-hidden");
  if (clearCurrent) HG.current = null;
}

function toggleIndex(idx) {
  const session = HG.current;
  if (!session) return;
  const count = session.payload?.batch_size ?? 0;
  if (idx < 0 || idx >= count) return;
  const mode = session.payload?.selection_mode || "single";
  if (mode === "single") {
    HG.selected = new Set([idx]);
  } else if (HG.selected.has(idx)) {
    HG.selected.delete(idx);
  } else {
    HG.selected.add(idx);
  }
  renderCurrent();
}

function renderCurrent() {
  const session = HG.current;
  if (!session) return;
  const payload = session.payload || {};
  document.getElementById("hg-title").textContent = session.kind || "HumanGate";
  document.getElementById("hg-message").textContent = payload.message || "Select and resume.";
  document.getElementById("hg-status").textContent = `${session.gate_id}`;
  document.getElementById("hg-all").disabled = payload.selection_mode !== "multiple";

  const body = document.getElementById("hg-body");
  body.innerHTML = "";
  const previews = payload.preview_urls || [];
  const labels = payload.labels || [];
  const texts = payload.texts || [];
  const count = payload.batch_size || Math.max(previews.length, texts.length, 1);

  if (previews.length) {
    const grid = document.createElement("div");
    grid.className = "hg-grid";
    for (let i = 0; i < count; i += 1) {
      const tile = document.createElement("button");
      tile.className = "hg-tile" + (HG.selected.has(i) ? " hg-selected" : "");
      tile.type = "button";
      tile.addEventListener("click", () => toggleIndex(i));
      const img = document.createElement("img");
      img.src = previews[i] || "";
      img.alt = labels[i] || `Image ${i + 1}`;
      const cap = document.createElement("span");
      cap.textContent = labels[i] || `${i + 1}`;
      tile.appendChild(img);
      tile.appendChild(cap);
      grid.appendChild(tile);
    }
    body.appendChild(grid);
  } else if (texts.length) {
    const list = document.createElement("div");
    list.className = "hg-text-list";
    for (let i = 0; i < texts.length; i += 1) {
      const item = document.createElement("button");
      item.className = "hg-text-item" + (HG.selected.has(i) ? " hg-selected" : "");
      item.type = "button";
      item.textContent = `${labels[i] || i + 1}: ${texts[i]}`;
      item.addEventListener("click", () => toggleIndex(i));
      list.appendChild(item);
    }
    body.appendChild(list);
  } else {
    const p = document.createElement("p");
    p.textContent = "Paused. Choose Resume or Stop.";
    body.appendChild(p);
  }
}

async function sendDecision(decision) {
  const session = HG.current;
  if (!session) return;
  const selected = Array.from(HG.selected).sort((a, b) => a - b);
  await api.fetchApi("/humangate/respond", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gate_id: session.gate_id,
      result: {
        decision,
        selected_indices: selected.length ? selected : [0],
      },
    }),
  });
  HG.sessions.delete(session.gate_id);
  closeOverlay(true);
}

async function pollSessions() {
  try {
    const res = await api.fetchApi("/humangate/sessions");
    const data = await res.json();
    const sessions = data.sessions || [];
    for (const session of sessions) {
      if (!HG.sessions.has(session.gate_id)) {
        HG.sessions.set(session.gate_id, session);
        if (!HG.current) openOverlay(session);
      }
    }
  } catch (err) {
    // Route may not be registered during startup. Keep polling.
  }
}

app.registerExtension({
  name: "ComfyUI.HumanGate",
  async setup() {
    ensureStyles();
    ensureOverlay();
    if (!HG.pollTimer) HG.pollTimer = setInterval(pollSessions, 750);
  },
});
