/* ===================================================================
   EdgeGD web UI
   Hash-routed single page: upload -> job progress -> report.
   The report is built entirely from the pipeline's result JSON, so it
   works for any speaker count, not just the two-person case.
   =================================================================== */

"use strict";

const view = document.getElementById("view");

/* ---------------- small helpers ---------------- */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/** Seconds -> m:ss.s, for timecodes inside a session. */
function clock(s) {
  const total = Math.max(0, Number(s) || 0);
  const m = Math.floor(total / 60);
  const r = total - m * 60;
  return `${m}:${r < 10 ? "0" : ""}${r.toFixed(1)}`;
}

/** Seconds -> "1h 02m" / "6m 12s" / "48s", for durations. */
function human(s) {
  const t = Math.round(Number(s) || 0);
  if (t >= 3600) return `${Math.floor(t / 3600)}h ${String(Math.floor((t % 3600) / 60)).padStart(2, "0")}m`;
  if (t >= 60) return `${Math.floor(t / 60)}m ${String(t % 60).padStart(2, "0")}s`;
  return `${t}s`;
}

function mmss(s) {
  const t = Math.max(0, Math.round(Number(s) || 0));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}

function bytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

function showError(node, message) {
  node.hidden = false;
  node.innerHTML = `<b>Couldn't do that.</b> ${esc(message)}`;
}

/* ---------------- speaker colour assignment ----------------
   Slot is bound to the speaker label (sorted), never to rank, so
   re-sorting a table never repaints a speaker. Nine or more speakers
   fold into a neutral slot rather than inventing a hue.            */

const MAX_SLOTS = 8;

function slotMap(ids) {
  const sorted = [...ids].sort();
  const map = new Map();
  sorted.forEach((id, i) => map.set(id, i < MAX_SLOTS ? i + 1 : 0));
  return map;
}

function swatch(slot) {
  return `<span class="swatch sw-${slot}"></span>`;
}

function seriesColor(slot) {
  return slot === 0 ? "var(--rule)" : `var(--s${slot})`;
}

/** Short display label: SPEAKER_00 -> S00, anything else passes through. */
function shortName(id) {
  const m = /^SPEAKER_?(\d+)$/i.exec(id);
  return m ? `S${m[1]}` : id;
}

/* ---------------- tooltip ---------------- */

const Tip = (() => {
  const node = document.getElementById("tip");
  let on = false;

  function move(ev) {
    if (!on) return;
    const pad = 14;
    const r = node.getBoundingClientRect();
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
    node.style.left = `${Math.max(8, x)}px`;
    node.style.top = `${Math.max(8, y)}px`;
  }
  function show(html, ev) {
    node.innerHTML = html;
    node.style.opacity = "1";
    node.setAttribute("aria-hidden", "false");
    on = true;
    move(ev);
  }
  function hide() {
    node.style.opacity = "0";
    node.setAttribute("aria-hidden", "true");
    on = false;
  }
  function bind(target, htmlFn) {
    target.addEventListener("mouseenter", (ev) => show(htmlFn(), ev));
    target.addEventListener("mousemove", move);
    target.addEventListener("mouseleave", hide);
  }
  return { show, hide, bind };
})();

/* ===================================================================
   VIEW: UPLOAD
   =================================================================== */

/* Measured on an M-series CPU with Whisper `base`: a 3-minute clip took about
   3 minutes end to end, diarization being the bulk of it. Roughly real time.
   Used only for a rough hint - a GPU box will be far quicker. */
const RUNTIME_RATIO = 1.0;

function renderUpload() {
  const frag = document.getElementById("tpl-upload").content.cloneNode(true);
  view.replaceChildren(frag);

  const form = $("#upload-form");
  const drop = $("#drop");
  const input = $("#file");
  const idle = $(".drop-idle");
  const chosen = $(".drop-file");
  const submit = $("#submit");
  const hint = $("#runtime-hint");
  const errBox = $("#upload-error");

  function setFile(file) {
    if (!file) return;
    input.files = fileListOf(file);
    $("#drop-name").textContent = file.name;
    $("#drop-meta").textContent = `${bytes(file.size)}${file.type ? ` · ${file.type}` : ""}`;
    idle.hidden = true;
    chosen.hidden = false;
    drop.classList.add("has-file");
    submit.disabled = false;
    errBox.hidden = true;
    estimate(file);
  }

  function clearFile() {
    input.value = "";
    idle.hidden = false;
    chosen.hidden = true;
    drop.classList.remove("has-file");
    submit.disabled = true;
    hint.textContent = "";
  }

  /* Read the media length in the browser purely to show a runtime hint.
     Formats Chrome can't decode (some mkv) simply get no hint. */
  function estimate(file) {
    const probe = document.createElement("video");
    const url = URL.createObjectURL(file);
    probe.preload = "metadata";
    probe.src = url;
    probe.addEventListener("loadedmetadata", () => {
      const dur = probe.duration;
      URL.revokeObjectURL(url);
      if (!isFinite(dur) || dur <= 0) return;
      const limit = parseFloat($("#limit").value);
      const effective = limit > 0 ? Math.min(dur, limit * 60) : dur;
      hint.textContent =
        `${human(dur)} of audio${effective < dur ? `, analysing the first ${human(effective)}` : ""} — ` +
        `on CPU expect roughly ${human(effective * RUNTIME_RATIO)} of processing.`;
    });
    probe.addEventListener("error", () => URL.revokeObjectURL(url));
  }

  function fileListOf(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    return dt.files;
  }

  drop.addEventListener("click", () => { if (!drop.classList.contains("has-file")) input.click(); });
  drop.addEventListener("keydown", (ev) => {
    if ((ev.key === "Enter" || ev.key === " ") && !drop.classList.contains("has-file")) {
      ev.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => setFile(input.files[0]));
  $("#drop-clear").addEventListener("click", (ev) => { ev.stopPropagation(); clearFile(); input.click(); });

  ["dragenter", "dragover"].forEach((t) =>
    drop.addEventListener(t, (ev) => { ev.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) =>
    drop.addEventListener(t, () => drop.classList.remove("is-over")));
  drop.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const file = ev.dataTransfer.files[0];
    if (file) setFile(file);
  });

  $("#limit").addEventListener("change", () => { if (input.files[0]) estimate(input.files[0]); });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!input.files[0]) return;

    submit.disabled = true;
    submit.textContent = "Uploading…";
    errBox.hidden = true;

    const body = new FormData();
    body.append("audio", input.files[0]);
    body.append("topic", $("#topic").value || "Group Discussion");
    if ($("#speakers").value) body.append("num_speakers", $("#speakers").value);
    if ($("#limit").value) body.append("limit_minutes", $("#limit").value);

    try {
      const job = await api("/api/analyze", { method: "POST", body });
      location.hash = `#/job/${job.job_id}`;
    } catch (e) {
      showError(errBox, e.message);
      submit.disabled = false;
      submit.textContent = "Run analysis";
    }
  });

  loadRecent();
}

async function loadRecent() {
  try {
    const { sessions } = await api("/api/sessions");
    if (!sessions.length) return;
    const panel = $("#recent-panel");
    panel.hidden = false;
    renderSessionRows($("#recent-list"), sessions.slice(0, 5));
  } catch (_) { /* the panel just stays hidden */ }
}

/* ===================================================================
   VIEW: JOB PROGRESS
   =================================================================== */

let pollTimer = null;

function renderJob(jobId) {
  const frag = document.getElementById("tpl-progress").content.cloneNode(true);
  view.replaceChildren(frag);

  const fill = $("#run-fill");
  const errBox = $("#run-error");
  const actions = $("#run-actions");
  let ticker = null;

  function stop() {
    clearTimeout(pollTimer);
    clearInterval(ticker);
    pollTimer = null;
  }

  async function poll() {
    let job;
    try {
      job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    } catch (e) {
      stop();
      showError(errBox, `${e.message}. If the server restarted, the job list was lost — the finished report may still be under Sessions.`);
      return;
    }

    $("#run-file").textContent = job.filename;
    $("#run-phase").textContent = job.phase;
    $("#run-pct").textContent = `${job.progress}%`;
    fill.style.width = `${job.progress}%`;
    $("#run-topic").textContent = job.topic;
    $("#run-speakers").textContent = job.num_speakers ? String(job.num_speakers) : "auto-detect";

    const dur = job.analysed_duration || job.source_duration;
    $("#run-duration").textContent = dur
      ? human(dur) + (job.limit_sec ? ` (of ${human(job.source_duration)})` : "")
      : "—";

    if (!ticker) {
      ticker = setInterval(() => {
        const from = job.started_at || job.created_at;
        const to = job.finished_at || Date.now() / 1000;
        $("#run-elapsed").textContent = mmss(to - from);
      }, 500);
    }

    if (job.status === "done") {
      stop();
      fill.classList.add("is-done");
      $("#run-status").textContent = "Complete";
      location.hash = `#/session/${job.session_id}`;
      return;
    }

    if (job.status === "error") {
      stop();
      fill.classList.add("is-error");
      fill.style.width = "100%";
      $("#run-status").textContent = "Failed";
      $("#run-note").hidden = true;
      showError(errBox, job.error || "The analyser stopped without a reason.");
      actions.hidden = false;
      const again = el("a", "primary", "Try another file");
      again.href = "#/";
      again.style.textDecoration = "none";
      actions.replaceChildren(again);
      return;
    }

    pollTimer = setTimeout(poll, 1200);
  }

  poll();
}

/* ===================================================================
   VIEW: SESSION LIST
   =================================================================== */

async function renderSessions() {
  const frag = document.getElementById("tpl-sessions").content.cloneNode(true);
  view.replaceChildren(frag);
  const list = $("#sessions-list");
  try {
    const { sessions } = await api("/api/sessions");
    renderSessionRows(list, sessions, true);
  } catch (e) {
    list.replaceChildren(el("p", "empty", `Couldn't load sessions: ${e.message}`));
  }
}

function renderSessionRows(container, sessions, withDelete = false) {
  container.replaceChildren();

  if (!sessions.length) {
    container.appendChild(el("p", "empty", "Nothing analysed yet."));
    return;
  }

  sessions.forEach((s) => {
    const row = el("div", "session-row");

    const main = el("div", "session-main");
    const link = el("a", null, s.topic || "Group Discussion");
    link.href = `#/session/${s.session_id}`;
    const sub = el("div", "session-sub",
      [s.session_id, s.date, `${s.participants} speakers`, human(s.duration_sec),
       s.filename].filter(Boolean).join("  ·  "));
    main.append(link, sub);

    const score = el("div", "session-score");
    score.append(el("span", "v", s.cohesion_score), el("span", "k", s.cohesion_label));

    const actions = el("div", "session-actions");
    const open = el("a", "ghost", "Open");
    open.href = `#/session/${s.session_id}`;
    const pdf = el("a", "ghost", "PDF");
    pdf.href = `/api/sessions/${s.session_id}/report.pdf`;
    pdf.target = "_blank";
    pdf.rel = "noopener";
    actions.append(open, pdf);

    if (withDelete) {
      const del = el("button", "ghost", "Delete");
      del.type = "button";
      del.addEventListener("click", async () => {
        del.disabled = true;
        del.textContent = "Deleting…";
        try {
          await api(`/api/sessions/${s.session_id}`, { method: "DELETE" });
          row.remove();
          if (!container.children.length) {
            container.appendChild(el("p", "empty", "Nothing analysed yet."));
          }
        } catch (e) {
          del.disabled = false;
          del.textContent = "Delete failed";
        }
      });
      actions.append(del);
    }

    row.append(main, score, actions);
    container.appendChild(row);
  });
}

/* ===================================================================
   VIEW: REPORT
   =================================================================== */

async function renderSession(sessionId) {
  view.replaceChildren(el("div", "loading", "Loading report…"));
  let data;
  try {
    data = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
  } catch (e) {
    const box = el("p", "error");
    showError(box, e.message);
    view.replaceChildren(box);
    return;
  }
  view.replaceChildren(buildReport(data));
}

const DEFAULT_CFG = {
  silent_pct: 10, moderate_pct: 20, dominant_pct: 40,
  pace_slow_wpm: 110, pace_fast_wpm: 160, long_pause_sec: 2.0,
};

function buildReport(data) {
  const cfg = Object.assign({}, DEFAULT_CFG, data.config || {});
  const frag = document.createDocumentFragment();

  const ids = Object.keys(data.participants);
  const slots = slotMap(ids);
  const byShare = [...ids].sort(
    (a, b) => data.participants[b].participation_pct - data.participants[a].participation_pct);
  const duration = data.total_duration_sec || 0;
  const P = (id) => data.participants[id];

  frag.append(
    reportHead(data),
    cohesionPanel(data, byShare),
    timelinePanel(data, slots, duration),
    splitPanel(data, byShare, slots, duration, cfg),
    pacePanel(data, byShare, slots, cfg),
    speakerCards(data, byShare, slots, cfg),
    dynamicsPanel(data, slots, ids),
    topicsPanel(data),
    transcriptPanel(data, slots, ids),
  );

  const artifacts = artifactsPanel(data);
  if (artifacts) frag.append(artifacts);

  return frag;
}

/* ---------------- report: header ---------------- */

function reportHead(data) {
  const head = el("header", "report-head");
  const source = data.source || {};

  const left = el("div");
  left.append(el("p", "eyebrow", `Session ${data.session_id}`));
  const h1 = el("h1", null, data.topic || "Group Discussion");
  left.append(h1);

  const actions = el("div", "report-actions");
  actions.style.marginTop = "10px";
  const pdf = el("a", "ghost", "Download PDF report");
  pdf.href = `/api/sessions/${data.session_id}/report.pdf`;
  pdf.target = "_blank";
  pdf.rel = "noopener";
  const again = el("a", "ghost", "Analyse another recording");
  again.href = "#/";
  actions.append(pdf, again);
  left.append(actions);

  const meta = el("div", "report-meta");
  const rows = [
    ["analysed", data.date],
    ["length", human(data.total_duration_sec)],
    ["speakers", String(Object.keys(data.participants).length)],
  ];
  if (source.original_filename) rows.unshift(["source", source.original_filename]);
  if (source.limit_sec && source.source_duration_sec) {
    rows.push(["clipped from", human(source.source_duration_sec)]);
  }
  rows.forEach(([k, v]) => { meta.append(el("b", null, k), el("span", null, v)); });

  head.append(left, meta);
  return head;
}

/* ---------------- report: cohesion ---------------- */

function cohesionPanel(data, byShare) {
  const panel = el("section", "panel");
  panel.append(panelHead("Phases 6–7 · group metrics", "Group cohesion"));

  const scores = el("div", "scores");

  const label = data.group_cohesion_label;
  const tone = { Excellent: "is-good", Good: "is-good", Average: "is-warning", Poor: "is-critical" }[label] || "";
  const band = { Excellent: "85–100", Good: "70–84", Average: "50–69", Poor: "below 50" }[label] || "";

  const hero = el("div", "hero-score");
  hero.innerHTML =
    `<div><span class="val">${esc(data.group_cohesion_score)}</span><span class="of"> / 100</span></div>
     <div class="label"><span class="pill ${tone}">${esc(label)}</span>
     <span class="note">band ${esc(band)}</span></div>`;

  const meanComm = byShare.length
    ? byShare.reduce((sum, id) => sum + data.participants[id].communication_score, 0) / byShare.length
    : 0;

  const meters = el("div", "meters");
  [
    ["Participation balance", data.participation_balance_score],
    ["Turn taking", data.turn_taking_score],
    ["Mean communication score", meanComm],
  ].forEach(([name, value]) => {
    const v = Number(value) || 0;
    const m = el("div", "meter");
    m.innerHTML =
      `<span class="name">${esc(name)}</span>
       <span class="v">${v.toFixed(2)}</span>
       <div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, v))}%"></div></div>`;
    meters.append(m);
  });

  scores.append(hero, meters);
  panel.append(scores);

  const overlapSec = (data.overlaps || []).reduce((s, o) => s + o.duration, 0);
  panel.append(el("p", "note",
    `Cohesion blends the three components above (weighted 0.35 / 0.30 / 0.35) and then subtracts an ` +
    `overlap penalty. ${overlapSec > 0
      ? `${overlapSec.toFixed(1)}s of simultaneous speech was detected, so a penalty applies.`
      : `No simultaneous speech was detected, so the penalty is zero.`}`));

  return panel;
}

/* ---------------- report: timeline ---------------- */

function timelinePanel(data, slots, duration) {
  const panel = el("section", "panel");
  panel.append(panelHead("Phase 1 · speaker-attributed transcript", "Who spoke when"));

  const lanes = [...slots.keys()].sort();
  const scroll = el("div", "tl-scroll");
  const tl = el("div", "tl");

  /* Ruler: aim for 6-10 labelled ticks whatever the recording length. */
  const steps = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
  const step = steps.find((s) => duration / s <= 10) || 3600;
  const ruler = el("div", "tl-ruler");
  for (let t = 0; t <= duration; t += step) {
    const pct = (t / duration) * 100;
    const tick = el("i");
    tick.style.left = `${pct}%`;
    const lab = el("b", null, mmss(t));
    lab.style.left = `${pct}%`;
    // Keep the end labels inside the track instead of half-clipping them.
    if (pct > 96) lab.style.transform = "translateX(-100%)";
    else if (pct < 2) lab.style.transform = "translateX(0)";
    ruler.append(tick, lab);
  }
  tl.append(ruler);

  const laneTracks = new Map();
  lanes.forEach((id) => {
    const row = el("div", "tl-lane");
    const name = el("div", "tl-name");
    name.innerHTML = `${swatch(slots.get(id))}${esc(id)}`;
    const track = el("div", "tl-track");
    laneTracks.set(id, track);
    row.append(name, track);
    tl.append(row);
  });

  (data.transcript || []).forEach((seg) => {
    const track = laneTracks.get(seg.speaker);
    if (!track) return;
    const slot = slots.get(seg.speaker);
    const box = el("div", "tl-seg");
    box.style.left = `${(seg.start / duration) * 100}%`;
    box.style.width = `${Math.max(((seg.end - seg.start) / duration) * 100, 0.25)}%`;
    box.style.background = seriesColor(slot);
    Tip.bind(box, () => {
      const text = seg.text.length > 200 ? `${seg.text.slice(0, 200)}…` : seg.text;
      return `<div class="t-head">${swatch(slot)}${esc(seg.speaker)} · ` +
             `${clock(seg.start)}–${clock(seg.end)} · ${(seg.end - seg.start).toFixed(2)}s</div>${esc(text)}`;
    });
    track.append(box);
  });

  scroll.append(tl);
  panel.append(scroll);
  panel.append(el("p", "note",
    `${(data.transcript || []).length} segments, produced by aligning Whisper word timestamps to ` +
    `pyannote diarization turns. Hover any block to read it.`));
  return panel;
}

/* ---------------- report: participation split ---------------- */

function splitPanel(data, byShare, slots, duration, cfg) {
  const panel = el("section", "panel");
  panel.append(panelHead("Phase 2 · participation metrics", "Speaking split"));

  const shares = byShare.map((id) => data.participants[id].participation_pct);
  const spoken = shares.reduce((a, b) => a + b, 0);
  const silence = Math.max(0, 100 - spoken);
  const scale = spoken > 100 ? 100 / spoken : 1;   // overlapping speech can exceed 100%

  const bar = el("div", "split");
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label",
    byShare.map((id) => `${id} ${data.participants[id].participation_pct}%`).join(", "));

  byShare.forEach((id) => {
    const seg = el("div");
    seg.style.width = `${data.participants[id].participation_pct * scale}%`;
    seg.style.background = seriesColor(slots.get(id));
    Tip.bind(seg, () =>
      `<div class="t-head">${swatch(slots.get(id))}${esc(id)}</div>` +
      `${data.participants[id].participation_pct}% · ${data.participants[id].speaking_time_sec.toFixed(2)}s · ` +
      `${esc(data.participants[id].classification)}`);
    bar.append(seg);
  });
  if (silence > 0.05) {
    const gap = el("div");
    gap.style.width = `${silence}%`;
    gap.style.background = "var(--grid)";
    Tip.bind(gap, () => `<div class="t-head">Unattributed</div>${silence.toFixed(2)}% — silence, ` +
      `crosstalk, or audio no speaker was credited with.`);
    bar.append(gap);
  }
  panel.append(bar);

  const labels = el("div", "split-labels");
  byShare.forEach((id) => {
    const p = data.participants[id];
    const s = el("span");
    s.innerHTML = `${swatch(slots.get(id))}${esc(id)} <b>${p.participation_pct}%</b> ` +
      `<span class="num">${p.speaking_time_sec.toFixed(1)}s</span>`;
    labels.append(s);
  });
  if (silence > 0.05) {
    const s = el("span");
    s.innerHTML = `<span class="swatch sw-0"></span>Silence / unattributed <b>${silence.toFixed(2)}%</b> ` +
      `<span class="num">${(duration * silence / 100).toFixed(1)}s</span>`;
    labels.append(s);
  }
  panel.append(labels);

  const dominant = byShare.filter((id) => data.participants[id].classification === "Dominant");
  const silent = byShare.filter((id) => data.participants[id].classification === "Silent");
  const parts = [
    `Above ${cfg.dominant_pct}% of the session a speaker is flagged Dominant; below ${cfg.silent_pct}%, Silent.`,
  ];
  if (dominant.length) parts.push(`Dominant: ${dominant.join(", ")}.`);
  if (silent.length) parts.push(`Barely spoke: ${silent.join(", ")}.`);
  if (!dominant.length && !silent.length) parts.push("No speaker hit either extreme.");
  if (spoken > 100) {
    parts.push("Shares sum above 100% because overlapping speech is credited to both speakers.");
  }
  panel.append(el("p", "note", parts.join(" ")));

  return panel;
}

/* ---------------- report: pace ---------------- */

function pacePanel(data, byShare, slots, cfg) {
  const panel = el("section", "panel");
  panel.append(panelHead("Phase 4 · communication quality", "Speaking pace"));

  const wpms = byShare.map((id) => data.participants[id].avg_wpm || 0);
  const max = Math.max(cfg.pace_fast_wpm + 60, ...wpms);
  const axisMax = Math.ceil(max / 50) * 50;
  const pct = (v) => (v / axisMax) * 100;

  const wrap = el("div", "pace");

  byShare.forEach((id) => {
    const p = data.participants[id];
    const row = el("div", "pace-row");

    const name = el("div", "pace-name");
    name.innerHTML = `${swatch(slots.get(id))}${esc(id)}`;

    const plot = el("div", "pace-plot");
    const band = el("div", "pace-band");
    band.style.left = `${pct(cfg.pace_slow_wpm)}%`;
    band.style.width = `${pct(cfg.pace_fast_wpm - cfg.pace_slow_wpm)}%`;
    const bar = el("div", "pace-bar");
    bar.style.width = `${pct(p.avg_wpm)}%`;
    bar.style.background = seriesColor(slots.get(id));
    Tip.bind(bar, () =>
      `<div class="t-head">${swatch(slots.get(id))}${esc(id)}</div>` +
      `${p.avg_wpm} WPM — ${p.word_count} words / ${p.speaking_time_sec.toFixed(2)}s<br>` +
      `Labelled ${esc(p.pace_label)} against the ${cfg.pace_slow_wpm}–${cfg.pace_fast_wpm} normal band.`);
    plot.append(band, bar);

    row.append(name, plot, el("div", "pace-num", (p.avg_wpm || 0).toFixed(1)));
    wrap.append(row);
  });

  const axisRow = el("div", "pace-row");
  const axis = el("div", "pace-axis");
  for (let t = 0; t <= axisMax; t += axisMax / 4) {
    const b = el("b", null, String(Math.round(t)));
    b.style.left = `${pct(t)}%`;
    axis.append(b);
  }
  axisRow.append(el("div"), axis, el("div"));
  wrap.append(axisRow);

  const noteRow = el("div", "pace-row");
  noteRow.append(el("div"),
    el("p", "note", `Words per minute. The shaded lane is the configured normal range, ` +
      `${cfg.pace_slow_wpm}–${cfg.pace_fast_wpm} WPM.`),
    el("div"));
  wrap.append(noteRow);

  panel.append(wrap);

  /* A rate this high is arithmetic, not delivery - worth saying so. */
  const implausible = byShare.filter((id) => (data.participants[id].avg_wpm || 0) > 220);
  if (implausible.length) {
    panel.append(el("p", "note",
      `${implausible.join(", ")} ${implausible.length === 1 ? "exceeds" : "exceed"} 220 WPM, which is ` +
      `above sustainable human speech. Word counts come from the Whisper text while the denominator is ` +
      `diarization-attributed time, so speech credited to the wrong lane inflates the rate. Treat these ` +
      `figures as a data-quality signal rather than a verdict on the speaker.`));
  }

  return panel;
}

/* ---------------- report: per-speaker cards ---------------- */

function speakerCards(data, byShare, slots, cfg) {
  const section = el("section");
  const cards = el("div", "cards");

  byShare.forEach((id) => {
    const p = data.participants[id];
    const card = el("div", "panel");

    const tone = {
      Dominant: "is-warning", Silent: "is-critical",
      Active: "is-good", Moderate: "is-good",
    }[p.classification] || "";

    const head = el("div", "card-head");
    head.innerHTML =
      `<h3>${swatch(slots.get(id))}${esc(id)}</h3>` +
      `<span class="pill ${tone}">${esc(p.classification)}</span>`;
    card.append(head);

    const rows = [
      ["Speaking time", `${p.speaking_time_sec.toFixed(2)} s`],
      ["Share of session", `${p.participation_pct}%`],
      ["Words", p.word_count],
      ["Pace", `${p.avg_wpm} WPM (${p.pace_label})`],
      ["Filler words", p.filler_count],
      ["Filler density", `${p.filler_density_pct}%`],
      [`Long pauses (≥${cfg.long_pause_sec}s)`, p.long_pauses],
      ["Hesitations", p.hesitation_pauses],
      ["Interruptions made / received", `${p.interruptions_made} / ${p.interruptions_received}`],
      ["Overlapped speech", `${p.overlap_time_sec.toFixed(2)} s`],
      ["Communication score", `${p.communication_score} / 100`],
    ];
    const scroll = el("div", "table-scroll");
    const table = el("table");
    const tbody = el("tbody");
    rows.forEach(([k, v]) => {
      const tr = el("tr");
      tr.append(el("th", null, k), el("td", "v", v));
      tbody.append(tr);
    });
    table.append(tbody);
    scroll.append(table);
    card.append(scroll);

    if (p.keywords && p.keywords.length) {
      const block = el("div");
      block.append(el("p", "eyebrow", "Keywords"));
      const chips = el("div", "chips");
      chips.style.marginTop = "7px";
      p.keywords.forEach((k) => chips.append(el("span", "chip", k)));
      block.append(chips);
      card.append(block);
    }

    [["Strengths", p.strengths, ""], ["Weaknesses", p.weaknesses, " minus"]].forEach(
      ([title, items, mod]) => {
        if (!items || !items.length) return;
        const box = el("div", `verdict${mod}`);
        box.append(el("p", "eyebrow", title));
        const ul = el("ul");
        items.forEach((t) => {
          const li = el("li");
          li.append(el("span", null, t));
          ul.append(li);
        });
        box.append(ul);
        card.append(box);
      });

    cards.append(card);
  });

  section.append(cards);
  return section;
}

/* ---------------- report: dynamics ---------------- */

function dynamicsPanel(data, slots, ids) {
  const panel = el("section", "panel");
  panel.append(panelHead("Phase 3 · conversation dynamics", "Turn taking and interruptions"));

  const seq = data.turn_sequence || [];
  const CAP = 200;
  const shown = seq.slice(0, CAP);

  const strip = el("div", "turns");
  strip.setAttribute("role", "img");
  strip.setAttribute("aria-label", `${seq.length} turns in order of occurrence`);
  shown.forEach((sp, i) => {
    const b = el("i");
    b.style.background = seriesColor(slots.get(sp) ?? 0);
    Tip.bind(b, () => `<div class="t-head">Turn ${i + 1} of ${seq.length}</div>${esc(sp)}`);
    strip.append(b);
  });
  panel.append(strip);

  const legend = el("div", "legend");
  [...slots.keys()].sort().forEach((id) => {
    const s = el("span");
    s.innerHTML = `${swatch(slots.get(id))}${esc(id)}`;
    legend.append(s);
  });
  panel.append(legend);
  if (seq.length > CAP) {
    panel.append(el("p", "note", `Showing the first ${CAP} of ${seq.length} turns.`));
  }

  const matrix = data.interruption_matrix || {};
  const totalInterruptions = Object.values(matrix)
    .reduce((sum, row) => sum + Object.values(row).reduce((a, b) => a + b, 0), 0);
  const edges = data.response_graph_edges || [];

  const facts = el("div", "facts");
  [
    [seq.length, "turns"],
    [totalInterruptions, "interruptions"],
    [(data.overlaps || []).length, "overlapping segments"],
    [edges.length, "distinct transitions"],
  ].forEach(([v, k]) => {
    const f = el("div", "fact");
    f.append(el("span", "v", v), el("span", "k", k));
    facts.append(f);
  });
  panel.append(facts);

  if (totalInterruptions > 0) {
    const scroll = el("div", "table-scroll");
    const table = el("table", "matrix");
    const sorted = [...ids].sort();

    const thead = el("thead");
    const hr = el("tr");
    hr.append(el("th", null, "Interrupter ↓ / interrupted →"));
    sorted.forEach((id) => hr.append(el("th", "v", shortName(id))));
    thead.append(hr);

    const tbody = el("tbody");
    sorted.forEach((from) => {
      const tr = el("tr");
      const th = el("th");
      th.innerHTML = `${swatch(slots.get(from))}${esc(from)}`;
      th.style.display = "flex";
      th.style.alignItems = "center";
      th.style.gap = "6px";
      tr.append(th);
      sorted.forEach((to) => {
        const n = (matrix[from] || {})[to] || 0;
        const td = el("td", n ? "v hot" : "v", from === to ? "—" : n);
        tr.append(td);
      });
      tbody.append(tr);
    });

    table.append(thead, tbody);
    scroll.append(table);
    panel.append(scroll);
  }

  if (data.overlaps && data.overlaps.length) {
    const scroll = el("div", "table-scroll");
    const table = el("table");
    table.innerHTML =
      `<thead><tr><th>Overlap</th><th>Speakers</th><th class="v">Duration</th></tr></thead>`;
    const tbody = el("tbody");
    data.overlaps.slice(0, 40).forEach((o) => {
      const tr = el("tr");
      tr.append(
        el("th", null, `${clock(o.start)}–${clock(o.end)}`),
        el("td", null, `${o.speaker_a} × ${o.speaker_b}`),
        el("td", "v", `${o.duration.toFixed(2)} s`));
      tbody.append(tr);
    });
    table.append(tbody);
    scroll.append(table);
    panel.append(scroll);
    if (data.overlaps.length > 40) {
      panel.append(el("p", "note", `Showing 40 of ${data.overlaps.length} overlaps.`));
    }
  }

  const notes = [];
  if (ids.length === 2) {
    notes.push(
      `With two participants there are only two possible transitions, so the transition-diversity ` +
      `half of the turn-taking score is capped near zero however well the pair actually converse. ` +
      `That metric is built for groups of three or more — read ${data.turn_taking_score} with that in mind.`);
  }
  if (totalInterruptions === 0) {
    notes.push(
      `No interruptions were detected: that requires one speaker to start more than ` +
      `0.5s before another finishes, with at least 0.3s of overlap. Clean hand-offs, or diarization ` +
      `that snapped the boundaries together, both produce zero.`);
  }
  if (notes.length) panel.append(el("p", "note", notes.join(" ")));

  return panel;
}

/* ---------------- report: topics ---------------- */

function topicsPanel(data) {
  const panel = el("section", "panel");
  panel.append(panelHead("Phase 5 · NLP analysis", "Topics over time"));

  const timeline = data.topic_timeline || [];
  const drift = data.topic_drift_events || [];
  const driftKey = (w) => `${w.start_min}-${w.end_min}`;
  const drifted = new Set(drift.map(driftKey));

  if (!timeline.length) {
    panel.append(el("p", "empty", "No topic windows were produced."));
    return panel;
  }

  const scroll = el("div", "table-scroll");
  const table = el("table");
  table.innerHTML = `<thead><tr><th>Window</th><th>Keywords</th><th class="v">Drift</th></tr></thead>`;
  const tbody = el("tbody");

  timeline.forEach((w) => {
    const tr = el("tr");
    tr.append(el("th", "num", `${w.start_min} – ${w.end_min} min`));

    const td = el("td");
    const chips = el("div", "chips");
    (w.topics || []).forEach((t) => chips.append(el("span", "chip", t)));
    if (!w.topics || !w.topics.length) chips.append(el("span", "muted-line", "—"));
    td.append(chips);
    tr.append(td);

    const hit = drift.find((d) => driftKey(d) === driftKey(w));
    tr.append(el("td", "v", hit ? `${hit.similarity_to_global} similarity` : "—"));
    tbody.append(tr);
  });

  table.append(tbody);
  scroll.append(table);
  panel.append(scroll);

  if (drift.length) {
    panel.append(el("p", "note",
      `${drift.length} window${drift.length === 1 ? "" : "s"} flagged as possible topic drift — ` +
      `keyword overlap with the session's overall keyword set fell below the configured threshold.`));
  } else if (timeline.length === 1) {
    panel.append(el("p", "note",
      `The recording is shorter than one 5-minute topic window, so drift detection had a single ` +
      `window and nothing to compare it against.`));
  } else {
    panel.append(el("p", "note", "No window drifted from the session's overall keyword set."));
  }

  return panel;
}

/* ---------------- report: transcript ---------------- */

function transcriptPanel(data, slots, ids) {
  const panel = el("section", "panel");
  panel.append(panelHead("Appendix", "Transcript"));

  const segments = data.transcript || [];
  const filters = el("div", "filters");
  const list = el("div", "tx");

  function draw(who) {
    list.replaceChildren();
    segments.forEach((seg) => {
      if (who !== "all" && seg.speaker !== who) return;
      const row = el("div", "tx-row");
      const meta = el("div", "tx-meta");
      const whoLine = el("span", "tx-who");
      whoLine.innerHTML = `${swatch(slots.get(seg.speaker) ?? 0)}${esc(seg.speaker)}`;
      meta.append(whoLine, el("span", "tx-time", `${clock(seg.start)}–${clock(seg.end)}`));
      row.append(meta, el("p", "tx-text", seg.text));
      list.append(row);
    });
    if (!list.children.length) list.append(el("p", "empty", "Nothing from this speaker."));
  }

  const buttons = [["all", `All ${segments.length}`]].concat(
    [...ids].sort().map((id) => [id, id]));

  buttons.forEach(([key, label], i) => {
    const b = el("button", null);
    b.type = "button";
    b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
    b.innerHTML = key === "all" ? esc(label) : `${swatch(slots.get(key))}${esc(label)}`;
    b.addEventListener("click", () => {
      $$("button", filters).forEach((other) =>
        other.setAttribute("aria-pressed", String(other === b)));
      draw(key);
    });
    filters.append(b);
  });

  panel.append(filters, list);
  draw("all");
  return panel;
}

/* ---------------- report: phase 8 artifacts ---------------- */

function artifactsPanel(data) {
  const available = data.artifacts || [];
  if (!available.length) return null;

  const panel = el("section", "panel prov");
  panel.append(panelHead("Phase 8–9 · generated files", "Charts and report"));

  if (available.includes("summary")) {
    const img = el("img", "summary-img");
    img.src = `/api/sessions/${data.session_id}/chart/summary`;
    img.alt = "Matplotlib summary: participation percentage and communication score per speaker";
    img.loading = "lazy";
    panel.append(img);
  }

  const links = el("div", "report-actions");
  const named = {
    participation_pie: "Participation pie",
    speaking_time_bar: "Speaking time bar",
    timeline: "Plotly timeline",
    network_graph: "Interaction network",
  };
  Object.entries(named).forEach(([key, label]) => {
    if (!available.includes(key)) return;
    const a = el("a", "ghost", label);
    a.href = `/api/sessions/${data.session_id}/chart/${key}`;
    a.target = "_blank";
    a.rel = "noopener";
    links.append(a);
  });
  const pdf = el("a", "ghost", "PDF report");
  pdf.href = `/api/sessions/${data.session_id}/report.pdf`;
  pdf.target = "_blank";
  pdf.rel = "noopener";
  links.append(pdf);
  panel.append(links);

  panel.append(el("p", "note",
    "These are the interactive Plotly files and PDF the pipeline writes to disk for this session. " +
    "They open in a new tab."));
  return panel;
}

/* ---------------- shared ---------------- */

function panelHead(eyebrow, title) {
  const head = el("div", "panel-head");
  head.append(el("p", "eyebrow", eyebrow), el("h2", null, title));
  return head;
}

/* ===================================================================
   ROUTER
   =================================================================== */

function route() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  Tip.hide();

  const hash = location.hash.replace(/^#/, "") || "/";
  const [, head, arg] = hash.split("/");

  $$(".nav a").forEach((a) => a.removeAttribute("aria-current"));

  if (head === "job" && arg) {
    renderJob(arg);
  } else if (head === "session" && arg) {
    renderSession(arg);
  } else if (head === "sessions") {
    $('.nav a[data-nav="sessions"]').setAttribute("aria-current", "page");
    renderSessions();
  } else {
    $('.nav a[data-nav="upload"]').setAttribute("aria-current", "page");
    renderUpload();
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}

window.addEventListener("hashchange", route);
route();
