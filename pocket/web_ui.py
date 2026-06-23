"""Local tracing & lineage UI for Pocket (POCKET-301).

A single, dependency-free HTML page served by the existing Starlette app at
``GET /``. It visualizes *how a query was routed* (which retrieval strategies
the chosen mode activates, whether each is available on the target, and how
many candidates each produced) and *which source files contributed* (the fused
hits, each tagged with the strategies that surfaced it, with per-file chunk
lineage on demand).

The page is plain HTML + vanilla JS with no build step or front-end framework,
so it stays true to Pocket's local-first, no-heavy-deps design. It talks to the
same JSON endpoints the CLI/MCP layers use: ``/trace`` and ``/lineage``.
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pocket — Query Tracing & Lineage</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 0 1rem 3rem; max-width: 980px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin: 1.2rem 0 0.2rem; }
  .sub { opacity: 0.7; margin: 0 0 1.2rem; font-size: 0.9rem; }
  form { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
         margin-bottom: 1rem; }
  input[type=text] { flex: 1 1 320px; padding: 0.5rem 0.7rem; font-size: 1rem;
         border: 1px solid #8884; border-radius: 6px; background: transparent;
         color: inherit; }
  select, input[type=number] { padding: 0.5rem; border: 1px solid #8884;
         border-radius: 6px; background: transparent; color: inherit; }
  input[type=number] { width: 5rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; border: 0; border-radius: 6px;
         background: #3b82f6; color: white; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  .route { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.5rem 0 1.2rem; }
  .chip { padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.8rem;
          border: 1px solid #8886; display: inline-flex; gap: 0.35rem; }
  .chip.on  { background: #22c55e22; border-color: #22c55e; }
  .chip.off { opacity: 0.55; }
  .chip.unavail { text-decoration: line-through; opacity: 0.45; }
  .chip .n { font-variant-numeric: tabular-nums; opacity: 0.8; }
  .hit { border: 1px solid #8883; border-radius: 8px; padding: 0.8rem 1rem;
         margin-bottom: 0.8rem; }
  .hit .top { display: flex; justify-content: space-between; gap: 0.5rem;
         flex-wrap: wrap; align-items: baseline; }
  .file { font-weight: 600; word-break: break-all; }
  .score { font-variant-numeric: tabular-nums; opacity: 0.8; font-size: 0.85rem; }
  .tags { display: flex; gap: 0.35rem; flex-wrap: wrap; margin: 0.4rem 0; }
  .tag { font-size: 0.72rem; padding: 0.1rem 0.45rem; border-radius: 999px;
         background: #3b82f622; border: 1px solid #3b82f6; }
  .tag.vector { background: #8b5cf622; border-color: #8b5cf6; }
  .tag.lexical { background: #f59e0b22; border-color: #f59e0b; }
  .tag.graph { background: #14b8a622; border-color: #14b8a6; }
  .snippet { white-space: pre-wrap; font-size: 0.9rem; opacity: 0.9;
         margin: 0.3rem 0 0; }
  .meta { font-size: 0.78rem; opacity: 0.65; }
  .lineage-btn { background: none; color: #3b82f6; border: 0; padding: 0;
         cursor: pointer; font-size: 0.8rem; }
  .lineage { margin-top: 0.5rem; font-size: 0.82rem; }
  .lineage table { width: 100%; border-collapse: collapse; }
  .lineage td { padding: 0.15rem 0.4rem; border-top: 1px solid #8882;
         vertical-align: top; }
  .lineage td.off { font-variant-numeric: tabular-nums; white-space: nowrap;
         opacity: 0.7; }
  .status { opacity: 0.7; margin: 1rem 0; }
  .error { color: #ef4444; }
  section.review { margin-top: 2.5rem; border-top: 1px solid #8883; padding-top: 1rem; }
  .review-btn { font-size: 0.78rem; padding: 0.2rem 0.7rem; margin-right: 0.4rem;
         background: #22c55e; }
  .review-btn.reject { background: #ef4444; }
  .pending-item .top { align-items: baseline; }
  .conf { font-variant-numeric: tabular-nums; opacity: 0.8; font-size: 0.85rem; }
  .pred { font-style: italic; opacity: 0.85; }
</style>
</head>
<body>
  <h1>Pocket — Query Tracing &amp; Lineage</h1>
  <p class="sub">See how a query is routed across vector, lexical, and graph
     strategies, and which source files answered it.</p>

  <form id="f">
    <input type="text" id="q" placeholder="Ask the knowledge base…" autofocus />
    <select id="mode">
      <option value="hybrid">hybrid</option>
      <option value="auto">auto</option>
      <option value="vector">vector</option>
      <option value="lexical">lexical</option>
      <option value="graph">graph</option>
    </select>
    <input type="number" id="limit" value="5" min="1" max="50" title="result limit" />
    <button type="submit" id="go">Trace</button>
  </form>

  <div id="route" class="route"></div>
  <div id="status" class="status"></div>
  <div id="results"></div>

  <section class="review">
    <h1>Pending review</h1>
    <p class="sub">Low-confidence graph facts the HITL gate staged instead of
       committing. Approve to admit them into retrieval, or reject to discard.</p>
    <form id="reviewForm">
      <button type="submit" id="loadPending">Load pending facts</button>
      <button type="button" id="approveAll" hidden>Approve all</button>
      <button type="button" id="rejectAll" hidden>Reject all</button>
    </form>
    <div id="reviewStatus" class="status"></div>
    <div id="pending"></div>
  </section>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function renderRoute(strategies) {
  $("route").innerHTML = strategies.map((s) => {
    let cls = "chip " + (s.active ? "on" : "off");
    if (!s.available) cls += " unavail";
    const note = !s.available ? " (unavailable)" : (s.active ? "" : " (off)");
    return `<span class="${cls}" title="${s.name}${note}">` +
           `${esc(s.name)}<span class="n">${s.candidates}</span></span>`;
  }).join("");
}

async function loadLineage(filePath, container) {
  container.textContent = "Loading lineage…";
  try {
    const r = await fetch("/lineage?file_path=" + encodeURIComponent(filePath));
    const d = await r.json();
    if (!r.ok) { container.innerHTML =
      `<span class="error">${esc(d.error || "error")}</span>`; return; }
    const rows = (d.chunks || []).map((c) =>
      `<tr><td class="off">${c.start_offset}\u2013${c.end_offset}</td>` +
      `<td>${esc(c.snippet)}</td></tr>`).join("");
    container.innerHTML = `<div>${d.chunk_count} chunk(s) in this file:</div>` +
      `<table>${rows}</table>`;
  } catch (e) {
    container.innerHTML = `<span class="error">${esc(e.message)}</span>`;
  }
}

function renderResults(data) {
  if (!data.results.length) {
    $("results").innerHTML = "<p class='status'>No results.</p>";
    return;
  }
  $("results").innerHTML = data.results.map((h, i) => {
    const tags = (h.contributors || []).map((c) =>
      `<span class="tag ${c}">${esc(c)}</span>`).join("");
    return `<div class="hit">
      <div class="top">
        <span class="file">[${i + 1}] ${esc(h.file_path)}</span>
        <span class="score">score ${h.score.toFixed(4)}</span>
      </div>
      <div class="tags">${tags || "<span class=meta>no strategy tags</span>"}</div>
      <div class="meta">chars ${h.start_offset}\u2013${h.end_offset}
        &middot; <button class="lineage-btn" data-file="${esc(h.file_path)}">
        show file lineage</button></div>
      <div class="lineage" hidden></div>
      <p class="snippet">${esc((h.text || "").trim().slice(0, 400))}</p>
    </div>`;
  }).join("");

  document.querySelectorAll(".lineage-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const box = btn.closest(".hit").querySelector(".lineage");
      if (box.hidden) {
        box.hidden = false;
        if (!box.dataset.loaded) {
          box.dataset.loaded = "1";
          loadLineage(btn.dataset.file, box);
        }
        btn.textContent = "hide file lineage";
      } else {
        box.hidden = true;
        btn.textContent = "show file lineage";
      }
    });
  });
}

$("f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  $("go").disabled = true;
  $("status").textContent = "Tracing…";
  $("status").className = "status";
  $("results").innerHTML = "";
  $("route").innerHTML = "";
  try {
    const params = new URLSearchParams({
      q, mode: $("mode").value, limit: $("limit").value });
    const r = await fetch("/trace?" + params.toString());
    const data = await r.json();
    if (!r.ok) {
      $("status").textContent = data.error || "error";
      $("status").className = "status error";
      return;
    }
    renderRoute(data.strategies);
    $("status").textContent =
      `mode "${data.mode}" \u2192 ${data.results.length} result(s)`;
    renderResults(data);
  } catch (e) {
    $("status").textContent = e.message;
    $("status").className = "status error";
  } finally {
    $("go").disabled = false;
  }
});

// ---- Pending review (POCKET-505) --------------------------------------------
function setReviewStatus(text, isError) {
  $("reviewStatus").textContent = text;
  $("reviewStatus").className = isError ? "status error" : "status";
}

function pendingItem(kind, id, label, conf, src) {
  return `<div class="hit pending-item" data-id="${esc(id)}">
    <div class="top">
      <span class="file">${label}</span>
      <span class="conf">conf ${Number(conf).toFixed(3)}</span>
    </div>
    <div class="meta">${esc(kind)}${src ? " &middot; " + esc(src) : ""}</div>
    <div class="tags">
      <button class="review-btn" data-act="approve" data-id="${esc(id)}">approve</button>
      <button class="review-btn reject" data-act="reject" data-id="${esc(id)}">reject</button>
    </div>
  </div>`;
}

function renderPending(d) {
  const ents = d.entities || [];
  const rels = d.relations || [];
  const total = ents.length + rels.length;
  $("approveAll").hidden = total === 0;
  $("rejectAll").hidden = total === 0;
  setReviewStatus(total
    ? `${total} fact(s) awaiting review` : "No facts pending review.", false);
  const items = ents.map((e) => pendingItem("entity", e.id,
      `${esc(e.name)} <span class="meta">(${esc(e.type)})</span>`,
      e.confidence, e.source_file))
    .concat(rels.map((r) => pendingItem("relation", r.id,
      `${esc(r.subject)} <span class="pred">${esc(r.predicate)}</span> ` +
      `${esc(r.object)}`, r.confidence, r.source_file)));
  $("pending").innerHTML = items.join("");
  document.querySelectorAll(".review-btn").forEach((b) => {
    b.addEventListener("click", () => review(b.dataset.act, [b.dataset.id]));
  });
}

async function loadPending() {
  setReviewStatus("Loading…", false);
  try {
    const r = await fetch("/pending");
    const d = await r.json();
    if (!r.ok) { setReviewStatus(d.error || "error", true); return; }
    renderPending(d);
  } catch (e) {
    setReviewStatus(e.message, true);
  }
}

async function review(action, ids) {
  setReviewStatus("Submitting…", false);
  try {
    const r = await fetch("/pending/" + action, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    const d = await r.json();
    if (!r.ok) { setReviewStatus(d.error || "error", true); return; }
    loadPending();
  } catch (e) {
    setReviewStatus(e.message, true);
  }
}

$("reviewForm").addEventListener("submit", (ev) => { ev.preventDefault(); loadPending(); });
$("approveAll").addEventListener("click", () => review("approve", null));
$("rejectAll").addEventListener("click", () => review("reject", null));
</script>
</body>
</html>
"""
