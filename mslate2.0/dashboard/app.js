/* TRAP-X dashboard client.
   Streams per-bar records from /api/stream and renders everything off a single
   "cursor" bar: hover > pinned > latest. */
(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const SESSION_MIN = 375;                       // 09:15 → 15:30
  const CHECKS = [
    ["Futures Displacement", 30], ["Option Ladder", 25], ["Volume Expansion", 20],
    ["Delta Acceleration", 10], ["Trap Structure", 10], ["Regime Alignment", 5],
  ];
  const STRATS = [
    { id: "S1", name: "Bounce PDL break", side: "call", rule: "Fake PDL break, up LIS > 2× EM, next bar holds" },
    { id: "S2", name: "LIS high retracement", side: "call", rule: "In a call, low retests a prior post-entry LIS high" },
    { id: "S3", name: "Top trade, less SL", side: "put", rule: "After 12:30 · 2 wick rejections + ITM PE selling" },
    { id: "S4", name: "Fibo retracement bottom", side: "call", rule: "After 12:30 · low tags Fib 50% with fresh up LIS" },
    { id: "S5", name: "Fake PDH break", side: "put", rule: "Fake PDH break, down LIS > 2× EM, next bar holds" },
  ];
  const ACTION_LABEL = {
    S1_INITIAL_CALL: "S1 · Initial call", S2_PYRAMID_CALL: "S2 · Pyramid call",
    S3_INITIAL_PUT: "S3 · Initial put", S4_INITIAL_CALL: "S4 · Initial call", S5_INITIAL_PUT: "S5 · Initial put",
    EXIT_STOP: "Exit · stop", EXIT_TRAIL: "Exit · trailing stop", EXIT_TARGET: "Exit · target",
    EXIT_EOD: "Exit · 15:15 square-off", EXIT_REVERSAL: "Exit · reversal", EXIT_KILL: "Exit · kill switch",
    KILL_SWITCH: "Risk · kill switch",
    PDL_BREAK: "PDL broken", PDL_FAKE: "PDL · fake breakout", PDL_REAL: "PDL · real breakout",
    PDH_BREAK: "PDH broken", PDH_FAKE: "PDH · fake breakout", PDH_REAL: "PDH · real breakout",
    ENTRY: "Trap engine · entry", TIME_EXIT: "Trap engine · time exit",
    STRUCTURAL_EXIT: "Trap engine · stop exit", EOD_EXIT: "Trap engine · EOD exit",
  };
  const labelOf = (e) => e.type === "SKIP"
    ? `${e.strategy} · ${e.action.endsWith("RISK_SKIP") ? "skipped (risk)" : "gated"}`
    : ACTION_LABEL[e.action] || e.action;

  const S = {
    bars: [], slots: [], events: [], log: [], ai: [],
    pdc: null, em: null, cfg: { capital: 1e6, lot_size: 65, max_daily_loss_pct: 2.5 }, aiStatus: null,
    summary: null, hover: null, pin: null,
    status: "idle", speed: "4x", es: null, ledgerFilter: "trades",
    flowMax: 10, pnlMax: 2000,
  };

  // ── formatting ───────────────────────────────────────────────────────────
  const nf2 = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const nf0 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
  const f2 = (v) => (v == null || isNaN(v) ? "—" : nf2.format(v));
  const inr = (v) => (v == null || isNaN(v) ? "—" : (v >= 0 ? "+" : "−") + "₹" + nf0.format(Math.abs(v)));
  const sgn = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const slotOf = (t) => { const [h, m] = t.split(":").map(Number); return h * 60 + m - 555; };
  const timeOf = (slot) => { const m = slot + 555; return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`; };
  const sideOf = (e) => (["PUT", "DOWN"].includes(e.direction) ? "put" : ["CALL", "UP"].includes(e.direction) ? "call" : "");
  const isTrade = (e) => ["ENTRY", "EXIT", "RISK", "TRAP"].includes(e.type);

  const cursorIdx = () => (S.bars.length ? S.hover ?? S.pin ?? S.bars.length - 1 : null);

  // ── data ingest ──────────────────────────────────────────────────────────
  function reset() {
    Object.assign(S, { bars: [], slots: [], events: [], log: [], ai: [], pdc: null, em: null, aiStatus: null,
      summary: null, hover: null, pin: null, flowMax: 10, pnlMax: 2000 });
    logDirty = true;
  }

  function addBar(b) {
    if (b.pdc) { S.pdc = b.pdc; S.em = b.em; }
    if (b.cfg) S.cfg = b.cfg;
    if (b.ai_status) S.aiStatus = b.ai_status;
    S.bars.push(b);
    S.slots.push(slotOf(b.t));
    S.flowMax = Math.max(S.flowMax, Math.abs(b.signal));
    S.pnlMax = Math.max(S.pnlMax, Math.abs(b.equity - S.cfg.capital) * 1.15);
    for (const e of b.exec_events || []) S.events.push({ ...e, i: b.i, t: b.t });
    for (const e of b.trade_events || []) S.events.push({ ...e, type: "TRAP", i: b.i, t: b.t });
    for (const o of b.ai || []) S.ai.push(o);
  }

  // ── chart ────────────────────────────────────────────────────────────────
  const canvas = $("#chart"), ctx = canvas.getContext("2d");
  let C = {}, geo = null, raf = 0;

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["paper", "paper-2", "paper-3", "ink", "ink-2", "ink-3", "ink-4", "rule", "rule-2",
      "up", "up-soft", "down", "down-soft", "accent", "signal", "signal-soft"]) C[k] = cs.getPropertyValue("--" + k).trim();
    C.mono = cs.getPropertyValue("--font-mono").trim();
  }

  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); renderPanels(); }); };

  function layout(W, H) {
    const padR = 68, axis = 22, gapT = 14, title = 16, small = W < 600;
    const lanes = [
      { key: "pos", title: "Workflow position", h: 12 },
      { key: "score", title: "Conviction · bar ▮  setup ○", h: small ? 40 : 52 },
      { key: "flow", title: "Options flow · final signal", h: small ? 36 : 44 },
      { key: "pnl", title: "P&L · ₹ marked to market", h: small ? 40 : 56 },
    ];
    let y = H - axis;
    for (let k = lanes.length - 1; k >= 0; k--) {
      const L = lanes[k];
      L.y1 = y; L.y0 = y - L.h; L.ty = L.y0 - 6; L.top = L.y0 - title - gapT;
      y = L.top;
    }
    const plotW = W - padR;
    return { W, H, padR, plotW, slot: plotW / SESSION_MIN, price: { y0: 6, y1: y - 8 }, lanes, axisY: H - axis };
  }

  function niceTicks(lo, hi, n) {
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
    return out;
  }

  function draw() {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width) return;
    if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
      canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    const G = (geo = layout(r.width, r.height));
    const B = S.bars, k = cursorIdx();
    const xc = (slot) => (slot + 0.5) * G.slot;
    ctx.font = `400 10px ${C.mono}`;
    ctx.textBaseline = "middle";

    // price domain — candles, reference levels, and any live stop/target
    let lo = Infinity, hi = -Infinity;
    for (const b of B) {
      if (b.l < lo) lo = b.l; if (b.h > hi) hi = b.h;
      if (b.pos) { lo = Math.min(lo, b.pos.stop); hi = Math.max(hi, b.pos.stop); }
    }
    if (S.pdc) { lo = Math.min(lo, S.pdc.prev_day_low, S.pdc.prev_day_close); hi = Math.max(hi, S.pdc.prev_day_close); }
    if (!isFinite(lo)) { lo = 25300; hi = 25700; }
    const pad = (hi - lo) * 0.08 || 10; lo -= pad; hi += pad;
    const P = G.price, py = (v) => P.y1 - ((v - lo) / (hi - lo)) * (P.y1 - P.y0);
    const inP = (v) => v >= lo && v <= hi;

    // time grid (hourly) + axis labels
    ctx.textAlign = "center";
    for (let s = 45; s < SESSION_MIN; s += 60) {
      const x = Math.round(xc(s)) + 0.5;
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, P.y0); ctx.lineTo(x, G.axisY); ctx.stroke();
    }
    ctx.fillStyle = C["ink-3"];
    const every = G.W < 600 ? 90 : 30;
    for (let s = 0; s < SESSION_MIN; s += every) {
      if (s === 0) { ctx.textAlign = "left"; ctx.fillText(timeOf(s), 0, G.axisY + 12); ctx.textAlign = "center"; continue; }
      ctx.fillText(timeOf(s), xc(s), G.axisY + 12);
    }

    // price grid
    ctx.textAlign = "left";
    for (const v of niceTicks(lo, hi, 5)) {
      const y = Math.round(py(v)) + 0.5;
      ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(G.plotW, y); ctx.stroke();
      ctx.fillStyle = C["ink-3"]; ctx.fillText(f2(v).replace(/\.00$/, ""), G.plotW + 10, y);
    }

    // reference levels
    if (S.pdc) {
      const fib = k != null ? S.bars[k].fib_50 : 0;
      const lv = [["PDL", S.pdc.prev_day_low, C["ink-4"]], ["PDC", S.pdc.prev_day_close, C["ink-4"]],
                  ["PDH", S.pdc.prev_day_high, C["ink-4"]], ["FIB 50%", fib, C.signal]];
      ctx.textAlign = "right";
      for (const [name, v, col] of lv) {
        if (!v) continue;
        if (!inP(v)) {
          if (name === "PDH" && v > hi) { ctx.fillStyle = C["ink-3"]; ctx.fillText(`PDH ${f2(v)} ↑`, G.plotW - 4, P.y0 + 8); }
          continue;
        }
        const y = Math.round(py(v)) + 0.5, label = `${name} ${f2(v)}`, tw = ctx.measureText(label).width + 8;
        ctx.setLineDash([3, 4]); ctx.strokeStyle = col;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(G.plotW, y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.paper; ctx.fillRect(G.plotW - tw - 2, y - 7, tw, 14);
        ctx.fillStyle = name.startsWith("FIB") ? C.signal : C["ink-3"]; ctx.fillText(label, G.plotW - 6, y);
      }
      ctx.textAlign = "left";
    }

    // open-position stop (steps up as it trails) and target
    for (const key of ["stop", "target"]) {
      ctx.lineWidth = key === "stop" ? 1.25 : 1;
      ctx.setLineDash(key === "stop" ? [] : [2, 3]);
      let open = false;
      for (let i = 0; i < B.length; i++) {
        const p = B[i].pos;
        if (!p || !inP(p[key])) { if (open) ctx.stroke(); open = false; continue; }
        const x0 = xc(S.slots[i]) - G.slot / 2, x1 = xc(S.slots[i]) + G.slot / 2, y = py(p[key]);
        if (!open) {
          ctx.strokeStyle = key === "target" ? C["ink-3"] : p.direction === "PUT" ? C.down : C.up;
          ctx.globalAlpha = key === "stop" ? 0.85 : 0.6;
          ctx.beginPath(); ctx.moveTo(x0, y); open = true;
        }
        ctx.lineTo(x0, y); ctx.lineTo(x1, y);
      }
      if (open) ctx.stroke();
      ctx.globalAlpha = 1; ctx.setLineDash([]); ctx.lineWidth = 1;
    }

    // candles
    const cw = Math.max(1, Math.min(7, G.slot * 0.62));
    for (let i = 0; i < B.length; i++) {
      const b = B[i], x = xc(S.slots[i]), up = b.c >= b.o;
      const dim = k != null && S.hover == null && S.pin != null && i > k;
      ctx.globalAlpha = dim ? 0.3 : 1;
      ctx.strokeStyle = ctx.fillStyle = up ? C.up : C.down;
      ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, py(b.h)); ctx.lineTo(Math.round(x) + 0.5, py(b.l)); ctx.stroke();
      const y0 = py(Math.max(b.o, b.c)), y1 = py(Math.min(b.o, b.c));
      ctx.fillRect(Math.round(x - cw / 2), y0, Math.round(cw), Math.max(1, y1 - y0));
    }
    ctx.globalAlpha = 1;

    // TWAP
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1.25; ctx.beginPath();
    let started = false;
    for (let i = 0; i < B.length; i++) {
      if (!B[i].twap) continue;
      const x = xc(S.slots[i]), y = py(B[i].twap);
      started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), (started = true));
    }
    ctx.stroke(); ctx.lineWidth = 1;

    // events: breakout verdicts, gated setups, entries, exits
    ctx.textAlign = "center";
    for (const e of S.events) {
      const b = B[e.i]; if (!b) continue;
      const x = xc(slotOf(e.t)), side = sideOf(e), col = side === "put" ? C.down : C.up;
      if (e.type === "BREAKOUT" && !e.action.endsWith("BREAK")) {
        const lvl = e.level === "PDH" ? S.pdc?.prev_day_high : S.pdc?.prev_day_low;
        if (lvl == null || !inP(lvl)) continue;
        // stack repeat verdicts on the same level so labels never collide
        const nth = S.events.filter((o) => o.type === "BREAKOUT" && o.level === e.level && !o.action.endsWith("BREAK") && o.i < e.i).length;
        const fake = e.action.endsWith("FAKE"), y = py(lvl) + (e.level === "PDH" ? -1 : 1) * (10 + nth * 11);
        ctx.font = `500 9px ${C.mono}`; ctx.fillStyle = fake ? C.signal : C["ink-3"];
        ctx.textAlign = "left";
        ctx.fillText(`${e.level} ${fake ? "FAKE" : "REAL"} ${e.t}`, Math.max(2, x - 4), y);
        ctx.textAlign = "center"; ctx.font = `400 10px ${C.mono}`;
      } else if (e.type === "SKIP") {
        ctx.strokeStyle = C["ink-3"]; ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.arc(x, py(e.entry), 4, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      } else if (e.type === "ENTRY" || (e.type === "TRAP" && e.action === "ENTRY")) {
        const label = e.strategy || "T";
        ctx.fillStyle = col; ctx.beginPath();
        if (side === "put") { const y = py(b.h) - 8; ctx.moveTo(x - 5, y - 6); ctx.lineTo(x + 5, y - 6); ctx.lineTo(x, y); ctx.fill(); ctx.fillText(label, x, y - 16); }
        else { const y = py(b.l) + 8; ctx.moveTo(x - 5, y + 6); ctx.lineTo(x + 5, y + 6); ctx.lineTo(x, y); ctx.fill(); ctx.fillText(label, x, y + 16); }
      } else if (e.type === "EXIT") {
        const y = py(e.price);
        ctx.fillStyle = C.paper; ctx.strokeStyle = C.ink;
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = e.pnl_inr >= 0 ? C.up : C.down;
        ctx.fillText(`${sgn(e.r_multiple, 2)}R`, x, y + (e.direction === "PUT" ? 14 : -14));
      }
    }

    // last / cursor price tag
    if (k != null) {
      const b = B[k], y = Math.round(py(b.c)) + 0.5, live = S.hover == null && S.pin == null;
      ctx.strokeStyle = C["ink-4"]; ctx.setLineDash(live ? [1, 3] : []);
      ctx.beginPath(); ctx.moveTo(live ? xc(S.slots[k]) : 0, y); ctx.lineTo(G.plotW, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = live ? (b.c >= (S.pdc?.prev_day_close ?? b.o) ? C.up : C.down) : C.ink;
      ctx.fillRect(G.plotW + 4, y - 9, G.padR - 4, 18);
      ctx.fillStyle = C.paper; ctx.textAlign = "left"; ctx.fillText(f2(b.c), G.plotW + 10, y);
    }

    // lanes
    ctx.textAlign = "left";
    for (const L of G.lanes) {
      ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(0, L.top + 0.5); ctx.lineTo(G.W, L.top + 0.5); ctx.stroke();
      ctx.fillStyle = C["ink-3"]; ctx.font = `400 10px ${C.mono}`;
      ctx.fillText(L.title.toUpperCase(), 0, L.ty);
      ({ pos: drawPos, score: drawScore, flow: drawFlow, pnl: drawPnl })[L.key](L, xc);
    }

    // crosshair
    if (k != null && (S.hover != null || S.pin != null)) {
      const x = Math.round(xc(S.slots[k])) + 0.5;
      ctx.strokeStyle = C["ink-3"]; ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(x, P.y0); ctx.lineTo(x, G.axisY); ctx.stroke(); ctx.globalAlpha = 1;
      const t = B[k].t, tw = 40;
      ctx.fillStyle = C.ink; ctx.fillRect(Math.min(Math.max(0, x - tw / 2), G.plotW - tw), G.axisY + 3, tw, 17);
      ctx.fillStyle = C.paper; ctx.textAlign = "center"; ctx.fillText(t, Math.min(Math.max(tw / 2, x), G.plotW - tw / 2), G.axisY + 12);
    }
  }

  const baseline = (L) => { ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(0, L.y1 + 0.5); ctx.lineTo(geo.plotW, L.y1 + 0.5); ctx.stroke(); };

  function drawPos(L, xc) {
    const B = S.bars; let i = 0;
    while (i < B.length) {
      const p = B[i].wf_position; let j = i;
      while (j + 1 < B.length && B[j + 1].wf_position === p) j++;
      if (p !== "IDLE") {
        const x0 = xc(S.slots[i]) - geo.slot / 2, x1 = xc(S.slots[j]) + geo.slot / 2;
        ctx.fillStyle = p === "CALL" ? C["up-soft"] : C["down-soft"];
        ctx.fillRect(x0, L.y0, x1 - x0, L.h);
        ctx.fillStyle = p === "CALL" ? C.up : C.down;
        ctx.fillRect(x0, L.y0, 2, L.h);
        if (x1 - x0 > 40) { ctx.font = `500 9px ${C.mono}`; ctx.fillText(p, x0 + 6, L.y0 + L.h / 2 + 0.5); ctx.font = `400 10px ${C.mono}`; }
      }
      i = j + 1;
    }
    baseline(L);
  }

  function drawScore(L, xc) {
    const w = Math.max(1, geo.slot * 0.62);
    for (let i = 0; i < S.bars.length; i++) {
      const s = S.bars[i].score; if (!s) continue;
      const h = (s / 100) * L.h;
      ctx.fillStyle = C["ink-4"];
      ctx.fillRect(xc(S.slots[i]) - w / 2, L.y1 - h, w, h);
    }
    // proposed setups, scored by the gate
    for (let i = 0; i < S.bars.length; i++) {
      for (const g of S.bars[i].gate || []) {
        const y = L.y1 - (g.score / 100) * L.h, x = xc(S.slots[i]);
        ctx.strokeStyle = g.tier ? (g.direction === "PUT" ? C.down : C.up) : C["ink-2"];
        ctx.fillStyle = g.tier ? ctx.strokeStyle : C.paper;
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }
    const y = Math.round(L.y1 - 0.7 * L.h) + 0.5;
    ctx.strokeStyle = C.signal; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(geo.plotW, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.signal; ctx.fillText("70 · act", geo.plotW + 10, y);
    baseline(L);
  }

  function drawFlow(L, xc) {
    const w = Math.max(1, geo.slot * 0.62), mid = Math.round((L.y0 + L.y1) / 2) + 0.5, half = L.h / 2;
    for (let i = 0; i < S.bars.length; i++) {
      const v = S.bars[i].signal; if (!v) continue;
      const h = (Math.abs(v) / S.flowMax) * half;
      ctx.fillStyle = v > 0 ? C.up : C.down; ctx.globalAlpha = 0.55;
      ctx.fillRect(xc(S.slots[i]) - w / 2, v > 0 ? mid - h : mid, w, h);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C["rule-2"]; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(geo.plotW, mid); ctx.stroke();
    ctx.fillStyle = C["ink-3"];
    ctx.fillText(`+${S.flowMax.toFixed(0)}`, geo.plotW + 10, L.y0 + 4);
    ctx.fillText(`−${S.flowMax.toFixed(0)}`, geo.plotW + 10, L.y1 - 4);
  }

  function drawPnl(L, xc) {
    const m = S.pnlMax, mid = (L.y0 + L.y1) / 2, yv = (v) => mid - (v / m) * (L.h / 2);
    ctx.strokeStyle = C["rule-2"]; ctx.beginPath(); ctx.moveTo(0, Math.round(mid) + 0.5); ctx.lineTo(geo.plotW, Math.round(mid) + 0.5); ctx.stroke();
    if (!S.bars.length) return;
    ctx.beginPath(); ctx.moveTo(xc(S.slots[0]), mid);
    for (let i = 0; i < S.bars.length; i++) ctx.lineTo(xc(S.slots[i]), yv(S.bars[i].equity - S.cfg.capital));
    const last = S.bars[S.bars.length - 1], v = last.equity - S.cfg.capital;
    ctx.lineTo(xc(S.slots[S.bars.length - 1]), mid); ctx.closePath();
    ctx.fillStyle = v >= 0 ? C["up-soft"] : C["down-soft"]; ctx.fill();
    ctx.beginPath();
    for (let i = 0; i < S.bars.length; i++) {
      const x = xc(S.slots[i]), y = yv(S.bars[i].equity - S.cfg.capital);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = v >= 0 ? C.up : C.down; ctx.lineWidth = 1.25; ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = C["ink-3"];
    ctx.fillText(inr(m), geo.plotW + 10, L.y0 + 4);
    ctx.fillText(inr(-m), geo.plotW + 10, L.y1 - 4);
  }

  // ── hit-testing ──────────────────────────────────────────────────────────
  function idxAtX(x) {
    if (!geo || !S.bars.length) return null;
    const slot = Math.floor(x / geo.slot);
    let lo = 0, hi = S.slots.length - 1;
    if (slot >= S.slots[hi]) return hi;
    while (lo < hi) { const m = (lo + hi) >> 1; S.slots[m] < slot ? (lo = m + 1) : (hi = m); }
    return lo > 0 && Math.abs(S.slots[lo - 1] - slot) < Math.abs(S.slots[lo] - slot) ? lo - 1 : lo;
  }

  const tip = document.createElement("div"); tip.className = "tip"; document.body.appendChild(tip);
  canvas.addEventListener("pointermove", (ev) => {
    const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left;
    S.hover = x <= (geo?.plotW ?? 0) ? idxAtX(x) : null;
    const evs = S.hover != null
      ? S.events.filter((e) => e.i === S.hover && !(e.type === "BREAKOUT" && e.action.endsWith("BREAK")))
      : [];
    if (evs.length) {
      tip.innerHTML = evs.map((e) => `<b>${esc(labelOf(e))}</b>${e.price != null ? ` @ ${f2(e.price)}` : ""}${e.score != null ? ` · ${e.score}/100` : ""}<br>${esc(noteFor(e).replace(/<[^>]+>/g, ""))}`).join("<br>");
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - 290) + "px";
      tip.style.top = ev.clientY + 14 + "px";
      tip.classList.add("show");
    } else tip.classList.remove("show");
    schedule();
  });
  canvas.addEventListener("pointerleave", () => { S.hover = null; tip.classList.remove("show"); schedule(); });
  canvas.addEventListener("click", () => { if (S.hover != null) { S.pin = S.pin === S.hover ? null : S.hover; schedule(); } });

  // ── panels ───────────────────────────────────────────────────────────────
  let lastLedgerKey = "";

  function renderPanels() {
    const k = cursorIdx(), b = k != null ? S.bars[k] : null;
    $("#empty").classList.toggle("hide", S.bars.length > 0);

    if (b) {
      $("#price").textContent = f2(b.c);
      const pdc = S.pdc?.prev_day_close;
      if (pdc) {
        const d = b.c - pdc;
        $("#delta").className = "delta " + (d >= 0 ? "up" : "down");
        $("#delta").textContent = `${sgn(d)}  ${sgn((d / pdc) * 100)}%  vs PDC`;
      }
      const live = S.hover == null && S.pin == null;
      const cl = $("#cursor-label");
      cl.className = "cursor-label" + (live && S.status === "running" ? " live" : "");
      cl.textContent = live ? (S.status === "running" ? `Live · bar ${b.t}` : `Close · bar ${b.t}`) : `${S.pin != null && S.hover == null ? "Pinned" : "Inspecting"} · bar ${b.t}`;
      $("#lede").innerHTML = lede(k);
    }
    const P = S.pdc;
    $("#f-pdh").textContent = P ? f2(P.prev_day_high) : "—";
    $("#f-pdl").textContent = P ? f2(P.prev_day_low) : "—";
    $("#f-pdc").textContent = P ? f2(P.prev_day_close) : "—";
    $("#f-vix").textContent = P ? P.vix_close.toFixed(2) : "—";
    $("#f-em").textContent = S.em ? `${S.em.toFixed(2)} pts` : "—";
    $("#f-regime").innerHTML = b ? `<span class="tag ${b.regime.toLowerCase()}">${b.regime}</span>` : "—";
    $("#f-score").textContent = b ? `${b.score} / 100` : "—";
    $("#f-pos").innerHTML = b ? `<span class="tag ${b.wf_position.toLowerCase()}">${b.wf_position}</span>` : "—";
    $("#f-pnl").textContent = b ? inr(b.realized) : "—";
    $("#f-dd").textContent = b ? `${b.dd.toFixed(2)}%` : "—";
    $("#f-bars").textContent = b ? `${k + 1} / ${S.status === "running" ? "…" : S.bars.length}` : "—";
    const lat = S.bars.length ? S.bars.reduce((a, x) => a + x.latency_ms, 0) / S.bars.length : null;
    $("#f-lat").textContent = lat != null ? `${lat.toFixed(1)} ms` : "—";
    $("#pipe-meta").textContent = b ? `${b.latency_ms.toFixed(1)} ms @ ${b.t}` : "13 nodes";

    $("#readout").innerHTML = b
      ? `<b>${b.t}</b><span class="sep">/</span>O <b>${f2(b.o)}</b>  H <b>${f2(b.h)}</b>  L <b>${f2(b.l)}</b>  C <b>${f2(b.c)}</b><span class="sep">/</span>TWAP <b>${f2(b.twap)}</b>  ATR <b>${b.atr.toFixed(2)}</b><span class="sep">/</span>Fut <b>${f2(b.fut_c)}</b><span class="sep">/</span>Flow <b>${sgn(b.signal)}</b>${b.lis ? `<span class="sep">/</span><b>LIS</b>` : ""}`
      : "&nbsp;";

    renderStrategies(k);
    renderChecklist(b);
    renderPosition(b);
    renderAI(b);
    renderLedger();
  }

  function lastEntry(k) {
    let e = null;
    for (const x of S.events) if (x.i <= k && x.type === "ENTRY") e = x;
    return e;
  }

  function lede(k) {
    const b = S.bars[k], here = S.events.filter((e) => e.i === k), parts = [];
    const bo = here.find((e) => e.type === "BREAKOUT" && !e.action.endsWith("BREAK"));
    const brk = here.find((e) => e.type === "BREAKOUT" && e.action.endsWith("BREAK"));
    const ent = here.find((e) => e.type === "ENTRY"), ex = here.find((e) => e.type === "EXIT");
    const gated = (b.gate || []).find((g) => !g.tier);
    if (bo) parts.push(`<b>${bo.level} break is ${bo.action.endsWith("FAKE") ? "a trap" : "real"}.</b>`);
    else if (brk) parts.push(`<b>${brk.level} just broke — validating, don’t chase.</b>`);
    if (ex) parts.push(`<b>${labelOf(ex).replace("Exit · ", "Closed at ")}: ${inr(ex.pnl_inr)} (${sgn(ex.r_multiple)}R).</b>`);
    if (ent) parts.push(`<b>${ent.strategy} enters ${ent.direction.toLowerCase()} at ${ent.score}/100.</b>`);
    else if (gated) parts.push(`${gated.strategy} proposed a ${gated.direction.toLowerCase()} but scored ${gated.score} — gated.`);
    const e = lastEntry(k);
    if (b.pos && e) parts.push(`Holding a <b>${b.pos.direction.toLowerCase()}</b> from ${b.pos.strategy}, stop ${f2(b.pos.stop)}${b.pos.trailing ? " (trailing)" : ""}.`);
    else if (!ex) parts.push(b.kill ? "Kill switch is on — flat for the day." : "Flat.");
    parts.push(b.regime === "UNDEFINED" ? "Regime still forming." : `Regime reads <b>${b.regime.toLowerCase()}</b>.`);
    return parts.join(" ");
  }

  function renderStrategies(k) {
    const b = k != null ? S.bars[k] : null;
    const after1230 = b && slotOf(b.t) >= slotOf("12:30");
    $("#strategies").innerHTML = STRATS.map((s) => {
      const entered = S.events.filter((e) => e.type === "ENTRY" && e.strategy === s.id && (k == null || e.i <= k)).pop();
      let lastGate = null;
      if (b) for (let i = k; i >= 0 && !lastGate; i--) lastGate = (S.bars[i].gate || []).find((g) => g.strategy === s.id) || null;
      let chip = `<span class="chip">Waiting</span>`;
      if (entered) chip = `<span class="chip fired ${s.side}">Entered ${entered.t}</span>`;
      else if (lastGate) chip = `<span class="chip armed">Gated ${lastGate.score} @ ${lastGate.bar_time}</span>`;
      else if (b) {
        const bo = b.bo || {};
        if (s.id === "S1") chip = bo.PDL ? `<span class="chip${bo.PDL === "FAKE" ? " armed" : ""}">PDL ${bo.PDL.toLowerCase()}</span>` : `<span class="chip">Watching PDL</span>`;
        if (s.id === "S5") chip = bo.PDH ? `<span class="chip${bo.PDH === "FAKE" ? " armed" : ""}">PDH ${bo.PDH.toLowerCase()}</span>` : `<span class="chip">Watching PDH</span>`;
        if (s.id === "S2") chip = b.wf_position === "CALL" ? `<span class="chip armed">Armed</span>` : `<span class="chip">Needs a call</span>`;
        if (s.id === "S3") chip = after1230 ? `<span class="chip${b.wick_rejections ? " armed" : ""}">Wicks ${b.wick_rejections}/2</span>` : `<span class="chip">Opens 12:30</span>`;
        if (s.id === "S4") chip = !after1230 ? `<span class="chip">Opens 12:30</span>` : b.fib_50 ? `<span class="chip armed">Fib ${f2(b.fib_50)}</span>` : `<span class="chip">No fib yet</span>`;
      }
      return `<li><span class="id">${s.id}</span><span class="name">${s.name}</span>${chip}<span class="rule">${s.rule}</span></li>`;
    }).join("");
  }

  function renderChecklist(b) {
    const g = b && (b.gate || [])[0];
    const score = g ? g.score : b ? b.score : 0, list = g ? g.checklist : b?.checklist;
    const sb = $("#scorebar");
    sb.querySelector(".fill").style.width = score + "%";
    sb.classList.toggle("pass", score >= 70);
    $("#conv-meta").textContent = b ? `${score}/100 · ${score >= 80 ? "FULL" : score >= 70 ? "HALF" : g ? "GATED" : "AVOID"}` : "—";
    $("#conv-src").innerHTML = g
      ? `Setup <b>${g.strategy} ${g.direction.toLowerCase()}</b> at the gate · measured over 5 bars`
      : "Bar score · trap engine";
    $("#checklist").innerHTML = CHECKS.map(([name, w]) => {
      const ok = list && list[name];
      return `<li class="${ok ? "ok" : ""}"><span class="box"></span>${name}<span class="w">${w}</span></li>`;
    }).join("");
  }

  function renderPosition(b) {
    const el = $("#position"), cfg = S.cfg;
    if (!b) { el.innerHTML = ""; $("#pos-meta").textContent = "—"; $("#risk-note").textContent = "—"; return; }
    const p = b.pos;
    if (p) {
      const d = p.direction === "CALL" ? 1 : -1, upts = (b.c - p.avg) * d, uinr = upts * p.lots * cfg.lot_size;
      $("#pos-meta").innerHTML = `<span class="tag ${p.direction.toLowerCase()}">${p.direction}</span>`;
      el.innerHTML = [
        ["Strategy", `${p.strategy} · ${p.tier.toLowerCase()} · ${p.score}/100`],
        ["Size", `${p.lots} lot${p.lots > 1 ? "s" : ""} × ${cfg.lot_size}`],
        ["Entry", f2(p.avg)],
        [p.trailing ? "Stop · trailing" : "Stop", f2(p.stop)],
        ["Target · 2R", f2(p.target)],
        ["Open P&L", `<span class="${uinr >= 0 ? "up" : "down"}">${inr(uinr)} · ${sgn(upts / p.R)}R</span>`],
      ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    } else {
      $("#pos-meta").textContent = b.kill ? "KILL SWITCH" : "FLAT";
      el.innerHTML = `<div class="flat">${b.kill ? "Daily loss limit hit — no more trades today." : "No open position."}</div>`;
    }
    const loss = Math.max(0, (cfg.capital - b.equity) / cfg.capital * 100), lim = cfg.max_daily_loss_pct;
    const rb = $("#riskbar");
    rb.querySelector(".fill").style.width = Math.min(100, (loss / lim) * 100) + "%";
    rb.className = "riskbar" + (b.kill ? " kill" : loss > lim * 0.6 ? " warn" : "");
    $("#risk-note").textContent = `Equity ₹${nf0.format(b.equity)} · day ${inr(b.equity - cfg.capital)} · max DD ${b.dd.toFixed(2)}% · kill at −${lim}%`;
  }

  function renderAI(b) {
    const el = $("#ai"), st = S.aiStatus;
    if (!st) { el.innerHTML = `<p class="ai-off">Waiting for a replay.</p>`; $("#ai-meta").textContent = "—"; return; }
    if (!st.enabled && !S.ai.length) {
      $("#ai-meta").textContent = "off";
      el.innerHTML = `<p class="ai-off">Off — ${esc(st.reason)}.</p><p class="ai-foot">Set the key, restart the server, replay.</p>`;
      return;
    }
    $("#ai-meta").textContent = [st.provider, st.model].filter(Boolean).join(" · ") || "ai";
    const upto = b ? slotOf(b.t) : Infinity;
    const op = S.ai.filter((o) => slotOf(o.bar_time) <= upto).pop();
    if (!op) { el.innerHTML = `<p class="ai-off">No opinion yet for this point in the session.</p>`; return; }
    const cls = op.verdict === "LEAN_BUY" ? "buy" : op.verdict === "LEAN_SELL" ? "sell" : "";
    el.innerHTML = `<div class="ai-head"><span class="verdict ${cls}">${op.verdict.replace("_", " ")}</span>
      <span class="conf">${Math.round(op.confidence * 100)}%</span>
      <span class="agree">${op.error ? "error" : op.agrees_with_engine ? "agrees with engine" : "differs from engine"}</span></div>
      <p class="ai-why">${esc(op.reasoning)}</p><p class="ai-foot">for bar ${op.bar_time} · ${esc(op.model || op.provider || "")} · ${S.ai.length} opinions${
        S.summary?.ai_status?.dropped ? ` · ${S.summary.ai_status.dropped} briefs skipped while busy` : ""}</p>`;
  }

  function noteFor(e) {
    const n = [];
    if (e.type === "BREAKOUT" || e.type === "RISK") return esc(e.note || "—");
    if (e.ignored) n.push(`<span class="flag">⚑</span> opposing flow overridden`);
    if (e.ref_lis_ts) n.push(`retest of LIS high @ ${e.ref_lis_ts}`);
    if (e.rej_count) n.push(`${e.rej_count} wick rejections + ITM PE selling`);
    if (e.fib_50) n.push(`Fib 50% ${f2(e.fib_50)}`);
    if (e.type === "SKIP" || e.type === "ENTRY") n.push(esc(e.note || ""));
    if (e.type === "EXIT") n.push(`${e.lots} lot(s) · entry ${f2(e.entry)} · ${sgn(e.pnl_pts)} pts`);
    if (e.type === "TRAP") { if (e.grade) n.push(`conviction ${e.conviction} · ${e.grade}`); if (e.reason) n.push(e.reason); }
    return n.filter(Boolean).join(" · ") || "—";
  }

  function renderLedger() {
    const rows = S.events.filter((e) => S.ledgerFilter === "all" || isTrade(e));
    const key = `${S.events.length}|${S.pin}|${S.ledgerFilter}|${S.status}`;
    if (key === lastLedgerKey) return;
    lastLedgerKey = key;
    $("#ledger-count").textContent = rows.length;
    $("#ledger").innerHTML = rows.length
      ? rows.map((e) => {
          const side = sideOf(e), pnl = e.pnl_inr ?? null;
          return `<tr data-i="${e.i}" class="k-${e.type.toLowerCase()} ${e.i === S.pin ? "sel" : ""}"><td class="mono">${e.t}</td><td>${esc(labelOf(e))}</td>
            <td>${side ? `<span class="tag ${side}">${side.toUpperCase()}</span>` : "—"}</td>
            <td class="r">${e.price != null ? f2(e.price) : e.entry != null ? f2(e.entry) : "—"}</td>
            <td class="r">${e.stop != null ? f2(e.stop) : "—"}</td>
            <td class="r">${e.score != null ? e.score : "—"}</td>
            <td class="r ${pnl == null ? "" : pnl >= 0 ? "up" : "down"}">${pnl != null ? inr(pnl) : e.type === "TRAP" && e.pnl_pts != null ? sgn(e.pnl_pts, 1) + " pts" : "—"}</td>
            <td class="note">${noteFor(e)}</td></tr>`;
        }).join("")
      : `<tr class="none"><td colspan="8">${S.status === "running" ? "Listening for signals…" : S.bars.length ? "No trades — every proposal was gated. Switch to All to see why." : "No signals yet."}</td></tr>`;
  }
  $("#ledger").addEventListener("click", (ev) => {
    const tr = ev.target.closest("tr[data-i]"); if (!tr) return;
    S.pin = +tr.dataset.i; schedule();
    $("#chart-wrap").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("#ledger-filter").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button"); if (!btn) return;
    S.ledgerFilter = btn.dataset.f;
    for (const x of $("#ledger-filter").children) x.setAttribute("aria-checked", x === btn);
    schedule();
  });

  function renderSummary() {
    const s = S.summary, el = $("#summary");
    if (!s) { el.innerHTML = `<div class="empty-note">Session totals appear once the replay completes.</div>`; return; }
    const ex = (s.exec_log || []).filter((e) => e.type === "EXIT");
    const entries = (s.exec_log || []).filter((e) => e.type === "ENTRY").length;
    const gated = (s.exec_log || []).filter((e) => e.type === "SKIP").length;
    const bo = (s.exec_log || []).filter((e) => e.type === "BREAKOUT" && !e.action.endsWith("BREAK"));
    const trapEntries = s.trade_log.filter((e) => e.action === "ENTRY").length;
    const rows = [
      ["Bars processed", s.bars], ["Final regime", `${s.regime} · ${Math.round((s.regime_confidence || 0) * 100)}%`],
      ["Intraday range", `${f2(s.intraday_low)} – ${f2(s.intraday_high)}`], ["Breakouts judged", bo.map((e) => `${e.level} ${e.action.split("_")[1].toLowerCase()}`).join(", ") || "none"],
      ["Workflow entries", entries], ["Setups gated (logged)", gated],
      ["Closed trades", `${ex.length} · ${ex.filter((e) => e.pnl_inr > 0).length} won`], ["Realized P&L", inr(s.realized_pnl)],
      ["Max drawdown", `${(s.max_drawdown_pct || 0).toFixed(2)}%`], ["Kill switch", s.kill_switch ? "triggered" : `not hit (−${s.cfg?.max_daily_loss_pct}%)`],
      ["Trap-engine entries", trapEntries], ["Latency mean · p95", s.latency?.mean != null ? `${s.latency.mean} · ${s.latency.p95} ms` : "—"],
      ["AI opinions", s.ai_status?.enabled ? (s.ai_opinions || []).length : "off"], ["Capital · lot", `₹${nf0.format(s.cfg?.capital || 0)} · ${s.cfg?.lot_size}`],
    ];
    el.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  }

  // ── pipeline pulse ───────────────────────────────────────────────────────
  let lastPulse = 0;
  function pulse(first) {
    const now = performance.now();
    if (!first && now - lastPulse < 380) return;
    lastPulse = now;
    const lis = [...document.querySelectorAll("#pipeline li")];
    let n = 0;
    for (const li of lis) {
      const skip = li.classList.contains("once") && !first;
      li.classList.toggle("skip", skip);
      li.classList.remove("hit");
      if (!skip) li.style.setProperty("--i", n++);
    }
    void document.body.offsetWidth;
    for (const li of lis) if (!li.classList.contains("skip")) li.classList.add("hit");
  }

  // ── engine log ───────────────────────────────────────────────────────────
  const logEl = $("#log");
  let logDirty = false, logTimer = 0;
  function classify(ln) {
    if (/^\s*[─═]{8,}/.test(ln)) return "dim";
    if (/BAR \d\d:\d\d CLOSED|MARKET OPEN|SESSION SUMMARY/.test(ln)) return "h";
    if (/💼|ENTRY|EXIT|KILL/.test(ln)) return "warn";
    if (/WF |>>|Workflow|GATE|score=/.test(ln)) return "wf";
    if (/TRG|LIS Leg|Sweep|Volume Node|BO |AI ANALYST/.test(ln)) return "trg";
    if (/⚠|SIGNAL|ERROR/.test(ln)) return "warn";
    return "";
  }
  function renderLog() {
    logTimer = 0;
    if (!logDirty || $("[data-panel=log]").hidden) return;
    logDirty = false;
    const q = $("#log-filter").value.trim().toLowerCase();
    let lines = q ? S.log.filter((l) => l.toLowerCase().includes(q)) : S.log;
    if (lines.length > 3000) lines = lines.slice(-3000);
    logEl.innerHTML = lines.map((l) => { const c = classify(l); return c ? `<span class="${c}">${esc(l)}</span>` : esc(l); }).join("\n");
    if ($("#log-follow").checked) logEl.scrollTop = logEl.scrollHeight;
  }
  const queueLog = () => { logDirty = true; $("#log-count").textContent = S.log.length; if (!logTimer) logTimer = setTimeout(renderLog, 250); };
  $("#log-filter").addEventListener("input", () => { logDirty = true; renderLog(); });

  // ── status / streaming ───────────────────────────────────────────────────
  function setStatus(state, text) {
    S.status = state;
    $("#status").dataset.state = state;
    $("#status-text").textContent = text;
    document.body.classList.toggle("running", state === "running");
    $("#run").disabled = state === "running";
    $("#run span").textContent = state === "done" ? "Replay again" : "Run replay";
  }

  function finish(summary, error) {
    S.summary = summary;
    if (summary) for (const o of summary.late_ai || []) S.ai.push(o);
    error ? setStatus("error", "Engine error — see log") : setStatus("done", `Complete · ${S.bars.length} bars`);
    renderSummary(); lastLedgerKey = ""; schedule();
  }

  function connect() {
    if (S.es) S.es.close();
    const es = (S.es = new EventSource("/api/stream"));
    es.addEventListener("bar", (m) => {
      const b = JSON.parse(m.data);
      addBar(b);
      pulse(b.i === 0);
      setStatus("running", `Replaying · ${b.t}`);
      schedule();
    });
    es.addEventListener("log", (m) => { S.log.push(JSON.parse(m.data)); queueLog(); });
    es.addEventListener("done", (m) => {
      const d = JSON.parse(m.data);
      es.close(); S.es = null;
      finish(d.summary, d.error);
    });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED && S.status === "running") setStatus("error", "Stream lost"); };
  }

  async function run() {
    if (S.status === "running") return;
    try {
      const r = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ speed: S.speed }) });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      reset(); lastLedgerKey = ""; renderSummary();
      setStatus("running", "Starting…");
      schedule(); connect();
    } catch (e) { setStatus("error", String(e.message || e)); }
  }

  async function init() {
    readColors(); renderSummary(); schedule();
    try {
      const r = await (await fetch("/api/results")).json();
      if (r.status === "running") { reset(); setStatus("running", "Rejoining…"); connect(); }
      else if (r.status === "done") {
        r.bars.forEach(addBar); S.log = r.log;
        $("#log-count").textContent = S.log.length; logDirty = true;
        finish(r.summary, r.error);
      }
    } catch (e) { setStatus("error", "Server unreachable"); }
  }

  // ── controls ─────────────────────────────────────────────────────────────
  $("#run").addEventListener("click", run);
  $("#speed").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button"); if (!btn) return;
    S.speed = btn.dataset.speed;
    for (const b of $("#speed").children) b.setAttribute("aria-checked", b === btn);
  });
  function toggleTheme() {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("trapx-theme", root.dataset.theme); } catch (e) {}
    readColors(); schedule();
  }
  $("#theme").addEventListener("click", toggleTheme);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); schedule(); });

  document.querySelectorAll(".tabs [role=tab]").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tabs [role=tab]").forEach((t) => t.setAttribute("aria-selected", t === tab));
    document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
    $("#log-tools").hidden = tab.dataset.tab !== "log";
    $("#ledger-filter").hidden = tab.dataset.tab !== "ledger";
    if (tab.dataset.tab === "log") { logDirty = true; renderLog(); }
  }));

  document.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, textarea") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const n = S.bars.length;
    if (ev.key === "r" || ev.key === "R") run();
    else if (ev.key === "t" || ev.key === "T") toggleTheme();
    else if (ev.key === "Escape") { S.pin = null; schedule(); }
    else if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && n) {
      const cur = S.pin ?? n - 1;
      S.pin = Math.max(0, Math.min(n - 1, cur + (ev.key === "ArrowLeft" ? -1 : 1)));
      ev.preventDefault(); schedule();
    }
  });

  new ResizeObserver(schedule).observe($("#chart-wrap"));
  document.fonts?.ready.then(() => { readColors(); schedule(); });
  init();
})();
