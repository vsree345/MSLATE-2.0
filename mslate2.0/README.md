# DRAVYAGATI (TRAP-X): Agentic Trading Engine for Institutional Trap Detection

> **An agent that remembers the whole trading day, reads spot, futures and options together, and only trades a breakout once enough evidence says it is a trap.**

DRAVYAGATI (engine codename **TRAP-X**, project folder `mslate2.0`) is a stateful, agent-based intraday trading engine for the **NIFTY 50** index, built on **LangGraph**. It replays real 1-minute market data one bar at a time. At every bar it decides whether a breakout of yesterday's high or low is **real** or a **trap set by institutions**. It trades the snap-back only when a 6-factor **Conviction Score** reaches 70/100 or more.

Major Project, Department of Computer Science & Engineering, Vasavi College of Engineering (Autonomous), Hyderabad, 2026.
**Team:** Sree Vaibhav Duvvuri · Lokesh Bethala · Kshitij Kumar Sona  **Guide:** Dr. M. Sunitha Reddy<br>
**Co-authors:** Kshitij Sona · Lokesh Naidu

---

## Table of Contents

1. [The Problem: Institutional Traps](#1-the-problem-institutional-traps)
2. [What DRAVYAGATI Does](#2-what-dravyagati-does)
3. [What Makes It Different](#3-what-makes-it-different)
4. [Why It Matters for Agentic Quant Trading](#4-why-it-matters-for-agentic-quant-trading)
5. [Who It Helps](#5-who-it-helps)
6. [Project Status: Simulation, Not Live Trading](#6-project-status-simulation-not-live-trading)
7. [Project Layout](#7-project-layout)
8. [Setup and Running](#8-setup-and-running)
9. [Architecture](#9-architecture)
10. [The LangGraph Pipeline (13 Nodes)](#10-the-langgraph-pipeline-13-nodes)
11. [Strategies](#11-strategies)
12. [Quant Formulas](#12-quant-formulas)
13. [Results: 20 Feb 2026](#13-results-20-feb-2026)
14. [Dashboard Guide](#14-dashboard-guide)
15. [Demo Walkthrough: Good Trade vs Bad Trade](#15-demo-walkthrough-good-trade-vs-bad-trade)
16. [Data Files](#16-data-files)
17. [Configuration Reference](#17-configuration-reference)
18. [Reading the CLI Output](#18-reading-the-cli-output)
19. [Limitations](#19-limitations)
20. [Roadmap](#20-roadmap)
21. [Troubleshooting](#21-troubleshooting)
22. [Report / Slides vs Code](#22-report--slides-vs-code)
23. [References](#23-references)

---

## 1. The Problem: Institutional Traps

Many retail traders watch two lines every day:

- **PDH (Previous Day High):** the highest price of the previous session, a "ceiling"
- **PDL (Previous Day Low):** the lowest price of the previous session, a "floor"

Their rule is simple: *price breaks above PDH → buy; price breaks below PDL → sell.*

```
25,885  ─────────────  PDH (yesterday's ceiling)
          yesterday's
          price range
25,388  ─────────────  PDL (yesterday's floor)
```

Institutions know where retail orders and stop-losses sit. So they:

1. Push the price **slightly** through the level (e.g. just below PDL).
2. Retail traders sell the "breakdown", and retail buyers' stop-losses are triggered.
3. Institutions fill their large buy orders against that liquidity at good prices.
4. Price **reverses sharply** and retail traders are left on the wrong side.

```
24,100              ↗  snap-back (trap)
24,000  ──────\────/──────  PDL
23,970         \__/   ← fake break: retail sold here
```

This engineered false breakout is a **liquidity trap**. In the moment it looks almost identical to a real breakout.

**Why existing systems fail:**

| Approach | Weakness |
|---|---|
| Rule-based bots (fixed IF-THEN) | Cannot adapt to market regime; fire on every breakout |
| Indicators (RSI, MACD, MA) | Lagging; react to price, no idea of liquidity or market structure |
| ML / deep learning models | Black box, need large labelled data, no real-time reasoning |
| Execution platforms (e.g. order routers) | Focus on fast order placement, not on trap detection |
| All of the above | **Stateless**: each candle is judged in isolation, with no memory of the session |

**Core problem:** detect institutional traps in real time using multi-factor evidence and persistent session memory.

---

## 2. What DRAVYAGATI Does

At every 1-minute bar the agent:

1. **Reads three markets at once:** NIFTY spot, NIFTY futures and the options chain (ITM / ATM / OTM call and put sentiment).
2. **Updates a session memory:** TWAP, ATR, big-player candles (LIS), volume nodes, impulse legs, liquidity sweeps and the market regime. Nothing is forgotten between bars.
3. **Validates every PDH/PDL break** as `PENDING → FAKE` (trap) or `PENDING → REAL`.
4. **Matches the situation to a strategy** (S1–S5) that *proposes* a trade.
5. **Scores the proposal 0–100** on six weighted factors. Only **≥ 70** is executed (70–79 half size, ≥ 80 full size).
6. **Sizes and manages the trade:** risk-based lots, 2R target, trailing stop, 15:15 square-off and a daily-loss kill switch.
7. **Asks an AI analyst for a second opinion** (advisory only; it never places trades).
8. **Explains itself:** every decision carries its checklist, visible in the ledger, chart, log and dashboard.

**Promises and where they live in code:**

| # | Promise | Implementation |
|---|---|---|
| 1 | Spots fake breakouts | `breakout_validator` marks every PDH/PDL break PENDING → FAKE (close reclaims by ≥ 0.5×EM within 10 bars) or REAL (5 closes beyond, ≥ 3×EM extension, or no reclaim) |
| 2 | Trades the snap-back | S1 (fake PDL → call) and S5 (fake PDH → put); trap structure in the score requires a FAKE verdict |
| 3 | Only trades ≥ 70 | Strategies only *propose*; `conviction_gate` scores each setup over a 5-bar window. ≥ 80 full size, 70–79 half, < 70 logged as gated |
| 4 | Remembers the day | LangGraph state + `MemorySaver` across all bars |
| 5 | Spot + futures + options | Futures displacement and per-minute volume, options ladder and flow in every score |
| 6 | Explains decisions | Every gate decision carries its six-factor checklist; ledger, chart and log show why |
| 7 | Manages the trade | `position_manager`: gap-aware stop, 2R target, trailing stop after +1R (1.5×EM), 15:15 square-off, reversal handling |
| 8 | AI second opinion | `ai_analyst` → `trapx_ai.py`; async, every 15 bars and on every gate decision; advisory only |
| 9 | Keeps losses small | Risk-based sizing (1% full / 0.5% half, 2% cap per trade) and a 2.5% daily-loss kill switch; drawdown tracked per bar |
| 10 | Fast | Checkpoint once per bar (`durability="exit"`); ~10–15 ms mean, ~20–25 ms p95 per bar |

---

## 3. What Makes It Different

- **Detective, not alarm.** A typical bot fires on a single trigger. DRAVYAGATI needs several independent clues to agree before it acts, so it ignores fake breakouts that fool simple systems.
- **Session memory.** Most retail bots judge each candle on its own. DRAVYAGATI carries the whole day's context (TWAP, ATR, LIS zones, regime, open positions) into every decision.
- **Reads the big players' footprints.** Institutions show up in futures and options before the index moves. Fusing spot, futures and options into one score is rare in retail tooling.
- **Turns the trap around.** It doesn't just avoid traps; it trades the reversal alongside the institutions.
- **Explainable by design.** Every trade, and every skipped setup, comes with a score and a ticked/unticked checklist. There's no black box.
- **AI as advisor, not boss.** Deterministic, testable rules make the decision. The LLM adds a reasoned second opinion on top, which is safer and auditable.
- **Adapts to volatility automatically.** Every threshold is expressed in the VIX-derived *expected move per minute*, so the same rules work on calm and volatile days.

---

## 4. Why It Matters for Agentic Quant Trading

"Agentic" trading means software that **observes → remembers → reasons → acts**, step by step, instead of running a fixed script. DRAVYAGATI is a working example of that pattern applied to a concrete market problem:

- **Modular agent graph.** Each responsibility (ingest, memory, regime, trigger, validation, strategy, scoring, execution, lifecycle, AI) is its own LangGraph node sharing one state object. A new strategy or checker is one new node, not a rewrite.
- **Hybrid quant + LLM template.** Math sets the guardrails; the LLM gives a calibrated opinion with reasoning. This is a practical answer to "how do we safely combine rules with AI?"
- **Auditability.** `MemorySaver` checkpoints the full state after every bar, so you can replay and inspect exactly what the agent knew at any minute. Risk teams and regulators care about this.
- **Path to production.** The engine is data-source agnostic. Swapping CSV replay for a live WebSocket feed and a broker order-management module turns it into a live agent without changing the decision logic.
- **Extensible to multi-agent setups.** Risk agent, news/sentiment agent and execution agent can be added as further nodes on the same shared state.

---

## 5. Who It Helps

| User | Their pain | What DRAVYAGATI gives them |
|---|---|---|
| **Retail traders** | Buy breakouts that immediately reverse; trade on FOMO; can't read futures and options; no exit discipline | Trap warnings, a ≥ 70 evidence filter, one combined score, reasons for every decision, automatic stop / target / square-off |
| **Strategy testers and small firms** | Need to test ideas safely; black-box backtesters; hard to extend | Bar-by-bar replay, full decision transparency, plug-in node architecture |
| **Educators and mentors** | Hard to teach "reading the trap" | Replay a day and point at the exact minute and clue that exposed it |
| **Brokers / platforms (future)** | Retail clients churn after losses | A "trap alert" feature that can be embedded in a trading app |

> SEBI studies have reported that the large majority of individual F&O traders in India lose money. False breakouts are one of the patterns behind those losses, and this is the problem DRAVYAGATI targets.

---

## 6. Project Status: Simulation, Not Live Trading

DRAVYAGATI is currently a **historical-replay simulation engine** with a **web dashboard** as its viewing layer. Think of a flight simulator: real physics (market rules), recorded conditions (20 Feb 2026 data), no real plane (no real money), and a cockpit screen (the dashboard).

The engine only sees data up to the current bar, so decisions are made exactly as they would be live, with no look-ahead.

```
1. Simulation on recorded days     ← current stage
2. Multi-month backtesting
3. Paper trading (live feed, fake money)
4. Live trading via broker API
```

| Capability | Status |
|---|---|
| Trap detection, conviction scoring, strategies | ✅ Built |
| Session memory and checkpointing | ✅ Built |
| Spot + futures + options fusion | ✅ Built |
| Explainable decisions (ledger, checklist, log) | ✅ Built |
| Automated trade management | ✅ Built (simulated) |
| AI second opinion | ✅ Built (async during replay; advisory) |
| Multi-month validation | ❌ Planned |
| Live feed and broker execution | ❌ Planned |

---

## 7. Project Layout

```
mslate2.0/
├── TrapX_agent.py                ← Main engine: LangGraph nodes, CFG, run_replay(), CLI
├── trapx_ai.py                   ← AI analyst (Gemini / Claude Code CLI / Anthropic SDK)
├── dashboard/
│   ├── server.py                 ← FastAPI: /api/simulate, /api/stream (SSE), /api/results
│   └── index.html · styles.css · app.js   ← Web dashboard (vanilla JS, no build step)
├── data/
│   ├── spot_fut_2026-02-20.csv   ← 1-min OHLC for NIFTY SPOT + FUTURE
│   ├── signals_2026-02-20.csv    ← Per-minute options-flow signal rows
│   ├── pdc_2026-02-20.csv        ← Previous-day context (PDH/PDL/PDC, VIX)
│   └── …                         ← options_oi, squeeze_signals, vix_data, etc.
├── data_sources/                 ← Scripts to fetch more historical data (HF, NSE bhavcopy, yfinance)
├── tests/
│   └── test_promises.py          ← 16 tests covering the 10 promises
├── requirements.txt
├── WorkFlow Enhancements.xlsx    ← Strategy spec (reference only)
├── Batch_21_Project_report.pdf   ← Project report
├── DRAVYAGATI.pptx               ← Project presentation
└── logs/                         ← Auto-created; timestamped CLI logs
```

---

## 8. Setup and Running

### Prerequisites

Python 3.10+.

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

### Web dashboard (recommended)

```bash
venv\Scripts\python dashboard\server.py
```

Open **http://localhost:8000** and press **Run replay** (or `R`). Bars stream in live over Server-Sent Events.

| Control | Action |
|---|---|
| `1× / 4× / Max` | Replay speed |
| Hover chart | Inspect a bar |
| Click chart / ledger row | Pin that bar; every panel follows it |
| `←` / `→` | Step one bar |
| `Esc` | Return to the latest bar |
| `T` | Toggle light / dark theme |

### CLI

```bash
venv\Scripts\python TrapX_agent.py
```

The engine will:
1. Load the CSV files from `data/`
2. Start a LangGraph-Viz server at http://localhost:8767 if `langgraph-viz` is installed (otherwise it runs headless)
3. Replay the full session (09:15–15:29), printing every node's output
4. Print a session summary at the end
5. Save all console output to `logs/trapx_YYYYMMDD_HHMMSS.log`
6. Keep the viz server alive until you press **Ctrl-C**

### AI analyst (optional)

Providers are tried in order (`TRAPX_AI_PROVIDER=auto | gemini | claude-cli | anthropic`):

1. **Gemini:** `GEMINI_API_KEY` from the environment or the gitignored `.env`. Default model `gemini-flash-lite-latest` (override with `TRAPX_AI_MODEL`). On a free key expect ~20–30 s per call.
2. **Headless Claude Code:** `claude -p` with your local login; used automatically if Gemini is unavailable.
3. **Anthropic SDK:** `ANTHROPIC_API_KEY`.

Calls run off the main loop with at most two in flight; while busy only the newest brief is kept, so a fast replay never builds a backlog. `TRAPX_AI=off` disables the analyst.

The analyst receives the last 30 candles, PDH/PDL/PDC, VIX, expected move, TWAP, regime, breakout states, an options-flow snapshot, the engine's position and any setups just scored. It returns `LEAN_BUY / LEAN_SELL / WAIT`, a confidence (0–1), whether it agrees with the engine, and two or three sentences of reasoning.

### Tests

```bash
venv\Scripts\python -m pytest tests -q
```

---

## 9. Architecture

### Three tiers

| Tier | Component | Role |
|---|---|---|
| **Frontend** | `dashboard/index.html`, `app.js`, `styles.css` | Trading-terminal UI: chart, pipeline, strategies, conviction checklist, ledger, log |
| **Backend** | `dashboard/server.py` (FastAPI) | `/api/simulate` starts a replay on a background thread, `/api/stream` streams bars over SSE, `/api/results` returns the final JSON |
| **Engine** | `TrapX_agent.py` (LangGraph) | The decision-making agent graph |

### State and memory

- **`TrapXState`** (`TypedDict`, `TrapX_agent.py:134`) is the shared "notebook" every node reads and writes: TWAP, ATR, expected move, LIS legs, volume nodes, impulse legs, sweeps, regime, breakout states, strategy candidates, gate decisions, position, equity, drawdown, exec log, AI opinions.
- **`MemorySaver`** checkpoints the state after every bar, so each invocation sees the full session context and any bar can be inspected afterwards.
- **Module-level accumulators** (`G_SPOT`, `G_FUT`, `G_SIG`, `G_PDC`) hold raw bar history outside LangGraph state to keep serialisation fast.

### Bar timing

- `bar_time` is the closed candle being analysed.
- `trigger_time` is the live-clock minute when analysis runs (always bar + 1 min). Execution uses data that would actually be available, so there's no look-ahead.

---

## 10. The LangGraph Pipeline (13 Nodes)

Every 1-minute bar is one graph invocation (`create_trapx_app`, `TrapX_agent.py:1588`):

```
START
  |
  +-- bar_index == 0 --> process_mkt_open     (PDC load, gap analysis, expected move; 09:15 only)
  |                           |
  +-- bar_index  > 0 ---------+
                              |
                       process_minute         (read this bar's spot / futures / options)
                   market_memory_engine       (TWAP, ATR, LIS legs, volume nodes, impulse, sweeps)
                    regime_detector           (TREND / MEAN / UNDEFINED)
                    check_trigger             (PDL/PDH/PDC breaks, new LIS / volume nodes)
                  breakout_validator          (PDH/PDL break → PENDING / FAKE / REAL)
              workflow_strategy_engine        (S1–S5 propose setups)
               trap_trigger_engine            (per-bar conviction 0–100, trap structure)
                  conviction_gate             (score each setup; ≥ 70 executes)
                  position_manager            (size, fill, stop / target / trail / EOD, kill switch)
             trade_lifecycle_manager          (trap-engine trades: stop / time / EOD)
                     ai_analyst               (async LLM second opinion)
                   minute_summary             (one-line bar summary)
                              |
                             END
```

| Node | What it does in plain words |
|---|---|
| `process_mkt_open` | Loads yesterday's levels and VIX, computes the opening gap and the expected move per minute |
| `process_minute` | Picks up the current minute's prices |
| `market_memory_engine` | Updates the day's average price, normal swing size, big-player candles, heavy-volume prices, strong runs and stop hunts |
| `regime_detector` | Decides whether the day is trending or bouncing around its average |
| `check_trigger` | Has price crossed yesterday's high, low or close? New big candles? |
| `breakout_validator` | Tracks each break until it is proven fake (trap) or real |
| `workflow_strategy_engine` | Checks the four (+ mirror) setups and proposes trades |
| `trap_trigger_engine` | Scores the current bar's evidence (shown as the dashboard's conviction strip) |
| `conviction_gate` | Scores each proposed setup; only ≥ 70 passes |
| `position_manager` | Opens, sizes, trails and closes trades; enforces risk limits |
| `trade_lifecycle_manager` | Manages trap-engine trades: stop, time stop, end of day |
| `ai_analyst` | Sends a market brief to the LLM and collects opinions |
| `minute_summary` | Writes the one-line summary for the log and dashboard |

---

## 11. Strategies

All strategies live in `workflow_strategy_engine`. They only **propose**; `conviction_gate` decides.

| Setup | Direction | Entry condition | Stop-loss |
|---|---|---|---|
| **S1 · Bounce PDL break** | CALL | PDL broken → UP LIS candle with body > 2×EM → price holds above the LIS open (×0.999), within 15 bars of the LIS | LIS low − EM |
| **S5 · Reject PDH break** | PUT | Mirror of S1: PDH broken → DOWN LIS body > 2×EM → price stays below the LIS open (×1.001) | LIS high + EM |
| **S2 · LIS high retracement** | CALL (pyramid) | Already in a CALL; low comes within 1.5×EM of a post-entry UP LIS high; ≥ 10 min since last pyramid | Existing stop |
| **S3 · Top trade, less SL** | PUT | After 12:30; ≥ 2 upper-wick rejections near the day high; options signal bearish and ITM put sentiment < −2.0 | Intraday high + 0.5×EM |
| **S4 · Fibonacci retracement bottom** | CALL | After 12:30; low within 2×EM of Fib 50%; UP LIS in the last 4 min; CE squeeze or OTM support | Fib 50% − EM |

**Options-signal overrides:**
- A bearish options signal is **ignored for a CALL** if OTM put sentiment > 0 (put writers adding), a CE squeeze is present (ITM CE < 0 and OTM CE > 0), or `otm_support > 0`.
- A bullish options signal is **ignored for a PUT** if OTM call sentiment > 0 or a PE squeeze is present (ITM PE < 0 and OTM PE > 0).

**Why Fib 50%:** after a morning impulse, the 50% retracement is a classic institutional re-entry zone; the options squeeze confirmation reduces false entries.

No new entries after **15:00**; all positions are squared off at **15:15**.

---

## 12. Quant Formulas

All formulas are as implemented in `TrapX_agent.py`.

### 12.1 Expected move per minute (EM): the master unit

```
EM = PDH × (VIX / 100) × √(1 / 94,500) × 1.5
```

- `94,500` = 375 trading minutes/day × 252 trading days/year
- `VIX / 100` = implied annual volatility
- `√(1 / 94,500)` scales annual volatility to one minute (square-root-of-time rule)
- `× 1.5` = buffer so a "significant" move is 1.5× a normal one

**20 Feb 2026:** 25,885.30 × 0.1346 × 0.003253 × 1.5 ≈ **17.00 pts** (11.33 before the 1.5× factor).

Almost every threshold is expressed in multiples of EM, so the system scales with volatility.

### 12.2 Opening gap

```
Gap (pts) = Open(09:15) − PDC
Gap %     = Gap / PDC × 100
Gap > +5 → GAP UP · Gap < −5 → GAP DOWN · otherwise FLAT
```

The same is computed for futures.

### 12.3 Market memory

```
Typical price = (High + Low + Close) / 3
TWAP          = mean(Typical price over all bars so far)

True Range    = max(High − Low, |High − PrevClose|, |Low − PrevClose|)
ATR           = mean(True Range over the last 14 bars)

Minute volume = CumVolume(t) − CumVolume(t − 1)      (feed volume is cumulative)
Body %        = |Close − Open| / (High − Low)
```

### 12.4 Institutional footprints

**LIS (Large Impulse Sequence) candle**
```
LIS if |Close − Open| > EM            (S1 / S5 require body > 2 × EM)
```

**Volume node**
```
Node if MinuteVol > 2.2 × mean(MinuteVol over previous 20 bars)  or  MinuteVol > 125,000
Strength(t+1) = Strength(t) × 0.97     (dropped when < 0.30)
```

**Impulse leg**
```
Move %   = (Close(t) − Close(t−5)) / Close(t−5)
Pullback = distance from 5-bar extreme / 5-bar range
Impulse if |Move %| ≥ 0.35%  and  Pullback < 50%
Invalidated if retrace ≥ 61.8% of the leg
```

**Liquidity sweep** (expires after 90 min)
```
Bearish: High > PrevHigh  and  Close < PrevHigh
Bullish: Low  < PrevLow   and  Close > PrevLow
```

**TWAP stretch**
```
Stretched if |Close − TWAP| > 0.6 × ATR
```

### 12.5 Regime detection

```
Range ratio = (Intraday high − Intraday low) / (PDH − PDL)
```

| Trend evidence | Pts | Mean-reversion evidence | Pts |
|---|---|---|---|
| Range ratio > 0.9 | 40 | TWAP crossings ≥ 3 | 40 |
| > 45 min on one side of TWAP | 35 | Candle overlap (30 bars) > 65% | 35 |
| Last-5-bar futures volume > session average | 25 | No active impulse leg | 25 |

```
Trend score ≥ 60 → TREND      Mean score ≥ 60 → MEAN      else UNDEFINED
Confidence = score / 100
Candle overlap = mean over 30 bars of  overlap(body_i, body_{i−1}) / size(body_{i−1})
```

### 12.6 Breakout validation

```
Break:  High > PDH  (or Low < PDL)                          → PENDING ("don't chase")
FAKE:   close back inside the level by ≥ 0.5 × EM           → trap, snap-back setups armed
REAL:   5 consecutive closes beyond  or  extension ≥ 3 × EM  or  10 bars without reclaim
```

A FAKE level can re-arm after 3 bars.

### 12.7 Strategy geometry

```
Upper wick     = High − max(Open, Close)
Wick rejection = Upper wick / (High − Low) > 0.40  and  (Intraday high − High) < EM

Fib 50%        = Swing low + 0.5 × (Swing high − Swing low)
                 (computed once, only if Swing high − Swing low > 5 × EM)

CE squeeze     = ITM call sentiment < 0  and  OTM call sentiment > 0
PE squeeze     = ITM put sentiment  < 0  and  OTM put sentiment  > 0
```

### 12.8 Conviction score (`_setup_score`, `conviction_gate`)

Measured over a 5-bar window, with direction d = +1 for CALL and −1 for PUT:

```
Conviction = Σ weight_i × clue_i          (capped at 100)
```

| Clue | Formula | Weight |
|---|---|---|
| Futures displacement | `(Fut(t) − Fut(t−5)) / Fut(t−5) × 100 × d ≥ 0.12` | 30 |
| Option ladder | ≥ 3 of 4 levels (ITM, ATM, OTM, total) where Δcall sentiment × d > 0 **and** ≥ 3 where Δput sentiment × d < 0 | 25 |
| Volume expansion | `mean(MinuteVol, last 5) > 1.5 × mean(MinuteVol, prior 20)` | 20 |
| Delta acceleration | `accel = Σ w_level` for levels where `Δcall × d / |call(t−5)| > 1.5%`, with w = ITM 0.4, ATM 0.3, OTM 0.2, total 0.1. Points = `10 × min(accel / 0.5, 1)` (partial credit) | ≤ 10 |
| Trap structure | Breakout validator verdict FAKE (or the setup's own trap check) | 10 |
| Regime alignment | Regime MEAN, or TREND with price on the trade's side of TWAP / an aligned active impulse | 5 |

```
Score ≥ 80  → FULL size
70–79       → HALF size
< 70        → SKIP (logged as gated)
```

### 12.9 Position sizing

```
R             = |Entry − Stop|
Risk per lot  = R × 65                      (NIFTY lot size, ₹ per point)
Risk budget   = Equity × 1.0%   (FULL)      or   Equity × 0.5%   (HALF)
Lots          = floor(Risk budget / Risk per lot)
                if 0 → 1 lot only if Risk per lot ≤ 2% of equity
                capped so the trade cannot breach the 2.5% daily-loss limit
Pyramid (S2)  = max(1, base lots // 2);  avg = (avg × lots + entry × add) / (lots + add)
```

### 12.10 Trade management

```
Target         = Entry + d × 2 × R
Trail starts   when (Peak − Entry) × d ≥ 1 × R
Trailing stop  = Peak − d × 1.5 × EM      (never worse than entry; only tightens)
Same-bar stop and target → stop assumed first (worst case)
Gap through stop          → filled at the open
No entries after 15:00; square-off at 15:15; opposite signal → REVERSAL exit
```

### 12.11 P&L and risk controls

```
P&L (pts)    = (Exit − Avg entry) × d
P&L (₹)      = P&L (pts) × lots × 65
R-multiple   = P&L (pts) / R
Equity       = ₹10,00,000 + realised P&L + unrealised P&L
Drawdown %   = (Peak equity − Equity) / Peak equity × 100
Kill switch  : (₹10,00,000 − Equity) / ₹10,00,000 ≥ 2.5%  → flatten and stop for the day
```

P&L is index points × lot size on ₹10,00,000 notional, a **delta-1 proxy**. Option premium, theta and IV are not modelled.

---

## 13. Results: 20 Feb 2026

Full session replay (09:15–15:29, 371 bars):

| Metric | Value |
|---|---|
| PDH / PDL / PDC | 25,885.30 / 25,388.75 / 25,454.35 |
| India VIX | 13.46 |
| Expected move / min | 17.00 pts |
| Intraday high / low / range | 25,662.90 / 25,381.25 / 281.7 pts |
| TWAP crossings | 11 |
| Final regime | TREND (60%) |
| Trades taken | **1** |
| Realised P&L | **−₹1,056 (−0.14R)** |
| Max drawdown | **0.76%** |
| Graph latency | ~15–30 ms per bar |

### Decision timeline

| Time | Event | Score | Decision |
|---|---|---|---|
| 09:15 | PDL 25,388.75 broken → reclaimed to 25,465 in the same bar | — | **PDL_FAKE** (trap); "don't chase" |
| 09:16 | S1 CALL proposed | 10 | Skipped |
| 09:21 | S1 CALL proposed | 20 | Skipped |
| 09:26 | S1 CALL proposed | 10 | Skipped |
| 09:29–09:30 | PDL broken again → reclaimed after 1 bar | — | **PDL_FAKE** |
| 12:44 | S3 PUT proposed (futures, delta, trap, regime ✔) | 49 | Skipped |
| 12:49 | S3 PUT proposed | 10 | Skipped |
| 12:54 | S3 PUT proposed | 15 | Skipped |
| 12:59 | S3 PUT proposed (ladder, delta, trap, regime ✔) | 50 | Skipped |
| **13:10** | **S3 PUT @ 25,553.15** (futures, volume, delta, trap, regime ✔) | **71** | **HALF size, 1 lot; stop 25,671.40; target 25,316.65; risk ₹7,686 (0.77%)** |
| 15:15 | EOD square-off @ 25,569.40 | — | −16.25 pts, −₹1,056 |

### Honest reading of the result

- Both PDL breaks were correctly identified as traps, and the engine refused to chase them.
- The filter is strict: the 09:16 S1 CALL (score 10) would have been profitable, since price later reached ~25,663 without hitting its stop. Calibrating that trade-off needs multi-day data.
- The one trade taken was a small, well-sized loss, closed on schedule. Good process does not guarantee a profitable trade, but it keeps losses small.
- **One day proves nothing statistically.** Multi-month validation is the next step.

---

## 14. Dashboard Guide

| Area | What it shows |
|---|---|
| **Top bar** | Market, date, "Historical replay", run status (`Complete · 371 bars`), speed, Replay again, theme toggle |
| **Session headline** | Price at the inspected bar, change vs PDC, and a plain-English sentence describing the agent's state (e.g. "Holding a put from S3 since 13:10. Conviction 35/100, so the trap engine stands aside. Regime reads mean.") |
| **Key figures** | PDH, PDL, PDC, India VIX, Exp. move/min, Regime, Conviction, Position, P&L, Drawdown, Bars, Graph latency |
| **Price & structure chart** | 1-min candles, TWAP (purple), PDH / PDC / PDL and Fib 50% levels (dotted), call ▲ / put ▼ entry markers; OHLC / TWAP / ATR / Fut / Flow readout for the inspected bar |
| **Workflow position strip** | When a CALL or PUT was open |
| **Conviction strip** | Per-bar conviction with the dotted `70 · act` line; only bars crossing it can trade |
| **Options flow strip** | Final options signal per bar (green = bullish, red = bearish) |
| **Pipeline** | The node chain and per-bar latency |
| **Strategies** | S1–S4 status (e.g. "PDL broke 09:15", "Needs a call", "Fired 13:10", "Fib 25,525.38") |
| **Conviction checklist** | The six clues ticked/unticked for the inspected bar, with score and verdict (AVOID / HALF / FULL) |
| **Ledger** | Trades (filter **Trades**) or every event including skipped setups and breakout verdicts (filter **All**); click a row to jump to that bar |
| **Engine log** | Every node message; filterable (e.g. `WF`, `TRG`, `SCORE`) |
| **Session & method** | Session statistics and the EM formula |

---

## 15. Demo Walkthrough: Good Trade vs Bad Trade

1. Start the dashboard, set speed to **Max**, press **Run replay**.
2. Set the Ledger filter to **All**.

**Scene 1: the bad trade (09:15).** Click the `PDL_BREAK` row. Price fell below PDL 25,388.75 and closed back at 25,465 in the same minute → `PDL_FAKE`. Anyone who sold the breakdown was trapped; the engine logged "validating, don't chase".

**Scene 2: same setup, rising evidence (12:44 → 12:59 → 13:10).** Click each S3 row and watch the checklist fill:

| Time | Score | Missing clues | Decision |
|---|---|---|---|
| 12:44 | 49 | Option ladder, volume | Skip |
| 12:59 | 50 | Futures, volume | Skip |
| 13:10 | 71 | Option ladder only | Trade (half) |

The difference between a bad and a good entry is the evidence, not luck.

**Scene 3: risk control (15:15).** Click `EXIT_EOD`: −16.25 pts, −0.14R, −₹1,056, half size, max drawdown 0.76%. Even good trades can lose; the loss stays small.

| Bad trade looks like | Good trade looks like |
|---|---|
| Price crosses yesterday's line, nothing else | Cross + confirmed trap |
| 1–2 clues (< 70) | 5–6 clues (≥ 70) |
| Futures and volume quiet | Futures displaced, volume expanding |
| Taken on excitement | Taken on agreement of evidence |
| No exit plan | Stop, target and size set in advance |

---

## 16. Data Files

### `spot_fut_2026-02-20.csv`

| Column | Description |
|---|---|
| `timestamp` | e.g. `20-02-2026 09:15` |
| `instrument_type` | `SPOT` or `FUTURE` |
| `instrument_token` | Exchange token |
| `open / high / low / close` | Price |
| `volume` | Cumulative futures volume (per-minute = difference) |
| `oi` | Open interest |
| `tick_count` | Ticks in the bar |

### `signals_2026-02-20.csv`

| Column | Meaning |
|---|---|
| `final_signal_value` | Net options-flow signal (+ bullish, − bearish) |
| `jerk` | % rate of change of trending OI |
| `itm_/atm_/otm_call_sentiments_sum` | Call sentiment by moneyness |
| `itm_/atm_/otm_put_sentiments_sum` | Put sentiment by moneyness (ITM PE < 0 = PE burning / squeeze) |
| `call_sentiments_sum`, `put_sentiments_sum` | Totals |
| `otm_support` | Composite OTM structural support flag (> 0 = supportive) |
| `reversal_warning` | Reversal flag from the options-flow model |

### `pdc_2026-02-20.csv`

Three rows: **NIFTY 50** (PDH / PDL / PDC), **NIFTY FUTURE** (futures PDH / PDL / PDC), **INDIA VIX** (close, used for EM).

### `data_sources/`

Fetch scripts for extending to multi-day data: Hugging Face NIFTY datasets, NSE bhavcopy, yfinance (historical and live polling). Check each source's licence before redistributing data; several are non-commercial or unstated.

---

## 17. Configuration Reference

All parameters live in the `CFG` dict at the top of `TrapX_agent.py`.

| Group | Parameter | Default | Effect |
|---|---|---|---|
| Session | `first_bar_time` / `session_end` | 09:15 / 15:30 | Replay window |
| Memory | `volume_node_multiplier` | 2.2 | Volume-node threshold vs 20-bar average |
| | `volume_node_decay` / `volume_node_expire` | 0.97 / 0.30 | Node fade rate and removal level |
| | `vol_threshold` | 125,000 | Absolute volume-node threshold |
| | `impulse_min_pct` / `impulse_min_duration` | 0.35% / 5 bars | Impulse-leg detection |
| | `impulse_invalidation_retrace` | 0.618 | Impulse invalidation |
| | `sweep_expiry_minutes` | 90 | Liquidity-sweep memory |
| | `vwap_stretch_atr_mult` | 0.6 | TWAP stretch threshold |
| | `atr_lookback` / `overlap_lookback` | 14 / 30 | Window lengths |
| Regime | `trend_day_range_pct` | 0.9 | Range ratio for trend points |
| | `trend_day_vwap_unclaimed` | 45 | Minutes on one side of TWAP |
| | `mean_day_overlap_pct` / `mean_day_vwap_crossings` | 0.65 / 3 | Mean-reversion evidence |
| Breakouts | `bo_fake_window` | 10 | Bars a break has to prove itself |
| | `bo_reclaim_em` | 0.5 | Reclaim distance (×EM) → FAKE |
| | `bo_real_hold_bars` / `bo_real_extension_em` | 5 / 3.0 | REAL criteria |
| | `s_entry_valid_bars` | 15 | S1/S5 expiry after the LIS |
| Scoring | `futures_displacement_pct` | 0.12 | Futures clue threshold (%) |
| | `futures_volume_mult` | 1.5 | Volume clue threshold |
| | `option_ladder_pass` | 3 | Levels needed for the ladder clue |
| | `weight_*` | 30 / 25 / 20 / 10 / 10 / 5 | Clue weights |
| | `setup_window` | 5 | Bars a setup is scored over |
| | `conviction_threshold` / `gate_full_score` | 70 / 80 | Half / full size thresholds |
| Trade mgmt | `target_r_multiple` | 2.0 | Target in R |
| | `trail_after_r` / `trail_em_mult` | 1.0 / 1.5 | Trailing-stop activation and distance |
| | `no_new_entries_after` / `eod_squareoff` | 15:00 / 15:15 | Session cut-offs |
| | `time_stop_candles` | 3 | Trap-engine time stop |
| Risk | `capital` | ₹10,00,000 | Notional capital |
| | `lot_size` | 65 | NIFTY lot |
| | `risk_per_trade_pct` / `max_risk_per_trade_pct` | 1.0 / 2.0 | Risk budget and cap |
| | `max_daily_loss_pct` | 2.5 | Kill switch |
| AI | `ai_every_bars` | 15 | Analyst cadence (also on every gate decision) |

**Strategy thresholds** (inside `workflow_strategy_engine`):

| Threshold | Value | Controls |
|---|---|---|
| S1/S5 LIS body | > 2×EM | LIS strength to confirm the bounce/rejection |
| S2 retracement tolerance | 1.5×EM | How close the low must be to a prior LIS high |
| S2 cooldown | 10 min | Between pyramids |
| S3 wick ratio | > 40% | Upper wick as a fraction of range |
| S3 ITM PE threshold | < −2.0 | ITM put selling intensity |
| S4 Fib tolerance | 2×EM | Distance from Fib 50% |
| S4 LIS recency | 4 min | How recent the UP LIS must be |

---

## 18. Reading the CLI Output

```
─────────────────────────────────────────────────────────────────
  ⏱️   [09:32 IST]  BAR 09:31 CLOSED  (bar #17/375)
─────────────────────────────────────────────────────────────────
  ⏱️   PROCESS MINUTE #17  [Triggered @ 09:32 | Bar 09:31]
  🧠  LAYER 1 · MARKET MEMORY  [...]
  📊  LAYER 2 · REGIME  [...] Regime=UNDEFINED
  🎯  LAYER X · CHECK TRIGGER  [...]
  📋  WORKFLOW STRATEGIES  [...]
  🎯  LAYER 3 · TRAP TRIGGER  [...]  SCORE: 12/100  AVOID
  💼  TRADE LIFECYCLE  [...]
  🤖  AI ANALYST  [for bar 09:30]  WAIT  conf=0.55  ...
  ✅  Bar#17  [...]  Score=12  InTrade=✘
```

At the end, the session summary prints the trade log, gated setups, workflow strategy signals, AI opinions, equity and drawdown.

---

## 19. Limitations

- **Single-day validation** (20 Feb 2026, a moderate-volatility session); no statistical edge is claimed yet.
- **Delta-1 P&L proxy:** option premium, theta, IV, slippage, spreads and broker latency are not modelled.
- **Historical replay only:** no live feed or order routing yet.
- **Equal-weight, single-instrument risk model;** no portfolio-level optimisation.
- **AI analyst is advisory** and depends on a third-party API (rate limits, latency).

---

## 20. Roadmap

1. **Multi-day backtesting:** 6–12 months of NIFTY data with walk-forward optimisation; track precision/recall of trap calls and calibrate the 70/80 thresholds.
2. **Paper and live trading:** WebSocket data feeds; an order-management module for Zerodha Kite Connect / Upstox (bracket orders, live positions, OAuth2, no stored credentials).
3. **Latency:** per-bar asynchronous inference; local quantised models (e.g. via vLLM / Ollama) to remove API limits.
4. **AI validation:** fine-tune on NIFTY price action, options anomalies and liquidity sweeps; map AI confidence against realised outcomes.
5. **Portfolio risk:** dynamic position sizing, drawdown controls, multi-instrument portfolio layer.
6. **Option-premium P&L:** model actual option prices, theta and IV.

---

## 21. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: langgraph` | `venv\Scripts\python -m pip install -r requirements.txt` |
| `KeyError: '__start__'` | LangGraph too old; use the versions in `requirements.txt` |
| `FileNotFoundError` on CSV | Ensure the data files are in `data/` |
| Browser blank at 8767 | Wait 2–3 s after startup, then refresh (needs `langgraph-viz`) |
| AI analyst shows "off" | Set `GEMINI_API_KEY` (env or `.env`), log in to Claude Code, or set `ANTHROPIC_API_KEY` |
| S1 never fires | Verify PDL is actually breached; check `wf_pdl_break_bar` in logs |
| S3 never fires | Check `wf_wick_rejections`; price may not revisit the intraday high after 12:30 |
| S4 Fib not computed | `wf_morning_peak_h` must exceed LIS low + 5×EM; check `wf_fib_computed` |
| Windows encoding error | Handled; all emoji/unicode output goes through the log tee |

---

## 22. Report / Slides vs Code

The code is the source of truth for the results above. Known differences from the project report and slides:

| Item | Report / slides | Code |
|---|---|---|
| Pipeline size | 6 nodes (report), 8 nodes (slides) | **13 nodes** |
| ATR | 14-period EMA | 14-period simple mean |
| TWAP | Mean of closes | Mean of (H + L + C) / 3 |
| LIS candle | Body > 2.5×ATR and ≥ 85% body | Body > EM (> 2×EM for S1/S5) |
| Regime | |Close − TWAP| > 1.5×ATR | 6-factor trend/mean point system |
| Target | 3×EM (1:3) | 2×R |
| Volume clue | 2.2× (slides); 1.5× vs 20-bar avg (report) | 1.5×, 5-bar avg vs prior 20 bars |
| AI workflow | Post-session 25-bar chunks to Groq (Llama 3.1 8B) | Async during replay, every 15 bars + each gate decision; Gemini / Claude Code / Anthropic |
| Latency | "Sub-millisecond" (slides) | ~10–30 ms per bar |

---

## 23. References

1. LangChain Inc., *LangGraph: Building Stateful, Multi-Actor Applications with LLMs*, 2024.
2. Cont, Kukanov & Stoikov, *The Price Impact of Order Book Events*, Journal of Financial Econometrics, 2014.
3. López de Prado, *Advances in Financial Machine Learning*, 2018.
4. Black & Scholes, *The Pricing of Options and Corporate Liabilities*, Journal of Political Economy, 1973.
5. Li et al., *FinAgent: A Multimodal Foundation Agent for Financial Trading*, arXiv:2402.18485, 2024.
6. Lopez-Lira & Tang, *Can ChatGPT Forecast Stock Price Movements?*, arXiv:2304.07619, 2023.
7. SEBI, reports on algorithmic trading and F&O participant outcomes.
8. NSE, *Market Statistics and Derivatives Data Handbook*, 2025.

---

*For research and education only. Not investment advice. No live trading is performed by this code.*
