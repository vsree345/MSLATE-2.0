# TRAP-X — Asymmetric Intraday Trap Detection & Execution Engine

## What This Project Is

TRAP-X is a **historical-replay trading strategy engine** for NIFTY intraday data.
It reads real 1-minute OHLC + options flow data from 20-Feb-2026 and runs four
rule-based strategies through a structured pipeline, printing every decision so
you can understand exactly *why* each trade signal fired.

The engine is built on **LangGraph** — every 1-minute bar is a separate graph
invocation that flows through eight analysis nodes in sequence. State (positions,
LIS legs, regime, etc.) accumulates across bars via LangGraph's `MemorySaver`
checkpoint. A live browser visualization at **http://localhost:8767** shows the
graph topology while the replay runs.

---

## Project Layout

```
mslate2.0/
├── TrapX_agent.py                ← Main engine (run this)
├── WorkFlow Enhancements.xlsx    ← Strategy spec (reference only)
├── data/
│   ├── spot_fut_2026-02-20.csv   ← 1-min OHLC for NIFTY SPOT + FUTURE
│   ├── signals_2026-02-20.csv    ← Per-minute options-flow signal rows
│   └── pdc_2026-02-20.csv        ← Previous-day context (PDH/PDL/PDC, VIX)
└── logs/                          ← Auto-created; timestamped log files
```

---

## Prerequisites

```bash
pip install langgraph langgraph-viz pandas numpy openpyxl
```

Python 3.10+ required.

---

## How to Run

```bash
cd c:\Users\sreev\Documents\mslate2.0
python TrapX_agent.py
```

The engine will:
1. Load all three CSV files from `data/`
2. Start a LangGraph-Viz server at http://localhost:8767
3. Replay all 375 bars (09:15 to 15:29), printing every node's output
4. Print a full session summary at the end
5. Save all console output to `logs/trapx_YYYYMMDD_HHMMSS.log`
6. Keep the viz server alive until you press **Ctrl-C**

---

## LangGraph Pipeline (9 Nodes)

Every 1-minute bar flows through these nodes in order:

```
START
  |
  +-- bar_index == 0 --> process_mkt_open     (PDC load + gap analysis, 09:15 only)
  |                           |
  +-- bar_index  > 0 ---------+
                              |
                       process_minute         (logs OHLC for this bar)
                              |
                   market_memory_engine       (TWAP, ATR, LIS legs, sweeps, impulse)
                              |
                    regime_detector           (TREND / MEAN / UNDEFINED)
                              |
                    check_trigger             (PDL/PDH/PDC breaks, new LIS/vol nodes)
                              |
              workflow_strategy_engine        (4 Excel strategies -- see below)
                              |
               trap_trigger_engine            (conviction scoring 0-100, trap structure)
                              |
             trade_lifecycle_manager          (entry / stop / time-exit)
                              |
                   minute_summary             (one-line bar summary)
                              |
                             END
```

---

## Data Files

### `spot_fut_2026-02-20.csv`
| Column | Description |
|--------|-------------|
| `timestamp` | e.g. `20-02-2026 09:15` |
| `instrument_type` | `SPOT` or `FUTURE` |
| `open / high / low / close` | Price in INR |
| `volume` | Cumulative futures volume |
| `oi` | Open interest |

### `signals_2026-02-20.csv`
Key columns used by the workflow engine:

| Column | Meaning |
|--------|---------|
| `final_signal_value` | Net options flow signal (positive = bullish, negative = bearish) |
| `jerk` | % rate of change of trending OI |
| `otm_put_sentiments_sum` | OTM PE OI aggregate (positive = PE writers adding = bullish divergence) |
| `otm_call_sentiments_sum` | OTM CE OI aggregate |
| `itm_put_sentiments_sum` | ITM PE OI (negative = ITM PE burning / PE squeeze) |
| `itm_call_sentiments_sum` | ITM CE OI |
| `otm_support` | Composite OTM structural support flag (> 0 = supportive) |
| `reversal_warning` | Reversal flag from options flow model |

### `pdc_2026-02-20.csv`
Three rows: **INDIA VIX** (VIX close for expected-move calculation),
**NIFTY 50** (PDH / PDL / PDC), **NIFTY FUTURE** (futures PDH / PDL / PDC).

---

## The 1-Minute Expected Move

Every strategy uses `expected_move_1min` as its unit of measure.
It is computed from VIX at market open:

```
expected_move_1min = PDH x (VIX / 100) x sqrt(1 / 94500) x 1.5
```

where 94500 = 375 trading minutes/day x 252 trading days.
For 20-Feb-2026 (VIX = 13.46, PDH = 25885), this is approximately **11.4 pts**.

---

## The Four Workflow Strategies

All four strategies are implemented in `workflow_strategy_engine` and maintain
their own position state (`wf_position`: IDLE | CALL | PUT), separate from
the main trap-engine.

### S1 — Bounce PDL Break -> Initial CALL

**When:** Morning session, after PDL is broken.

**Conditions:**
1. SPOT low breaks below Previous Day Low (PDL).
2. On the same or next bar, an UP LIS candle forms with body > 2x expected move.
   This confirms institutional absorption of the breakdown -- the trap is set.
3. The bar immediately after the LIS holds its low above the LIS open (support held).

**Action:** Enter Initial CALL. Stop = LIS low minus 1x expected move.

**Signal filter:** If the options signal is net bearish at entry, the signal is
*ignored* when any of these is true:
- `otm_put_sentiments_sum > 0` (OTM put writers building = bullish divergence+1)
- CE squeeze (ITM CE < 0 and OTM CE > 0)
- `otm_support > 0`

---

### S2 — LIS High Retracement -> Pyramid CALL

**When:** Active only when already in a CALL from S1.

**Conditions:**
1. Track all UP LIS candles that formed after the S1 entry.
2. Current bar's LOW touches within 1.5x expected move of any prior UP-LIS candle's HIGH.
   This means price retraced to a confirmed support level.

**Action:** Add to the existing CALL (pyramid).

**Filters:**
- Same signal-ignore logic as S1.
- Cooldown: at least 10 minutes since last pyramid.
- Only post-entry LIS legs counted (avoids pre-crash reference levels).

---

### S3 — Top Trade Less SL -> Initial PUT  (after 12:30 only)

**When:** Afternoon session, time >= 12:30.

**Conditions:**
1. Accumulate upper-wick rejections near the intraday high:
   - Wick > 40% of candle range AND price within 1x expected move of day high.
   - Need 2+ such rejections.
2. Options signal turns bearish (`final_signal_value < 0`) AND
   strong ITM PE selling (`itm_put_sentiments_sum < -2.0`).

**Action:** Close any open CALL, enter Initial PUT.
Stop = intraday high + 0.5x expected move (tight stop above rejected high).

---

### S4 — Fibonacci Retracement Bottom -> Initial CALL  (after 12:30 only)

**When:** Afternoon session, time >= 12:30.

**Conditions:**
1. Fibonacci 50% level is computed once (locked after first calculation):
   - Swing low = LIS-confirmed bottom's candle low (from S1 setup)
   - Swing high = morning peak (max high from LIS bottom to 12:30)
2. Current bar's LOW falls within 2x expected move of the Fib 50%.
3. Recent UP LIS in the last 4 minutes.
4. CE squeeze (`itm_call < 0 and otm_call > 0`) OR `otm_support > 0`.

**Action:** Close any open PUT, enter Initial CALL.
Stop = Fib 50% minus 1x expected move.

**Why Fib 50%:** After a morning impulse, the 50% retracement is a classic
institutional re-entry zone. The options squeeze confirmation reduces false entries.

---

## Trap-Engine Conviction Scoring (Layer 3)

The `trap_trigger_engine` node scores every bar independently, 0-100:

| Dimension | Max Pts | Signal |
|-----------|---------|--------|
| Futures Displacement | 30 | Futures close moved >= 0.12% vs prev bar |
| Option Ladder | 25 | CE and PE OI both aligned at 3+ delta levels |
| Volume Expansion | 20 | Futures volume > 1.5x rolling average |
| Delta Acceleration | 10 | Weighted CE delta change > threshold |
| Trap Structure | 10 | Strong-body candle + gap + fill pattern |
| Regime Alignment | 5 | Current regime consistent with trap direction |

Score >= 70 = MODERATE signal. Score >= 80 = HIGH signal. Below 70 = AVOID.

---

## Reading the Output

Each bar prints all 8 nodes in sequence:

```
─────────────────────────────────────────────────────────────────
  ⏱️   [09:32 IST]  BAR 09:31 CLOSED  (bar #17/375)
─────────────────────────────────────────────────────────────────

  ⏱️   PROCESS MINUTE #17  [Triggered @ 09:32 | Bar 09:31]

  🧠  LAYER 1 · MARKET MEMORY  [...]
      📊 SPOT LIS Leg @ 09:30  [+40.80 pts | UP]  (stack depth: 1)

  📊  LAYER 2 · REGIME  [...] Regime=UNDEFINED

  🎯  LAYER X · CHECK TRIGGER  [...]
      TRG 🔔  NEW SPOT LIS Leg @ 09:30  body=+40.80 pts

  📋  WORKFLOW STRATEGIES  [...]
      WF-LIS: LIS BOTTOM confirmed @ 09:30  body=+40.80 pts  (> 2x EM=11.40)
      WF S1 >> INITIAL CALL @ 25402.85  Stop=25370.75  [neg sig IGNORED - OTM PE / CE squeeze]

  🎯  LAYER 3 · TRAP TRIGGER  [...]
      CONVICTION CHECKLIST ...  SCORE: 12/100  AVOID

  💼  TRADE LIFECYCLE  [...]

  ✅  Bar#17  [...]  Score=12  InTrade=✘  Trades=0
```

**bar_time** = the closed candle being analysed.
**Triggered @** = the live-clock minute when analysis runs (always bar + 1 min).

---

## Session Summary

At the end of replay, the engine prints two sections:

**1. Trap-Engine Trade Log** -- trades triggered by the conviction scorer (score >= 70).

**2. Workflow Strategy Signals** -- all four strategy events with bar, price,
stop, Fib level, and whether a bearish signal was overridden.

Example:
```
── Workflow Strategy Signals (4 events) ──────────────────────
  Bar 09:31  S1 Initial CALL    @25402.85  SL=25370.75  [neg-sig IGNORED]
  Bar 10:10  S2 Pyramid CALL    @25485.40  ref=09:42
  Bar 12:44  S3 Initial PUT     @25606.95  SL=25668.60
  Bar 13:13  S4 Initial CALL    @25555.50  SL=25508.37  Fib50=25519.43

Workflow final position : CALL
Workflow last entry     : S4_Fibo_Retracement_Bottom  @ 25555.50
```

---

## Tuning Parameters

### Trap-Engine (`CFG` dict in `TrapX_agent.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `trap_body_pct` | 0.90 | Min body-to-range for a trap candle |
| `conviction_threshold` | 70 | Min score to generate a trap trade signal |
| `futures_displacement_pct` | 0.12 | Min futures move % for displacement score |
| `time_stop_candles` | 3 | Exit unfavorable trades after N bars |

### Workflow Strategies (inside `workflow_strategy_engine`)

| Threshold | Value | Controls |
|-----------|-------|---------|
| S1 LIS body | > 2x em | Strength to confirm LIS bottom |
| S2 retracement tol | 1.5x em | How close LOW must be to prior LIS HIGH |
| S3 wick ratio | > 40% | Upper wick as fraction of range |
| S3 ITM PE squeeze | < -2.0 | ITM put selling intensity threshold |
| S4 Fib tolerance | 2x em | Distance from Fib 50% to trigger |
| S4 LIS recency | 4 min | How recent the UP LIS must be |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: langgraph` | `pip install langgraph langgraph-viz` |
| `FileNotFoundError` on CSV | Ensure all three files are in `data/` subfolder |
| Browser blank at 8767 | Wait 2-3 s after startup then refresh |
| S1 never fires | Verify PDL is actually breached in spot data; check `wf_pdl_break_bar` in logs |
| S3 never fires | Check `wf_wick_rejections` count in logs; price may not reach intraday high near 12:30 |
| S4 fib not computed | Check `wf_fib_computed` flag; `wf_morning_peak_h` must be > LIS low + 5x EM |
| Windows encoding error | Already handled; all emoji/unicode output goes through the log tee |

---

## Architecture Notes

- **No live data** -- purely historical replay from CSV files.
- **Module-level globals** (`G_SPOT`, `G_FUT`, `G_SIG`, `G_PDC`) accumulate
  data outside LangGraph state to keep serialization fast. State holds only
  lightweight scalars and small lists.
- **MemorySaver** checkpoints every bar, so each invocation sees all prior
  accumulated context.
- **LangGraph-Viz** wraps the compiled graph and serves a topology browser at
  port 8767. The `with visualize(...) as viz_app` block starts and stops it.
- All console output is mirrored to a timestamped log file in `logs/`.
