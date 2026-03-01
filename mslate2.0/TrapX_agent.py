"""TRAP-X: Asymmetric Intraday Trap Detection & Execution Engine.

Single-graph design – one URL (http://localhost:8767):

  process_mkt_open → process_minute → market_memory_engine
  → regime_detector → check_trigger → workflow_strategy_engine
  → trap_trigger_engine → trade_lifecycle_manager → minute_summary → END

Timing model (mirrors live-market semantics):
  • process_mkt_open is the graph entry on EVERY invocation.
      – bar_index == 0 (09:15 IST, opening bell):
          loads PDC (prev_day_high, prev_day_low, prev_day_close)
          runs gap analysis vs 09:15 opening price
      – bar_index  > 0 : instant no-op
  • process_minute runs on every bar and logs the current bar.
  • At live clock T+1 we process the bar that CLOSED at T.
      e.g. at 09:16 IST → analyse the 09:15 closed candle.
  • Every print/label shows:  bar_time | trigger_time

PDC fields exposed in G_PDC global:
  prev_day_close, prev_day_high, prev_day_low
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import time as dtime, datetime as dt, timedelta
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

SPOT_FUT_CSV = os.path.join(_DATA, "spot_fut_2026-02-20.csv")
SIGNALS_CSV  = os.path.join(_DATA, "signals_2026-02-20.csv")
PDC_CSV      = os.path.join(_DATA, "pdc_2026-02-20.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Replay starts from this closed bar (processed at 09:16 IST)
    first_bar_time              = dtime(9, 15),
    session_end                 = dtime(15, 30),

    volume_node_multiplier      = 2.2,
    volume_node_strength_init   = 1.0,
    volume_node_decay           = 0.97,
    volume_node_expire          = 0.30,

    impulse_min_pct             = 0.35,
    impulse_min_duration        = 5,
    impulse_invalidation_retrace= 0.618,

    sweep_expiry_minutes        = 90,
    vwap_stretch_atr_mult       = 0.6,

    trend_day_range_pct         = 0.9,
    trend_day_vwap_unclaimed    = 45,
    mean_day_overlap_pct        = 0.65,
    mean_day_vwap_crossings     = 3,

    trap_body_pct               = 0.90,
    futures_displacement_pct    = 0.12,
    futures_volume_mult         = 1.5,
    conviction_threshold        = 70,
    option_ladder_pass          = 3,

    weight_futures_displacement = 30,
    weight_option_ladder        = 25,
    weight_volume_expansion     = 20,
    weight_delta_acceleration   = 10,
    weight_trap_structure       = 10,
    weight_regime_alignment     = 5,

    time_stop_candles           = 3,
    atr_lookback                = 14,
    overlap_lookback            = 30,

    vol_threshold               = 125000,
    vix_1m_threshold            = 1.5,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level accumulators (NOT in LangGraph state → fast serialisation)
# ─────────────────────────────────────────────────────────────────────────────
G_SPOT: List[Dict] = []   # SPOT bars accumulated oldest → newest
G_FUT:  List[Dict] = []   # FUTURE bars
G_SIG:  List[Dict] = []   # Option signal rows
G_PDC:  Dict       = {}  # Previous-day context (set once)

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State  (lightweight scalars + small lists)
# ─────────────────────────────────────────────────────────────────────────────
class TrapXState(TypedDict):
    # ── Timing ──────────────────────────────────────────────────────────────
    bar_index:          int    # 0-based position in replay
    bar_time:           str    # "HH:MM" – timestamp of the CLOSED bar
    trigger_time:       str    # "HH:MM" – live clock time (bar_time + 1 min)
    pdc_loaded:         bool

    # ── Layer 1 outputs ──────────────────────────────────────────────────────
    running_twap:       float
    running_atr:        float
    intraday_high:      float
    intraday_low:       float
    volume_nodes:       List[Dict]
    vol_5m_nodes:       List[Dict]
    impulse_legs:       List[Dict]
    liquidity_sweeps:   List[Dict]
    big_lis_legs:       List[Dict]
    fut_lis_legs:       List[Dict]
    vwap_stretches:     List[Dict]
    vwap_crossings:     int
    minutes_above_twap: int
    minutes_below_twap: int

    trig_pdc_broken:  bool
    trig_pdh_broken:  bool
    trig_pdl_broken:  bool
    triggers_list:       List[Dict]

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    regime:             str
    regime_confidence:  float

    # ── Layer 3 ──────────────────────────────────────────────────────────────
    trap_detected:      bool
    trap_direction:     str
    conviction_score:   int
    conviction_detail:  Dict
    trade_signal:       Optional[Dict]

    # ── Trade lifecycle ───────────────────────────────────────────────────────
    trade_state:        str
    trade_entry:        Optional[Dict]
    candles_in_trade:   int
    trade_log:          List[Dict]

    # ── Diagnostics ───────────────────────────────────────────────────────────
    processed_bars:     int
    expected_move_1min: float

    # ── Workflow Strategy State (4 Excel strategies) ─────────────────────────
    wf_pdl_break_bar:    str            # "HH:MM" when PDL was first broken
    wf_lis_bottom:       Optional[Dict] # first UP LIS candle after PDL break
    wf_morning_peak_h:   float          # max high from LIS bottom to 12:30
    wf_morning_peak_bar: str            # bar_time of the morning peak
    wf_fib_50:           float          # Fibonacci 50% level
    wf_fib_computed:     bool           # True once fib is locked in
    wf_position:         str            # "IDLE" | "CALL" | "PUT"
    wf_entry:            Optional[Dict] # current workflow trade entry info
    wf_pyramid_done:     bool           # S2 pyramid applied in this trade
    wf_last_pyramid_bar: str            # bar_time of last pyramid (cooldown)
    wf_wick_rejections:  int            # cumulative upper-wick rejections (S3)
    wf_last_wick_bar:    str            # bar_time of most recent wick rejection
    wf_trade_log:        List[Dict]     # workflow-specific signal & trade log


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _body_pct(c: Dict) -> float:
    rng = c["high"] - c["low"]
    return abs(c["close"] - c["open"]) / rng if rng > 1e-6 else 0.0

def _in_session(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return CFG["first_bar_time"] <= t <= CFG["session_end"]

def _twap(bars: List[Dict]) -> float:
    if not bars:
        return 0.0
    return sum((b["high"] + b["low"] + b["close"]) / 3 for b in bars) / len(bars)

def _atr(bars: List[Dict], period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(bars))):
        h, l, pc = bars[-i]["high"], bars[-i]["low"], bars[-i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else 0.0

def _rolling_vol_avg(fut_bars: List[Dict], lookback: int = 20) -> float:
    recent = fut_bars[-lookback:] if len(fut_bars) >= lookback else fut_bars
    return sum(b.get("volume", 0) for b in recent) / len(recent) if recent else 0.0

def _candle_overlap(bars: List[Dict], lookback: int = 30) -> float:
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    if len(recent) < 2:
        return 0.0
    ovs = []
    for i in range(1, len(recent)):
        plo = min(recent[i-1]["open"], recent[i-1]["close"])
        phi = max(recent[i-1]["open"], recent[i-1]["close"])
        clo = min(recent[i]["open"],   recent[i]["close"])
        chi = max(recent[i]["open"],   recent[i]["close"])
        ovs.append(max(0.0, min(phi, chi) - max(plo, clo)) / max(phi - plo, 1e-6))
    return sum(ovs) / len(ovs)

def _fmt(bar_time: str, trigger_time: str) -> str:
    """Label used in every node print: 'Triggered @ 09:16 | Bar 09:15'"""
    return f"Triggered @ {trigger_time} | Bar {bar_time}"


# ─────────────────────────────────────────────────────────────────────────────
# Workflow helpers  (used by workflow_strategy_engine)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_time(t: str) -> dtime:
    """Parse 'HH:MM' string to datetime.time (returns 00:00 on error)."""
    try:
        h, m = map(int, t.split(":"))
        return dtime(h, m)
    except Exception:
        return dtime(0, 0)


def _spot_bar_by_ts(bar_time: str) -> Optional[Dict]:
    """Return the G_SPOT bar whose timestamp matches 'HH:MM', or None."""
    for b in reversed(G_SPOT):
        ts = b.get("timestamp", "")
        try:
            if pd.Timestamp(ts).strftime("%H:%M") == bar_time:
                return b
        except Exception:
            pass
    return None


def _wf_sig_negative(sig: Dict) -> bool:
    """True when the options signal is net bearish."""
    return float(sig.get("final_signal_value", 0)) < 0


def _wf_ce_squeeze(sig: Dict) -> bool:
    """CE squeeze: ITM calls shedding (< 0) while OTM calls building (> 0).
    Indicates smart money is positioning bullish via OTM calls."""
    return (float(sig.get("itm_call_sentiments_sum", 0)) < 0 and
            float(sig.get("otm_call_sentiments_sum", 0)) > 0)


def _wf_ignore_neg_for_call(sig: Dict) -> bool:
    """Return True when a negative options signal should be IGNORED for a CALL
    trade — divergence+1 / CE squeeze / OTM support all override the bear signal.

    Conditions (any one suffices):
      • OTM PE positive  → put writers adding positions = smart-money bullish
      • CE squeeze       → institutional loading of OTM calls
      • otm_support > 0  → OTM structure supporting upside
    """
    otm_pe_positive = float(sig.get("otm_put_sentiments_sum", 0)) > 0
    ce_squeeze      = _wf_ce_squeeze(sig)
    otm_support     = float(sig.get("otm_support", 0)) > 0
    return otm_pe_positive or ce_squeeze or otm_support


# ─────────────────────────────────────────────────────────────────────────────
# NODE A – process_mkt_open
#   Runs ONCE at 09:15 IST (opening bell), routed here via conditional edge.
#   bar_index == 0 only – never called for subsequent bars.
#     • Loads PDC: prev_day_high, prev_day_low, prev_day_close
#     • Gap analysis: 09:15 open price vs PDC close
# ─────────────────────────────────────────────────────────────────────────────
def process_mkt_open(state: TrapXState) -> TrapXState:
    print("\n" + "═"*65)
    print("  🔔  PROCESS MARKET OPEN  [09:15 IST – Opening Bell]")
    print("═"*65)

    # ── Step 1: Load Previous Day Data ───────────────────────────────────────
    pdc_df  = pd.read_csv(PDC_CSV)
    nifty   = pdc_df[pdc_df["instrument_name"] == "NIFTY 50"].iloc[0].to_dict()
    vix_row = pdc_df[pdc_df["instrument_name"] == "INDIA VIX"].iloc[0].to_dict()
    fut_row = pdc_df[pdc_df["instrument_name"] == "NIFTY FUTURE"].iloc[0].to_dict()

    G_PDC["prev_day_high"]  = float(nifty["high"])
    G_PDC["prev_day_low"]   = float(nifty["low"])
    G_PDC["prev_day_close"] = float(nifty["close"])
    G_PDC["pdc_range"]      = G_PDC["prev_day_high"] - G_PDC["prev_day_low"]
    G_PDC["vix_close"]      = float(vix_row["close"])

    G_PDC["fut_prev_high"]  = float(fut_row["high"])
    G_PDC["fut_prev_low"]   = float(fut_row["low"])
    G_PDC["fut_prev_close"] = float(fut_row["close"])
    G_PDC["pdc_fut_range"]      = G_PDC["fut_prev_high"] - G_PDC["fut_prev_low"]

    pdh = G_PDC["prev_day_high"]
    pdl = G_PDC["prev_day_low"]
    pdc = G_PDC["prev_day_close"]

    fut_pdh = G_PDC["fut_prev_high"]
    fut_pdl = G_PDC["fut_prev_low"]
    fut_pdc = G_PDC["fut_prev_close"]    


	# Indian market constants
    MARKET_MINUTES_PER_DAY = 375  # 09:15 to 15:30 = 6.25 hours = 375 minutes
    TRADING_MINUTES_PER_YEAR = 94500  # 375 min/day * 250 trading days	

    
    # Use fixed market minutes for Indian markets (09:15 to 15:30 = 375 minutes)
    market_minutes = MARKET_MINUTES_PER_DAY

    # Expected move for the entire day
    # Formula: (price * (vix / 100)) * sqrt(market_minutes / 94500)
    sqrt_time_factor = float(np.sqrt(market_minutes / TRADING_MINUTES_PER_YEAR))

    # expected_move_day = pdh * (G_PDC['vix_close'] / 100) * sqrt_time_factor
    # G_PDC["expected_move_day"] = expected_move_day

    # Expected move for 1 minute  — cast to float so msgpack can serialise it
    sqrt_time_factor   = float(np.sqrt(1 / TRADING_MINUTES_PER_YEAR))
    expected_move_1min = float(pdh * (G_PDC['vix_close'] / 100) * sqrt_time_factor)

    # Z-score threshold (1.5 σ)
    zscore_threshold = 1.5 * expected_move_1min
    

    print(f"  Previous Day  PDH={pdh:.2f}  PDL={pdl:.2f}  PDC={pdc:.2f}")
    print(f"  Fut Prev Day  PDH={fut_pdh:.2f}  PDL={fut_pdl:.2f}  PDC={fut_pdc:.2f}")    
    print(f"  PDC Range     : {G_PDC['pdc_range']:.2f} pts")
    print(f"  FUT PDC Range     : {G_PDC['pdc_fut_range']:.2f} pts")    
    print(f"  India VIX     : {G_PDC['vix_close']:.4f}")
    print(f"  NF Fut Close  : {G_PDC['fut_prev_close']:.2f}")
    print(f"==>  📈 Expected 1-Min Move for the Day: {zscore_threshold:.2f} points ")    

    # ── Step 2: Gap Analysis (uses 09:15 opening price from CSV) ─────────────
    #   In live market at 09:15 we see the first tick / opening price.
    #   We use the 09:15 bar's OPEN from the CSV as a proxy.
    print()
    if G_SPOT:
        bar_0915   = G_SPOT[0]          # 09:15 bar (first row accumulated)
        open_price = bar_0915["open"]
        gap_pts    = open_price - pdc
        gap_pct    = gap_pts / pdc * 100 if pdc > 0 else 0.0
        gap_type   = ("GAP UP" if gap_pts > 5
                      else "GAP DOWN" if gap_pts < -5
                      else "FLAT OPEN")
        above_pdh  = open_price > pdh
        below_pdl  = open_price < pdl
        
        fut_bar_0915   = G_FUT[0]          # 09:15 bar (first row accumulated)
        fut_open_price = fut_bar_0915["open"]
        fut_gap_pts    = fut_open_price - fut_pdc
        fut_gap_pct    = fut_gap_pts / fut_pdc * 100 if fut_pdc > 0 else 0.0
        fut_gap_type   = ("GAP UP" if fut_gap_pts > 5
                      else "GAP DOWN" if fut_gap_pts < -5
                      else "FLAT OPEN")
        fut_above_pdh  = fut_open_price > fut_pdh
        fut_below_pdl  = fut_open_price < fut_pdl


        print(f"  ┌── OPENING GAP ANALYSIS ─────────────────────────────────────")
        print(f"  │  09:15 OPEN  : {open_price:.2f}")
        print(f"  │  PDH / PDL / PDC : {pdh:.2f} / {pdl:.2f} / {pdc:.2f}")
        print(f"  │  Gap         :  {gap_pts:+.2f} pts  ({gap_pct:+.2f}%)  → {gap_type}")
        print(f"  │  Above PDH   :  {above_pdh}")
        print(f"  │  Below PDL   :  {below_pdl}")
        if above_pdh:
            print(f"  │  ⚠️  STRONG BULLISH OPEN – price above prev-day high")
        elif below_pdl:
            print(f"  │  ⚠️  STRONG BEARISH OPEN – price below prev-day low")
        elif gap_pts > 5:
            print(f"  │  ↑  Moderate gap-up – watch for early filling")
        elif gap_pts < -5:
            print(f"  │  ↓  Moderate gap-down – watch for early filling")
        else:
            print(f"  │  ─  Near-flat open – no structural gap")
        print(f"  └─────────────────────────────────────────────────────────────\n")

        print(f"  ┌── OPENING FUT GAP ANALYSIS ──────────────────────────────────")
        print(f"  │  09:15 OPEN  : {fut_open_price:.2f}")
        print(f"  │  PDH / PDL / PDC : {fut_pdh:.2f} / {fut_pdl:.2f} / {fut_pdc:.2f}")
        print(f"  │  Gap         :  {fut_gap_pts:+.2f} pts  ({fut_gap_pct:+.2f}%)  → {fut_gap_type}")
        print(f"  │  Above PDH   :  {fut_above_pdh}")
        print(f"  │  Below PDL   :  {fut_below_pdl}")
        if fut_above_pdh:
            print(f"  │  ⚠️  STRONG BULLISH OPEN – price above prev-day high")
        elif fut_below_pdl:
            print(f"  │  ⚠️  STRONG BEARISH OPEN – price below prev-day low")
        elif fut_gap_pts > 5:
            print(f"  │  ↑  Moderate gap-up – watch for early filling")
        elif fut_gap_pts < -5:
            print(f"  │  ↓  Moderate gap-down – watch for early filling")
        else:
            print(f"  │  ─  Near-flat open – no structural gap")
        print(f"  └─────────────────────────────────────────────────────────────")

    else:
        print("  (09:15 bar not yet available – gap analysis skipped)")
    
    print("═"*65)

    # Initialise state fields that persist into bar_app
    state["pdc_loaded"]         = True
    state["intraday_high"]      = 0.0
    state["intraday_low"]       = 9_999_999.0
    state["vwap_crossings"]     = 0
    state["minutes_above_twap"] = 0
    state["minutes_below_twap"] = 0
    state["expected_move_1min"] = round(float(zscore_threshold), 2)
    state["trig_pdc_broken"]    = False
    state["trig_pdh_broken"]    = False

    # ── Workflow strategy state (fresh for each session) ─────────────────────
    state["wf_pdl_break_bar"]    = ""
    state["wf_lis_bottom"]       = None
    state["wf_morning_peak_h"]   = 0.0
    state["wf_morning_peak_bar"] = ""
    state["wf_fib_50"]           = 0.0
    state["wf_fib_computed"]     = False
    state["wf_position"]         = "IDLE"
    state["wf_entry"]            = None
    state["wf_pyramid_done"]     = False
    state["wf_last_pyramid_bar"] = ""
    state["wf_wick_rejections"]  = 0
    state["wf_last_wick_bar"]    = ""
    state["wf_trade_log"]        = []

    return state

# ─────────────────────────────────────────────────────────────────────────────
# NODE B – process_minute
#   Runs for EVERY closed bar from 09:15 bar onwards (triggered at 09:16+).
#   Timing note: at live clock T+1 we process the bar that CLOSED at T.
#   e.g. triggered at 09:16 → processes the 09:15 closed bar.
# ─────────────────────────────────────────────────────────────────────────────
def process_minute(state: TrapXState) -> TrapXState:
    bar_time     = state.get("bar_time",     "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    idx          = state.get("bar_index",    0)
    spot_close   = G_SPOT[-1]["close"] if G_SPOT else 0.0
    spot_open    = G_SPOT[-1]["open"] if G_SPOT else 0.0

    print(f"\n  ⏱️   PROCESS MINUTE #{idx + 1}  [{_fmt(bar_time, trigger_time)}]"
          f"\n\t  SPOT O={spot_open:.2f}"
          f"\n\t  SPOT C={spot_close:.2f}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE D – market_memory_engine  (Layer 1)  — shared by both apps
# ─────────────────────────────────────────────────────────────────────────────
def market_memory_engine(state: TrapXState) -> TrapXState:
    if not G_SPOT:
        return state

    bar_time     = state.get("bar_time", "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    cur          = G_SPOT[-1]
    close        = cur["close"]
    body         = cur["close"] - cur["open"]
    fut_cur          = G_FUT[-1]
    fut_close        = fut_cur["close"]
    idx          = state.get("bar_index",    0)
    fut_body         = fut_cur["close"] - fut_cur["open"]
    is_5minute_bar = int(trigger_time.split(":")[1]) % 5 == 0 if trigger_time != "??:??" else False

    print(f"\n  🧠  LAYER 1 · MARKET MEMORY  [{_fmt(bar_time, trigger_time)}]")

    twap = _twap(G_SPOT)
    atr  = _atr(G_SPOT, CFG["atr_lookback"])
    state["running_twap"] = round(twap, 2)
    state["running_atr"]  = round(atr,  2)
    print(f"     TWAP={twap:.2f}  ATR={atr:.2f}  Close={close:.2f}")

    state["intraday_high"] = max(state.get("intraday_high", 0.0), cur["high"])
    state["intraday_low"]  = min(state.get("intraday_low", 9_999_999.0), cur["low"])
    # if state["pdc_loaded"] and (idx > 1 ) and G_PDC:
    #     if state.get("trig_pdc_broken", False) == False and cur["low"] < G_PDC.get('prev_day_close',0):
    #         state["trig_pdc_broken"] = True
    #         print(f"     ⚠️  PDC BROKEN at {bar_time}")
    #     if state.get("trig_pdl_broken", False) == False and cur["low"] < G_PDC.get('prev_day_low',0):
    #         state["trig_pdl_broken"] = True
    #         print(f"     ⚠️  PDL BROKEN at {bar_time}")        
    #     if state.get("trig_pdh_broken", False) == False and cur["high"] > G_PDC.get('prev_day_high',0):
    #         state["trig_pdh_broken"] = True
    #         print(f"     ⚠️  PDH BROKEN at {bar_time}")

    # TWAP crossing
    if len(G_SPOT) >= 2:
        pc = G_SPOT[-2]["close"]
        if (pc < twap and close >= twap) or (pc > twap and close <= twap):
            state["vwap_crossings"] = state.get("vwap_crossings", 0) + 1
    state["minutes_above_twap"] = state.get("minutes_above_twap", 0) + (1 if close >= twap else 0)
    state["minutes_below_twap"] = state.get("minutes_below_twap", 0) + (0 if close >= twap else 1)

    # Volume Node
    volume_nodes = list(state.get("volume_nodes", []))    
    if G_FUT:
        fc      = G_FUT[-1]
        fut_vol = fc.get("volume", 0)
        avg_vol = _rolling_vol_avg(G_FUT)
        if avg_vol > 0 and fut_vol > CFG["volume_node_multiplier"] * avg_vol:
            volume_nodes.append(dict(price_level=close,
                                     timestamp_created=bar_time,
                                     strength=CFG["volume_node_strength_init"]))
            print(f"     🔴 Volume Node @ {close:.2f}  vol={fut_vol:,}")

        # Get the last and second-to-last volumes
        current_vol = G_FUT[-1].get("volume", 0)
        previous_vol = G_FUT[-2].get("volume", 0) if len(G_FUT) > 1 else 0

        # Calculate the difference
        fut_1m_vol = current_vol - previous_vol
        if fut_1m_vol > CFG["vol_threshold"] :
            volume_nodes.append(dict(price_level=close,
                                     timestamp_created=bar_time,
                                     strength=CFG["vol_threshold"]))
            print(f"     🔊 Volume Node @ {close:.2f}  vol={fut_vol:,}")            
    state["volume_nodes"] = [
        {**n, "strength": round(n["strength"] * CFG["volume_node_decay"], 4)}
        for n in volume_nodes
        if n["strength"] * CFG["volume_node_decay"] >= CFG["volume_node_expire"]
    ]

    # Volume Node
    vol_5m_nodes = list(state.get("vol_5m_nodes", []))    
    if G_FUT and is_5minute_bar:
        fc      = G_FUT[-1]
        fut_vol = fc.get("volume", 0)
        previous_5m_vol = G_FUT[-6].get("volume", 0) if len(G_FUT) > 5 else 0
        fut_5m_vol = current_vol - previous_5m_vol
        if fut_5m_vol >= CFG["vol_threshold"] :
            # Convert string to datetime object
            t = dt.strptime(trigger_time, "%H:%M")

            # Subtract 5 minutes
            prev_5m_time_obj = t - timedelta(minutes=5)
            
            # Format back to string (e.g., "09:25")
            bar_5m_time = prev_5m_time_obj.strftime("%H:%M")
            vol_5m_nodes.insert(0, dict(vol_5m=int(fut_5m_vol),
                                        ts=bar_5m_time,
                                        bar_ts=bar_time))   # bar_ts lets check_trigger detect "added this bar"
            print(f"     🔊 5-min Volume Node @ {bar_5m_time}  vol={fut_5m_vol}")
    state["vol_5m_nodes"] = vol_5m_nodes

    # ── LIS Leg Stacks (newest entry at index 0, oldest at end) ──────────────
    # insert(0, ...) keeps the stack ordered: stack[0] = most recent leg

    big_lis_legs = list(state.get("big_lis_legs", []))
    if abs(body) > state["expected_move_1min"]:
        big_lis_legs.insert(0, dict(
            body      = round(float(body), 2),
            direction = "UP" if body > 0 else "DOWN",
            ts        = bar_time,
        ))
        print(f"      📊 SPOT LIS Leg @ {bar_time}  [{body:+.2f} pts | {'UP' if body > 0 else 'DOWN'}]  "
              f"(stack depth: {len(big_lis_legs)})")
    state["big_lis_legs"] = big_lis_legs          # no trim – keep all

    fut_lis_legs = list(state.get("fut_lis_legs", []))
    if abs(fut_body) > state["expected_move_1min"]:
        fut_lis_legs.insert(0, dict(
            body      = round(float(fut_body), 2),
            direction = "UP" if fut_body > 0 else "DOWN",
            ts        = bar_time,
        ))
        print(f"      📈 FUT  LIS Leg @ {bar_time}  [{fut_body:+.2f} pts | {'UP' if fut_body > 0 else 'DOWN'}]  "
              f"(stack depth: {len(fut_lis_legs)})")
    state["fut_lis_legs"] = fut_lis_legs          # no trim – keep all

    # Impulse Leg
    dur          = CFG["impulse_min_duration"]
    impulse_legs = list(state.get("impulse_legs", []))
    if len(G_SPOT) >= dur + 1:
        window  = G_SPOT[-(dur + 1):]
        sp, ep  = window[0]["close"], window[-1]["close"]
        pct     = (ep - sp) / max(abs(sp), 1e-6)
        dirn    = "UP" if pct > 0 else "DOWN"
        highs   = [b["high"] for b in window]
        lows    = [b["low"]  for b in window]
        frng    = max(highs) - min(lows)
        pb      = (max(highs) - ep) / max(frng, 1e-6) if dirn == "UP" else (ep - min(lows)) / max(frng, 1e-6)
        if abs(pct) * 100 >= CFG["impulse_min_pct"] and pb < 0.50:
            if not impulse_legs or impulse_legs[-1].get("direction") != dirn:
                impulse_legs.append(dict(direction=dirn, start_price=sp, end_price=ep,
                                         slope=round((ep-sp)/dur, 2), active=True,
                                         timestamp=bar_time))
                print(f"     ⚡ Impulse Leg {dirn}  {sp:.1f}→{ep:.1f}")
    updated = []
    for leg in impulse_legs:
        leg = dict(leg)
        if leg.get("active"):
            total   = abs(leg["end_price"] - leg["start_price"])
            retrace = (leg["end_price"] - close) if leg["direction"] == "UP" else (close - leg["end_price"])
            if retrace > 0 and total > 0 and retrace / total >= CFG["impulse_invalidation_retrace"]:
                leg["active"] = False
                print(f"     ✗  Impulse Leg {leg['direction']} INVALIDATED")
        updated.append(leg)
    state["impulse_legs"] = updated

    # Liquidity Sweeps
    sweeps = list(state.get("liquidity_sweeps", []))
    ts_bar = pd.Timestamp(cur["timestamp"])
    if len(G_SPOT) >= 3:
        prev, curr = G_SPOT[-2], G_SPOT[-1]
        if curr["high"] > prev["high"] and curr["close"] < prev["high"]:
            sweeps.append(dict(sweep_level=prev["high"], direction="BEARISH",
                               timestamp=bar_time,
                               expiry_time=str(ts_bar + pd.Timedelta(minutes=CFG["sweep_expiry_minutes"]))))
            print(f"     🌊 Sweep BEARISH @ {prev['high']:.2f}")
        if curr["low"] < prev["low"] and curr["close"] > prev["low"]:
            sweeps.append(dict(sweep_level=prev["low"], direction="BULLISH",
                               timestamp=bar_time,
                               expiry_time=str(ts_bar + pd.Timedelta(minutes=CFG["sweep_expiry_minutes"]))))
            print(f"     🌊 Sweep BULLISH @ {prev['low']:.2f}")
    state["liquidity_sweeps"] = [s for s in sweeps if pd.Timestamp(s["expiry_time"]) > ts_bar]

    # TWAP Stretch
    stretches = list(state.get("vwap_stretches", []))
    if atr > 0 and abs(close - twap) > CFG["vwap_stretch_atr_mult"] * atr:
        dirn = "ABOVE" if close > twap else "BELOW"
        stretches.append(dict(direction=dirn, distance=round(abs(close - twap), 2),
                              timestamp=bar_time))
        stretches = stretches[-20:]
        print(f"     📏 TWAP Stretch {dirn}  dist={abs(close-twap):.2f}  ATR={atr:.2f}")
    state["vwap_stretches"] = stretches

    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE E – regime_detector  (Layer 2)
# ─────────────────────────────────────────────────────────────────────────────
def regime_detector(state: TrapXState) -> TrapXState:
    if len(G_SPOT) < 30:
        state["regime"] = "UNDEFINED"
        return state

    bar_time     = state.get("bar_time", "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    print(f"\n  📊  LAYER 2 · REGIME  [{_fmt(bar_time, trigger_time)}]")

    intraday_range = state["intraday_high"] - state["intraday_low"]
    pdc_range      = G_PDC.get("pdc_range", 496.0)
    range_ratio    = intraday_range / max(pdc_range, 1.0)
    max_unclaimed  = max(state.get("minutes_above_twap", 0), state.get("minutes_below_twap", 0))
    overlap        = _candle_overlap(G_SPOT, CFG["overlap_lookback"])
    active_legs    = [l for l in state.get("impulse_legs", []) if l.get("active")]

    last5  = G_FUT[-5:]  if len(G_FUT) >= 5 else G_FUT
    first5 = G_FUT[:5]   if len(G_FUT) >= 5 else G_FUT
    fut_vol_expanding = (sum(b.get("volume", 0) for b in last5) >
                         sum(b.get("volume", 0) for b in first5))

    xings = state.get("vwap_crossings", 0)
    trend_score = mean_score = 0
    if range_ratio > CFG["trend_day_range_pct"]:        trend_score += 40
    if max_unclaimed > CFG["trend_day_vwap_unclaimed"]: trend_score += 35
    if fut_vol_expanding:                               trend_score += 25
    if xings >= CFG["mean_day_vwap_crossings"]:         mean_score += 40
    if overlap > CFG["mean_day_overlap_pct"]:           mean_score += 35
    if not active_legs:                                 mean_score += 25

    if trend_score >= 60:
        regime, conf = "TREND", min(1.0, trend_score / 100)
    elif mean_score >= 60:
        regime, conf = "MEAN",  min(1.0, mean_score / 100)
    else:
        regime, conf = "UNDEFINED", 0.5

    state["regime"]            = regime
    state["regime_confidence"] = round(conf, 2)
    print(f"     Regime={regime}  Conf={conf:.0%}  "
          f"RangeRatio={range_ratio:.2f}  VWAPx={xings}  Overlap={overlap:.0%}")
    return state



# ─────────────────────────────────────────────────────────────────────────────
# NODE X – check_trigger  (Layer 2)
# ─────────────────────────────────────────────────────────────────────────────
def check_trigger(state: TrapXState) -> TrapXState:

    bar_time     = state.get("bar_time", "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    cur          = G_SPOT[-1]
    idx          = state.get("bar_index", 0)
    print(f"\n  🎯  LAYER X · CHECK TRIGGER  [{_fmt(bar_time, trigger_time)}]")

    # ── PDC / PDH / PDL level breaks ────────────────────────────────────────────
    if state["pdc_loaded"] and (idx > 1) and G_PDC:
        if not state.get("trig_pdc_broken", False) and cur["low"] < G_PDC.get("prev_day_close", 0):
            state["trig_pdc_broken"] = True
            print(f"  TRG 🔔   PDC BROKEN @ {bar_time}  "
                  f"(low={cur['low']:.2f} < PDC={G_PDC['prev_day_close']:.2f})")
        if not state.get("trig_pdl_broken", False) and cur["low"] < G_PDC.get("prev_day_low", 0):
            state["trig_pdl_broken"] = True
            print(f"  TRG 🔔   PDL BROKEN @ {bar_time}  "
                  f"(low={cur['low']:.2f} < PDL={G_PDC['prev_day_low']:.2f})")
        if not state.get("trig_pdh_broken", False) and cur["high"] > G_PDC.get("prev_day_high", 0):
            state["trig_pdh_broken"] = True
            print(f"  TRG 🔔   PDH BROKEN @ {bar_time}  "
                  f"(high={cur['high']:.2f} > PDH={G_PDC['prev_day_high']:.2f})")

    # ── Volume node detection (added this bar?) ─────────────────────────────────
    # 1-min volume node: timestamp_created == bar_time (set directly in market_memory_engine)
    vol_nodes   = state.get("volume_nodes", [])
    new_vol_1m  = any(n.get("timestamp_created") == bar_time for n in vol_nodes)
    if new_vol_1m:
        matched = next(n for n in vol_nodes if n.get("timestamp_created") == bar_time)
        print(f"  TRG 🔔  NEW 1-min Vol Node @ {bar_time}  "
              f"price={matched.get('price_level', 0):.2f}  "
              f"vol={matched.get('strength', 0):,}")

    # 5-min volume node: bar_ts == bar_time (stored alongside the node entry)
    vol_5m      = state.get("vol_5m_nodes", [])
    new_vol_5m  = bool(vol_5m) and vol_5m[0].get("bar_ts") == bar_time
    if new_vol_5m:
        node = vol_5m[0]
        print(f"  TRG 🔔  NEW 5-min Vol Node @ {bar_time}  "
              f"window_start={node.get('ts', '?')}  "
              f"vol={node.get('vol_5m', 0):,}")

    # ── LIS Leg detection (added this bar?) ──────────────────────────────────────
    # Both stacks use insert(0, ...) with ts=bar_time, so stack[0]["ts"] == bar_time
    # means a new leg was recorded this bar.

    big_lis = state.get("big_lis_legs", [])
    new_lis_spot = bool(big_lis) and big_lis[0].get("ts") == bar_time
    if new_lis_spot:
        leg = big_lis[0]
        print(f"  TRG 🔔  NEW SPOT LIS Leg @ {bar_time}  "
              f"body={leg.get('body', 0):+.2f} pts  "
              f"dir={leg.get('direction', '?')}  "
              f"(stack depth: {len(big_lis)})")

    fut_lis = state.get("fut_lis_legs", [])
    new_lis_fut = bool(fut_lis) and fut_lis[0].get("ts") == bar_time
    if new_lis_fut:
        leg = fut_lis[0]
        print(f"  TRG 🔔  NEW FUT  LIS Leg @ {bar_time}  "
              f"body={leg.get('body', 0):+.2f} pts  "
              f"dir={leg.get('direction', '?')}  "
              f"(stack depth: {len(fut_lis)})")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE F – trap_trigger_engine  (Layer 3)
# ─────────────────────────────────────────────────────────────────────────────
def trap_trigger_engine(state: TrapXState) -> TrapXState:
    if len(G_SPOT) < 3:
        state.update(trap_detected=False, trap_direction="", trade_signal=None)
        return state

    bar_time     = state.get("bar_time", "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    print(f"\n  🎯  LAYER 3 · TRAP TRIGGER  [{_fmt(bar_time, trigger_time)}]")

    prev = G_SPOT[-2]
    curr = G_SPOT[-1]

    body_pct    = _body_pct(prev)
    strong_body = body_pct >= CFG["trap_body_pct"]
    gap_up      = curr["open"] > prev["close"]
    gap_down    = curr["open"] < prev["close"]
    bull_fill   = gap_down and curr["high"] >= prev["close"]
    bear_fill   = gap_up   and curr["low"]  <= prev["close"]
    gap_fill    = bull_fill or bear_fill
    trap_struct = strong_body and (gap_up or gap_down) and gap_fill
    trap_dir    = ("UP" if bull_fill else "DOWN") if trap_struct else ""
    print(f"     Body={body_pct:.0%}  GapUp={gap_up}  GapDn={gap_down}  Fill={gap_fill}")

    atr         = state.get("running_atr", 0.0)
    active_legs = [l for l in state.get("impulse_legs", []) if l.get("active")]
    near_vnode  = any(abs(n["price_level"] - curr["close"]) < max(atr, 5.0)
                      for n in state.get("volume_nodes", []))
    against_imp = (any(l["direction"] == "UP"   and trap_dir == "DOWN" for l in active_legs) or
                   any(l["direction"] == "DOWN" and trap_dir == "UP"   for l in active_legs))
    ts_bar      = pd.Timestamp(curr["timestamp"])
    recent_sweep= any((pd.Timestamp(s["timestamp"] if len(s["timestamp"]) > 5
                                    else f"2026-02-20 {s['timestamp']}:00")
                       + pd.Timedelta(minutes=30)) > ts_bar
                      for s in state.get("liquidity_sweeps", []))
    context_ok  = trap_struct and (near_vnode or against_imp or recent_sweep)

    fut_disp_ok = fut_vol_ok = False
    if len(G_FUT) >= 2:
        fp, fc   = G_FUT[-2], G_FUT[-1]
        disp_pct = abs(fc["close"] - fp["close"]) / max(fp["close"], 1e-6) * 100
        fut_disp_ok = disp_pct >= CFG["futures_displacement_pct"]
        avg_fvol    = _rolling_vol_avg(G_FUT)
        fut_vol_ok  = avg_fvol > 0 and fc.get("volume", 0) > CFG["futures_volume_mult"] * avg_fvol
        print(f"     FutDisp={disp_pct:.3f}%  ok={fut_disp_ok}  "
              f"FutVol={fc.get('volume',0):,}  VolumeOk={fut_vol_ok}")

    opt_score, ladder_table = 0, []
    if len(G_SIG) >= 2:
        ls, ps = G_SIG[-1], G_SIG[-2]
        dmap   = {0.9: ("itm_call_sentiments_sum","itm_put_sentiments_sum"),
                  0.7: ("atm_call_sentiments_sum","atm_put_sentiments_sum"),
                  0.5: ("otm_call_sentiments_sum","otm_put_sentiments_sum"),
                  0.4: ("call_sentiments_sum",    "put_sentiments_sum")}
        cp = pp = 0
        for delta, (cc, pc) in dmap.items():
            ce_ok = (ls.get(cc,0) > ps.get(cc,0)) if trap_dir == "UP" else (ls.get(cc,0) < ps.get(cc,0))
            pe_ok = (ls.get(pc,0) < ps.get(pc,0)) if trap_dir == "UP" else (ls.get(pc,0) > ps.get(pc,0))
            cp += int(ce_ok); pp += int(pe_ok)
            ladder_table.append({"delta": delta, "CE": ce_ok, "PE": pe_ok})
        opt_score = 1 if cp >= CFG["option_ladder_pass"] and pp >= CFG["option_ladder_pass"] else 0

    delta_accel = 0.0
    if len(G_SIG) >= 2:
        cols = {0.9:"itm_call_sentiments_sum",0.7:"atm_call_sentiments_sum",
                0.5:"otm_call_sentiments_sum",0.4:"call_sentiments_sum"}
        for d, w in {0.9:0.4, 0.7:0.3, 0.5:0.2, 0.4:0.1}.items():
            pv, nv = G_SIG[-2].get(cols[d],0), G_SIG[-1].get(cols[d],0)
            if abs(pv) > 1e-6 and abs((nv-pv)/pv) > 0.015:
                delta_accel += w

    regime    = state.get("regime", "UNDEFINED")
    regime_ok = (regime == "MEAN" and trap_struct) or \
                (regime == "TREND" and any(l["direction"] == trap_dir for l in active_legs))

    score, detail = 0, {}
    def add(key, ok, weight, partial=1.0):
        nonlocal score
        pts = int(weight * partial) if ok else 0
        score += pts
        detail[key] = (bool(ok), weight)
    add("Futures Displacement", fut_disp_ok, CFG["weight_futures_displacement"])
    add("Option Ladder",        opt_score,   CFG["weight_option_ladder"])
    add("Volume Expansion",     fut_vol_ok,  CFG["weight_volume_expansion"])
    add("Delta Acceleration",   delta_accel >= 0.5,
        CFG["weight_delta_acceleration"],
        min(delta_accel / 0.5, 1.0) if delta_accel < 0.5 else 1.0)
    add("Trap Structure",       trap_struct, CFG["weight_trap_structure"])
    add("Regime Alignment",     regime_ok,   CFG["weight_regime_alignment"])
    score = min(score, 100)
    grade = "HIGH" if score >= 80 else "MODERATE" if score >= CFG["conviction_threshold"] else "AVOID"

    print(f"\n     ╔══  CONVICTION CHECKLIST @ bar {bar_time} ══╗")
    for label, (ok, wt) in detail.items():
        print(f"     {'✔' if ok else '✘'}  {label:<24}  [{wt:2d} pts]")
    print(f"     ╠══  SCORE: {score}/100  ·  {grade}")
    print(f"     ╚{'═'*40}╝")

    state.update(trap_detected=bool(trap_struct and context_ok and fut_disp_ok),
                 trap_direction=trap_dir,
                 conviction_score=score,
                 conviction_detail=detail)

    if state["trap_detected"] and score >= CFG["conviction_threshold"]:
        state["trade_signal"] = dict(
            bar_time=bar_time, trigger_time=trigger_time,
            direction=trap_dir, entry_price=curr["close"],
            stop_price=prev["high"] if trap_dir == "DOWN" else prev["low"],
            conviction=score, grade=grade,
        )
        print(f"\n  🚨  SIGNAL {trap_dir}  Entry={curr['close']:.2f}"
              f"  Stop={state['trade_signal']['stop_price']:.2f}  Grade={grade}")
    else:
        state["trade_signal"] = None
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE G – trade_lifecycle_manager
# ─────────────────────────────────────────────────────────────────────────────
def trade_lifecycle_manager(state: TrapXState) -> TrapXState:
    if not G_SPOT:
        return state
    bar_time     = state.get("bar_time", "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    tstate       = state.get("trade_state", "IDLE")
    tlog         = list(state.get("trade_log", []))
    print(f"\n  💼  TRADE LIFECYCLE  [{_fmt(bar_time, trigger_time)}]  state={tstate}")

    cur = G_SPOT[-1]

    if tstate == "IN_TRADE":
        entry     = state.get("trade_entry", {})
        n         = state.get("candles_in_trade", 0) + 1
        state["candles_in_trade"] = n
        direction = entry.get("direction", "UP")
        entry_p   = entry.get("entry_price", cur["close"])
        stop_p    = entry.get("stop_price",  cur["close"])

        if n >= CFG["time_stop_candles"]:
            favorable = ((direction == "UP" and cur["close"] > entry_p) or
                         (direction == "DOWN" and cur["close"] < entry_p))
            if not favorable:
                pnl = cur["close"] - entry_p if direction == "UP" else entry_p - cur["close"]
                tlog.append(dict(bar_time=bar_time, trigger_time=trigger_time,
                                 action="TIME_EXIT", price=cur["close"],
                                 pnl_pts=round(pnl, 2), reason="3-candle rule"))
                print(f"     ⏱️  TIME EXIT @ {cur['close']:.2f}  pnl={pnl:+.1f}  (3-candle rule)")
                state.update(trade_state="IDLE", trade_entry=None, candles_in_trade=0)
                state["trade_log"] = tlog
                return state

        stop_hit = ((direction == "DOWN" and cur["high"] >= stop_p) or
                    (direction == "UP"   and cur["low"]  <= stop_p))
        if stop_hit:
            pnl = cur["close"] - entry_p if direction == "UP" else entry_p - cur["close"]
            tlog.append(dict(bar_time=bar_time, trigger_time=trigger_time,
                             action="STRUCTURAL_EXIT", price=stop_p,
                             pnl_pts=round(pnl, 2), reason="Stop hit"))
            print(f"     🛑  STRUCTURAL EXIT @ {stop_p:.2f}  pnl={pnl:+.1f}")
            state.update(trade_state="IDLE", trade_entry=None, candles_in_trade=0)

    elif tstate == "IDLE":
        sig = state.get("trade_signal")
        if sig:
            state.update(trade_state="IN_TRADE", trade_entry=sig, candles_in_trade=0)
            tlog.append(dict(bar_time=bar_time, trigger_time=trigger_time,
                             action="ENTRY", direction=sig["direction"],
                             price=sig["entry_price"], stop=sig["stop_price"],
                             conviction=sig["conviction"], grade=sig["grade"]))
            print(f"     📌 ENTRY {sig['direction']} @ {sig['entry_price']:.2f}"
                  f"  Stop={sig['stop_price']:.2f}  Grade={sig['grade']}")

    state["trade_log"] = tlog
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE H – minute_summary
# ─────────────────────────────────────────────────────────────────────────────
def minute_summary(state: TrapXState) -> TrapXState:
    bar_time     = state.get("bar_time", "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    state["processed_bars"] = state.get("processed_bars", 0) + 1
    n        = state["processed_bars"]
    trades   = [e for e in state.get("trade_log", []) if e.get("action") == "ENTRY"]
    in_trade = state.get("trade_state", "IDLE") == "IN_TRADE"
    print(f"\n  ✅  Bar#{n}  [{_fmt(bar_time, trigger_time)}]  "
          f"Regime={state.get('regime','?')}  "
          f"TWAP={state.get('running_twap',0):.2f}  "
          f"Score={state.get('conviction_score',0)}  "
          f"InTrade={'✔' if in_trade else '✘'}  "
          f"Trades={len(trades)}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE WF – workflow_strategy_engine
#
#  Implements the 4 strategies from WorkFlow Enhancements.xlsx:
#    S1  Bounce PDL Break         → Initial CALL
#        • PDL broken → first UP LIS > 2× EM confirms bottom
#        • Next bar holds support above LIS open → CALL entry
#        • Ignore bearish signal when OTM PE positive / CE squeeze / otm_support
#
#    S2  LIS High Retracement     → Pyramid CALL (add to existing CALL)
#        • In CALL, price retraces and low touches a prior UP-LIS high
#        • Cooldown: ≥ 10 min since last pyramid; only post-entry LIS legs counted
#        • Ignore bearish signal (same filter as S1)
#
#    S3  Top Trade Less SL        → Initial PUT  (only after 12:30)
#        • 2+ upper-wick rejections near intraday high
#        • Negative signal + strong ITM PE selling (itm_put < -2)
#        • Closes any open CALL and enters PUT with tight SL above day high
#
#    S4  Fibo Retracement Bottom  → Initial CALL (only after 12:30)
#        • Fib 50% computed from LIS-bottom low to morning peak (before 12:30)
#        • Current bar low within 2× EM of fib 50%
#        • Recent UP LIS (≤ 4 bars ago) + CE squeeze OR otm_support > 0
#        • Closes any open PUT and enters CALL
# ─────────────────────────────────────────────────────────────────────────────
def workflow_strategy_engine(state: TrapXState) -> TrapXState:
    if not G_SPOT or not G_PDC:
        return state

    bar_time     = state.get("bar_time",     "??:??")
    trigger_time = state.get("trigger_time", "??:??")
    cur          = G_SPOT[-1]
    em           = float(state.get("expected_move_1min", 11.0))
    sig          = G_SIG[-1]           if G_SIG             else {}
    prev_sig     = G_SIG[-2]           if len(G_SIG) >= 2   else {}

    print(f"\n  📋  WORKFLOW STRATEGIES  [{_fmt(bar_time, trigger_time)}]")

    wf_pos        = state.get("wf_position",      "IDLE")
    wf_pdl_break  = state.get("wf_pdl_break_bar", "")
    wf_lis_bottom = state.get("wf_lis_bottom",    None)
    pdl           = G_PDC.get("prev_day_low", 0.0)
    bar_t         = _parse_time(bar_time)

    # ── Track PDL break (first bar whose low goes under prev-day low) ─────────
    if not wf_pdl_break and cur["low"] < pdl:
        wf_pdl_break              = bar_time
        state["wf_pdl_break_bar"] = bar_time
        print(f"     WF-PDL: PDL BROKEN @ {bar_time}  "
              f"low={cur['low']:.2f} < PDL={pdl:.2f}")

    # ── Track first UP LIS after PDL break (LIS body > 2× expected move) ─────
    if wf_pdl_break and not wf_lis_bottom:
        big_lis = state.get("big_lis_legs", [])
        if (big_lis
                and big_lis[0].get("ts") == bar_time
                and big_lis[0].get("direction") == "UP"
                and big_lis[0]["body"] > 2 * em):
            wf_lis_bottom = dict(
                ts    = bar_time,
                open  = cur["open"],
                high  = cur["high"],
                low   = cur["low"],
                close = cur["close"],
                body  = big_lis[0]["body"],
            )
            state["wf_lis_bottom"] = wf_lis_bottom
            print(f"     WF-LIS: LIS BOTTOM confirmed @ {bar_time}  "
                  f"body={big_lis[0]['body']:+.2f} pts  (> 2x EM={em:.2f})")

    # ── Track morning peak: max high from LIS bottom until 12:30 ─────────────
    if wf_lis_bottom and bar_t < dtime(12, 30):
        curr_peak = float(state.get("wf_morning_peak_h", 0.0))
        if cur["high"] > curr_peak:
            state["wf_morning_peak_h"]   = round(cur["high"], 2)
            state["wf_morning_peak_bar"] = bar_time

    # ═════════════════════════════════════════════════════════════════════════
    # S1 – Bounce PDL Break → Initial CALL
    # Entry bar = FIRST bar after LIS-bottom that holds support above LIS open
    # ═════════════════════════════════════════════════════════════════════════
    if (wf_lis_bottom is not None
            and wf_pos == "IDLE"
            and bar_time != wf_lis_bottom["ts"]):

        lis_open   = float(wf_lis_bottom["open"])
        support_ok = cur["low"] > lis_open * 0.999   # within 0.1% of LIS open

        if support_ok:
            sig_neg = _wf_sig_negative(sig)
            proceed = (not sig_neg) or _wf_ignore_neg_for_call(sig)

            if proceed:
                entry_p = round(cur["close"], 2)
                stop_p  = round(wf_lis_bottom["low"] - em, 2)
                state["wf_position"]     = "CALL"
                state["wf_pyramid_done"] = False
                state["wf_entry"] = dict(
                    strategy    = "S1_Bounce_PDL_Break",
                    direction   = "CALL",
                    entry_bar   = bar_time,
                    entry_price = entry_p,
                    stop_price  = stop_p,
                )
                wf_tlog = list(state.get("wf_trade_log", []))
                ignored = sig_neg and _wf_ignore_neg_for_call(sig)
                wf_tlog.append(dict(
                    bar_time     = bar_time,
                    trigger_time = trigger_time,
                    action       = "S1_INITIAL_CALL",
                    price        = entry_p,
                    stop         = stop_p,
                    sig_neg      = sig_neg,
                    ignored      = ignored,
                ))
                state["wf_trade_log"] = wf_tlog
                tag = " [neg sig IGNORED - OTM PE / CE squeeze]" if ignored else ""
                print(f"     WF S1 >> INITIAL CALL @ {entry_p:.2f}  "
                      f"Stop={stop_p:.2f}{tag}")
                wf_pos = "CALL"

    # ═════════════════════════════════════════════════════════════════════════
    # S2 – LIS High Retracement → Pyramid CALL
    # Price retraces and current bar's LOW touches a prior UP-LIS candle HIGH
    # ═════════════════════════════════════════════════════════════════════════
    if (wf_lis_bottom is not None
            and wf_pos == "CALL"
            and not state.get("wf_pyramid_done", False)):

        tol         = em * 1.5
        entry_bar   = state.get("wf_entry", {}).get("entry_bar", "")
        big_lis     = state.get("big_lis_legs", [])

        # Only UP LIS legs that formed AFTER initial CALL entry
        post_entry_up = [
            leg for leg in big_lis
            if leg.get("direction") == "UP" and leg.get("ts", "") > entry_bar
        ]

        near_lis_high = False
        ref_ts        = ""
        for leg in post_entry_up:
            spot_b = _spot_bar_by_ts(leg["ts"])
            if spot_b is not None:
                lis_h = float(spot_b["high"])
                if abs(cur["low"] - lis_h) <= tol:
                    near_lis_high = True
                    ref_ts        = leg["ts"]
                    break

        if near_lis_high:
            sig_neg = _wf_sig_negative(sig)
            proceed = (not sig_neg) or _wf_ignore_neg_for_call(sig)

            # Cooldown: at least 10 minutes since last pyramid
            last_pyr    = state.get("wf_last_pyramid_bar", "")
            cooldown_ok = True
            if last_pyr:
                try:
                    dt1 = dt.strptime(bar_time, "%H:%M")
                    dt2 = dt.strptime(last_pyr,  "%H:%M")
                    cooldown_ok = abs((dt1 - dt2).seconds // 60) >= 10
                except Exception:
                    pass

            if proceed and cooldown_ok:
                state["wf_pyramid_done"]     = True
                state["wf_last_pyramid_bar"] = bar_time
                wf_tlog = list(state.get("wf_trade_log", []))
                ignored = sig_neg and _wf_ignore_neg_for_call(sig)
                wf_tlog.append(dict(
                    bar_time     = bar_time,
                    trigger_time = trigger_time,
                    action       = "S2_PYRAMID_CALL",
                    price        = round(cur["close"], 2),
                    ref_lis_ts   = ref_ts,
                    sig_neg      = sig_neg,
                    ignored      = ignored,
                ))
                state["wf_trade_log"] = wf_tlog
                tag = " [neg sig IGNORED]" if ignored else ""
                print(f"     WF S2 >> PYRAMID CALL @ {cur['close']:.2f}  "
                      f"(prior LIS high @ {ref_ts}){tag}")

    # ═════════════════════════════════════════════════════════════════════════
    # S3 – Top Trade Less SL → Initial PUT   (time-gated: after 12:30)
    # Requires 2+ upper-wick rejections near intraday high, then neg signal
    # with strong ITM PE selling.
    # ═════════════════════════════════════════════════════════════════════════
    if wf_pos in ("IDLE", "CALL") and bar_t >= dtime(12, 30):

        # Accumulate wick rejections near the intraday high
        wick     = cur["high"] - max(cur["open"], cur["close"])
        rng      = cur["high"] - cur["low"]
        id_high  = float(state.get("intraday_high", 0.0))
        near_top = (id_high - cur["high"]) < em

        if rng > 0.5 and (wick / rng) > 0.40 and near_top:
            rej = state.get("wf_wick_rejections", 0) + 1
            state["wf_wick_rejections"] = rej
            state["wf_last_wick_bar"]   = bar_time
            print(f"     WF S3: Upper-wick rejection #{rej} @ {bar_time}  "
                  f"wick={wick:.2f} ({wick/rng:.0%} of range)")

        rej_count = state.get("wf_wick_rejections", 0)
        if rej_count >= 2:
            sig_neg   = _wf_sig_negative(sig)
            pe_squeeze = float(sig.get("itm_put_sentiments_sum", 0)) < -2.0

            if sig_neg and pe_squeeze:
                entry_p = round(cur["close"], 2)
                stop_p  = round(id_high + em * 0.5, 2)

                if wf_pos == "CALL":
                    print(f"     WF S3: Closing existing CALL -> pivoting to PUT")
                state["wf_position"]        = "PUT"
                state["wf_pyramid_done"]    = False
                state["wf_wick_rejections"] = 0   # reset for next S3 cycle
                state["wf_entry"] = dict(
                    strategy    = "S3_Top_Trade_Less_SL",
                    direction   = "PUT",
                    entry_bar   = bar_time,
                    entry_price = entry_p,
                    stop_price  = stop_p,
                )
                wf_tlog = list(state.get("wf_trade_log", []))
                wf_tlog.append(dict(
                    bar_time     = bar_time,
                    trigger_time = trigger_time,
                    action       = "S3_INITIAL_PUT",
                    price        = entry_p,
                    stop         = stop_p,
                    rej_count    = rej_count,
                ))
                state["wf_trade_log"] = wf_tlog
                print(f"     WF S3 >> INITIAL PUT @ {entry_p:.2f}  "
                      f"Stop={stop_p:.2f}  "
                      f"({rej_count} wick rejections + PE squeeze)")
                wf_pos = "PUT"

    # ═════════════════════════════════════════════════════════════════════════
    # S4 – Fibo Retracement Bottom → Initial CALL  (time-gated: after 12:30)
    # Fib 50% from LIS-bottom low to morning peak; current bar tests that level
    # with a recent UP LIS and CE-squeeze / OTM-support confirmation.
    # ═════════════════════════════════════════════════════════════════════════
    if (wf_lis_bottom is not None
            and wf_pos in ("IDLE", "PUT")
            and bar_t >= dtime(12, 30)):

        # Compute Fibonacci 50% level once (locked after first calculation)
        if not state.get("wf_fib_computed", False):
            swing_low  = float(wf_lis_bottom["low"])
            swing_high = float(state.get("wf_morning_peak_h", 0.0))
            if swing_high > swing_low + em * 5:
                fib_50 = swing_low + 0.5 * (swing_high - swing_low)
                state["wf_fib_50"]       = round(fib_50, 2)
                state["wf_fib_computed"] = True
                print(f"     WF S4: Fib levels  "
                      f"swing_low={swing_low:.2f}  "
                      f"swing_high={swing_high:.2f}  "
                      f"50%={fib_50:.2f}")

        fib_50   = float(state.get("wf_fib_50", 0.0))
        tol      = em * 2.0
        near_fib = fib_50 > 0 and abs(cur["low"] - fib_50) <= tol

        if near_fib:
            big_lis  = state.get("big_lis_legs", [])
            # UP LIS formed in the last 4 minutes
            cutoff   = (dt.strptime(bar_time, "%H:%M") - timedelta(minutes=4)).strftime("%H:%M")
            recent_lis_up = any(
                l.get("direction") == "UP" and l.get("ts", "") >= cutoff
                for l in big_lis
            )
            ce_squeeze = _wf_ce_squeeze(sig)
            otm_sup    = float(sig.get("otm_support", 0)) > 0

            if recent_lis_up and (ce_squeeze or otm_sup):
                entry_p = round(cur["close"], 2)
                stop_p  = round(fib_50 - em, 2)

                if wf_pos == "PUT":
                    print(f"     WF S4: Closing existing PUT -> pivoting to CALL")
                state["wf_position"]     = "CALL"
                state["wf_pyramid_done"] = False
                state["wf_entry"] = dict(
                    strategy    = "S4_Fibo_Retracement_Bottom",
                    direction   = "CALL",
                    entry_bar   = bar_time,
                    entry_price = entry_p,
                    stop_price  = stop_p,
                    fib_50      = fib_50,
                )
                wf_tlog = list(state.get("wf_trade_log", []))
                conf_reason = "CE-squeeze" if ce_squeeze else "OTM-PE-support"
                wf_tlog.append(dict(
                    bar_time     = bar_time,
                    trigger_time = trigger_time,
                    action       = "S4_INITIAL_CALL",
                    price        = entry_p,
                    stop         = stop_p,
                    fib_50       = fib_50,
                    dist_to_fib  = round(cur["low"] - fib_50, 2),
                    confirmation = conf_reason,
                ))
                state["wf_trade_log"] = wf_tlog
                print(f"     WF S4 >> INITIAL CALL @ {entry_p:.2f}  "
                      f"Stop={stop_p:.2f}  Fib50={fib_50:.2f}  "
                      f"[{conf_reason} + recent LIS UP]")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Single graph  –  one URL, one timeline
#
# Routing from START (conditional):
#   bar_index == 0  →  process_mkt_open → process_minute → … → END
#   bar_index  > 0  →  process_minute   → …              → END
#
# process_mkt_open therefore appears in the Event Timeline ONLY on the
# first bar (09:15 IST).  All subsequent bars skip it entirely.
# ─────────────────────────────────────────────────────────────────────────────
def create_trapx_app(memory: MemorySaver):
    """Single compiled graph, one URL (port 8767).

    Bar numbering (1-based, bar position in session):
      Bar 1  = 09:15 IST  →  process_mkt_open → process_minute #1 → analysis
                              (PDC + gap analysis, THEN full bar analysis)
      Bar 2  = 09:16 IST  →  process_minute #2 → analysis chain
      Bar 3  = 09:17 IST  →  process_minute #3 → analysis chain
      ...

    Routing from START (conditional on bar_index):
      bar_index == 0  →  process_mkt_open → process_minute → analysis → END
      bar_index  > 0  →  process_minute   → analysis                 → END
    """
    from langgraph.graph import START

    wf = StateGraph(TrapXState)

    wf.add_node("process_mkt_open",         process_mkt_open)
    wf.add_node("process_minute",           process_minute)
    wf.add_node("market_memory_engine",     market_memory_engine)
    wf.add_node("regime_detector",          regime_detector)
    wf.add_node("check_trigger",            check_trigger)
    wf.add_node("workflow_strategy_engine", workflow_strategy_engine)
    wf.add_node("trap_trigger_engine",      trap_trigger_engine)
    wf.add_node("trade_lifecycle_manager",  trade_lifecycle_manager)
    wf.add_node("minute_summary",           minute_summary)

    # Conditional entry from START
    wf.add_conditional_edges(
        START,
        lambda s: "process_mkt_open" if s.get("bar_index", 0) == 0 else "process_minute",
        {"process_mkt_open": "process_mkt_open", "process_minute": "process_minute"},
    )

    # Bar 1 (bar_index 0, 09:15): after opening context, run the full bar analysis too
    wf.add_edge("process_mkt_open",         "process_minute")

    # All bars: full analysis chain
    wf.add_edge("process_minute",           "market_memory_engine")
    wf.add_edge("market_memory_engine",     "regime_detector")
    wf.add_edge("regime_detector",          "check_trigger")
    # Workflow strategies run after structural trigger detection
    wf.add_edge("check_trigger",            "workflow_strategy_engine")
    wf.add_edge("workflow_strategy_engine", "trap_trigger_engine")

    wf.add_edge("trap_trigger_engine",      "trade_lifecycle_manager")
    wf.add_edge("trade_lifecycle_manager",  "minute_summary")
    wf.add_edge("minute_summary",           END)

    return wf.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────────────────────────
# Session summary
# ─────────────────────────────────────────────────────────────────────────────
def print_session_summary(st: Dict):
    tlog   = st.get("trade_log", [])
    trades = [e for e in tlog if e["action"] == "ENTRY"]
    exits  = [e for e in tlog if e["action"] in ("TIME_EXIT", "STRUCTURAL_EXIT")]
    pnls   = [e.get("pnl_pts", 0) for e in exits]

    print("\n" + "═"*65)
    print("  📋  SESSION SUMMARY  –  NIFTY 2026-02-20")
    print("═"*65)
    print(f"  Bars processed    : {st.get('processed_bars', 0)}")
    print(f"  Final Regime      : {st.get('regime','?')} (conf={st.get('regime_confidence',0):.0%})")
    print(f"  Intraday H/L      : {st.get('intraday_high',0):.2f} / {st.get('intraday_low',0):.2f}")
    print(f"  PDH/PDL/PDC       : {G_PDC.get('prev_day_high',0):.2f} / "
          f"{G_PDC.get('prev_day_low',0):.2f} / {G_PDC.get('prev_day_close',0):.2f}")
    print(f"  TWAP Crossings    : {st.get('vwap_crossings', 0)}")
    print(f"  Volume Nodes live : {len(st.get('volume_nodes',[]))}")
    print(f"\n  Trap-engine trades: {len(trades)}")
    if exits:
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        print(f"  Wins / Losses     : {len(wins)} / {len(losses)}")
        print(f"  Total P&L (pts)   : {sum(pnls):+.2f}")
    print("\n  ── Trap-Engine Trade Log ───────────────────────────────────")
    for e in tlog:
        dirn   = f" [{e['direction']}]" if "direction" in e else ""
        price  = e.get("price") or e.get("entry_price", 0)
        pnl    = f"  pnl={e['pnl_pts']:+.1f}" if "pnl_pts" in e else ""
        print(f"    Bar {e.get('bar_time','?')} "
              f"(triggered @ {e.get('trigger_time','?')})  "
              f"{e['action']:<20}{dirn}  @{price:.2f}{pnl}")

    # ── Workflow strategy signals ─────────────────────────────────────────────
    wf_tlog = st.get("wf_trade_log", [])
    print(f"\n  ── Workflow Strategy Signals ({len(wf_tlog)} events) ──────────────")
    _ACTION_LABEL = {
        "S1_INITIAL_CALL": "S1 Initial CALL   ",
        "S2_PYRAMID_CALL": "S2 Pyramid CALL   ",
        "S3_INITIAL_PUT":  "S3 Initial PUT    ",
        "S4_INITIAL_CALL": "S4 Initial CALL   ",
    }
    for e in wf_tlog:
        label = _ACTION_LABEL.get(e.get("action", ""), e.get("action", "?")[:18].ljust(18))
        price = e.get("price", 0)
        stop  = f"  SL={e['stop']:.2f}" if "stop" in e else ""
        fib   = f"  Fib50={e['fib_50']:.2f}" if "fib_50" in e else ""
        ign   = "  [neg-sig IGNORED]" if e.get("ignored") else ""
        ref   = f"  ref={e['ref_lis_ts']}" if "ref_lis_ts" in e else ""
        print(f"    Bar {e.get('bar_time','?'):5s}  {label}  @{price:.2f}{stop}{fib}{ref}{ign}")

    wf_pos = st.get("wf_position", "IDLE")
    wf_entry = st.get("wf_entry") or {}
    print(f"\n  Workflow final position : {wf_pos}")
    if wf_entry:
        print(f"  Workflow last entry     : {wf_entry.get('strategy','?')}"
              f"  @ {wf_entry.get('entry_price', 0):.2f}")
    print("═"*65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Tee: mirror all print() output to a timestamped log file ────────────
    class _Tee:
        """Writes to both the real stdout/stderr AND a log file."""
        def __init__(self, real_stream, log_file):
            self._real  = real_stream
            self._log   = log_file
        def write(self, data):
            self._real.write(data)
            self._log.write(data)
            self._log.flush()          # flush every write – so tail -f works
        def flush(self):
            self._real.flush()
            self._log.flush()
        def __getattr__(self, name):
            return getattr(self._real, name)

    _LOG_DIR  = os.path.join(_HERE, "logs")
    os.makedirs(_LOG_DIR, exist_ok=True)
    _LOG_TS   = time.strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(_LOG_DIR, f"trapx_{_LOG_TS}.log")
    _log_fh   = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

    # ── Write PID file so only THIS process can be targeted for stop ──────────
    _PID_FILE = os.path.join(_HERE, "trapx.pid")
    with open(_PID_FILE, "w") as _f:
        _f.write(str(os.getpid()))
    print(f"  📋  Log file   : examples/logs/trapx_{_LOG_TS}.log")
    print(f"  📌  PID {os.getpid()} written to examples/trapx.pid")
    print(f"  🛑  To stop only this process:")
    print(f"        PowerShell:  Stop-Process -Id (Get-Content examples/trapx.pid)")
    print(f"        Or just:     Ctrl+C  in this terminal")

    print("═"*65)
    print("  🚀  TRAP-X  –  Asymmetric Intraday Trap Detection Engine")
    print("  📊  Replay: NIFTY 2026-02-20")
    print()
    print("  TIMING MODEL:")
    print("  • 09:15 IST → process_mkt_open (PDC load + gap analysis)")
    print("  • 09:16 IST → process_minute for 09:15 closed bar")
    print("  • 09:17 IST → process_minute for 09:16 closed bar  … etc")
    print("═"*65)

    from langgraph_viz import visualize

    # ── Load CSV data (outside LangGraph, stored in module globals) ────────────
    print("\n  📂  Loading CSV data …")
    df = pd.read_csv(SPOT_FUT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    spot_df = df[df["instrument_type"] == "SPOT"].copy().reset_index(drop=True)
    fut_df  = df[df["instrument_type"] == "FUTURE"].copy().reset_index(drop=True)
    sig_df  = pd.read_csv(SIGNALS_CSV)
    sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
    sig_df = sig_df.sort_values("timestamp").reset_index(drop=True)

    session_start_ts = pd.Timestamp("2026-02-20 09:15:00")
    spot_session     = spot_df[spot_df["timestamp"] >= session_start_ts].copy()
    spot_minutes     = spot_session["timestamp"].sort_values().unique()
    last_bar_ts      = pd.Timestamp(spot_minutes[-1])

    print(f"  SPOT bars in session : {len(spot_minutes)}")
    print(f"  First closed bar     : {spot_minutes[0].strftime('%H:%M')}  "
          f"[analysed at {(spot_minutes[0] + pd.Timedelta(minutes=1)).strftime('%H:%M')} IST]")
    print(f"  Last  closed bar     : {last_bar_ts.strftime('%H:%M')}  "
          f"[analysed at {(last_bar_ts + pd.Timedelta(minutes=1)).strftime('%H:%M')} IST]")
    print(f"  FUTURE bars          : {len(fut_df)}")
    print(f"  Signal rows          : {len(sig_df)}")

    # ── Single graph, single memory, single config ────────────────────────────
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "trapx-replay-001"}}
    app    = create_trapx_app(memory)

    print("\n  🎨  Starting LangGraph-Viz server …")
    print("  🌐  Open http://localhost:8767 in your browser")
    print(f"  ⏩  Replaying {len(spot_minutes)} bars  "
          f"(09:15→{last_bar_ts.strftime('%H:%M')} closed)\n")

    with visualize(app, port=8767) as viz_app:
        start_ts = time.time()

        for i, bar_ts in enumerate(spot_minutes):
            # ── Accumulate data up to this bar ─────────────────────────────────
            row = spot_df[spot_df["timestamp"] == bar_ts]
            if row.empty:
                continue
            G_SPOT.append(row.iloc[0].to_dict())

            fut_rows = fut_df[fut_df["timestamp"] <= bar_ts]
            if not fut_rows.empty:
                lf = fut_rows.iloc[-1].to_dict()
                if not G_FUT or G_FUT[-1]["timestamp"] != lf["timestamp"]:
                    G_FUT.append(lf)

            sig_rows = sig_df[sig_df["timestamp"] <= bar_ts]
            if not sig_rows.empty:
                ls = sig_rows.iloc[-1].to_dict()
                if not G_SIG or G_SIG[-1].get("timestamp") != ls.get("timestamp"):
                    G_SIG.append(ls)

            # ── Build timing labels ────────────────────────────────────────────
            bar_time_str     = bar_ts.strftime("%H:%M")
            if i == 0:
                # At 09:15, process_mkt_open fires AT the opening bell
                trigger_time_str = "09:15"
                header = f"  🔔  [09:15 IST]  MARKET OPEN  (process_mkt_open + process_minute)"
            else:
                # Every other bar: closed bar T is analysed at T+1
                trigger_time_str = (bar_ts + pd.Timedelta(minutes=1)).strftime("%H:%M")
                header = (f"  ⏱️   [{trigger_time_str} IST]  BAR {bar_time_str} CLOSED  "
                          f"(bar #{i+1}/{len(spot_minutes)})")

            print(f"\n{'─'*65}")
            print(header)
            print(f"{'─'*65}")

            # ── Single invoke ──────────────────────────────────────────────────
            viz_app.invoke({
                "bar_index":    i,
                "bar_time":     bar_time_str,
                "trigger_time": trigger_time_str,
            }, config)

            # Progress every 50 bars
            if (i + 1) % 50 == 0 or (i + 1) == len(spot_minutes):
                elapsed = time.time() - start_ts
                print(f"\n  … {i+1}/{len(spot_minutes)} bars  "
                      f"({(i+1)/len(spot_minutes)*100:.0f}%)  "
                      f"elapsed={elapsed:.1f}s\n")

            time.sleep(0.02)

        elapsed = time.time() - start_ts
        print(f"\n  ✅  Replay complete: {len(spot_minutes)} bars in {elapsed:.1f}s")

        try:
            final = viz_app.get_state(config).values
            print_session_summary(final)
        except Exception as e:
            print(f"  (Could not fetch final state: {e})")

        print("\n  ⏸️   Server running. Open http://localhost:8767 to explore.")
        print("  Press Ctrl-C to exit.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  👋  Shutting down TRAP-X …")

    # ── Clean up PID file and log on exit ───────────────────────────────────────
    try:
        os.remove(_PID_FILE)
        print(f"  🗑️   PID file removed ({_PID_FILE})")
    except OSError:
        pass

    print("\n  ✓  Done!\n")

    # Restore stdout/stderr and close log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fh.close()
    print(f"\n  📄  Log saved → {_LOG_PATH}")
