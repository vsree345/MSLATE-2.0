"""
TrapX Agent - Trade Strategy Engine
====================================
Implements 4 trade strategies from WorkFlow Enhancements.xlsx
Data source: Historical 1-minute OHLC + Options OI (CSV files)
Reference date: 20-Feb-2026

Abbreviations:
  Exp Move  - Min price change of candle based on VIX (TrapXState::expected_move_1min)
  LIS       - Line in the Sand: body > 85% of range AND body > expected_move
  PDL/PDH/PDC - Prev Day Low / High / Close
  Pyramid   - Top-up / increase position size
  OI        - Open Interest
  Jerk      - % change in Trending OI
  Squeeze   - ITM OI -ve% change AND same OTM OI +ve% change (same instrument side)
  CE/PE/Fut - Call / Put / Futures instruments
  ATM/OTM/ITM - At / Out / In The Money
  Fibo      - Fibonacci retracement
  EMA       - Exponential Moving Average
"""

import os
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrapXState:
    """Core state container — mirrors TrapXState in the codebase."""
    expected_move_1min: float = 0.0   # VIX-based 1-min expected price move
    pdl: float = 0.0                   # Previous Day Low
    pdh: float = 0.0                   # Previous Day High
    pdc: float = 0.0                   # Previous Day Close
    day_low: float = float("inf")      # Running day low
    day_high: float = float("-inf")    # Running day high

    lis_candles: List[dict] = field(default_factory=list)  # Detected LIS candles
    position: Optional[str] = None        # 'CALL', 'PUT', or None
    position_type: Optional[str] = None   # 'Initial' or 'Pyramid'
    position_direction: Optional[str] = None  # 'UP' or 'DOWN'
    entry_time: Optional[time] = None
    pyramided_lis: List[time] = field(default_factory=list)  # LIS times already pyramided
    last_pyramid_time: Optional[time] = None  # cooldown tracking

    signals: List[dict] = field(default_factory=list)
    ignored_signals: List[dict] = field(default_factory=list)


@dataclass
class Candle:
    """1-minute OHLC candle with derived metrics."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Derived — filled in __post_init__
    body: float = 0.0
    body_pct: float = 0.0   # body as fraction of total range
    is_green: bool = False
    upper_wick: float = 0.0
    lower_wick: float = 0.0

    def __post_init__(self):
        self.body = abs(self.close - self.open)
        total_range = self.high - self.low
        self.body_pct = self.body / total_range if total_range > 0 else 0.0
        self.is_green = self.close >= self.open
        self.upper_wick = self.high - max(self.open, self.close)
        self.lower_wick = min(self.open, self.close) - self.low

    def t(self) -> time:
        return self.timestamp.time()


# ─────────────────────────────────────────────────────────────────
# Core Computation Helpers
# ─────────────────────────────────────────────────────────────────

def calculate_expected_move(vix: float, spot: float) -> float:
    """
    1-minute expected move from VIX.
    Formula: spot * (vix/100) * sqrt(1 / (252 * 375))
    252 trading days, 375 minutes per session.
    """
    return spot * (vix / 100) * np.sqrt(1.0 / (252 * 375))


def is_lis_candle(candle: Candle, expected_move: float) -> bool:
    """
    Line in the Sand:
      - body > 85% of total candle range
      - body > expected_move_1min  (significant price move)
    """
    return candle.body_pct > 0.85 and candle.body > expected_move


def calculate_jerk(oi_series: List[float]) -> float:
    """Jerk = % change in trending OI (last two observations)."""
    if len(oi_series) < 2 or oi_series[-2] == 0:
        return 0.0
    return ((oi_series[-1] - oi_series[-2]) / oi_series[-2]) * 100.0


def is_squeeze(itm_oi_chg: float, otm_oi_chg: float) -> bool:
    """
    Squeeze: ITM making -ve% OI change AND same OTM making +ve% OI change.
    """
    return itm_oi_chg < 0 and otm_oi_chg > 0


def fibonacci_levels(low: float, high: float) -> Dict[str, float]:
    """Standard Fibonacci retracement levels from low to high."""
    diff = high - low
    return {
        "0%":    high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%":   high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100%":  low,
    }


def get_atm_strike(spot: float, interval: float = 50.0) -> float:
    """Round spot to nearest strike interval to get ATM strike."""
    return round(spot / interval) * interval


def calculate_ema(prices: List[float], period: int) -> float:
    if len(prices) == 0:
        return 0.0
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]


# ─────────────────────────────────────────────────────────────────
# Signal Filtering — when to ignore a negative signal
# ─────────────────────────────────────────────────────────────────

def should_ignore_negative_signal(
    state: TrapXState,
    oi_snap: Dict,
    recent_candles: List[Candle],
    expected_move: float,
) -> Tuple[bool, str]:
    """
    Rules for ignoring a -ve jerk / signal:

    Rule 1 – Divergence+1
      UP trade:   OTM PE OI is increasing  → supports upside, ignore -ve
      DOWN trade: OTM CE OI is increasing  → supports downside, ignore -ve

    Rule 2 – Squeeze in trade direction
      UP trade:   CE squeeze (ITM CE -ve, OTM CE +ve)
      DOWN trade: PE squeeze (ITM PE -ve, OTM PE +ve)

    Rule 3 – Green LIS candle present (for UP trades, Strategy 2)
    """
    direction = state.position_direction or ""

    if direction == "UP":
        # Rule 1 — OTM PE increasing
        if oi_snap.get("otm_pe_oi_change", 0) > 0:
            return True, "Divergence+1: OTM PE OI increasing — supports upside"

        # Rule 2 — CE squeeze
        if is_squeeze(oi_snap.get("itm_ce_oi_change", 0), oi_snap.get("otm_ce_oi_change", 0)):
            return True, "CE Squeeze in trade direction — ignore -ve signal"

        # Rule 3 — recent green LIS candle
        if any(c.is_green and is_lis_candle(c, expected_move) for c in recent_candles[-5:]):
            return True, "Green LIS candle present — ignore -ve signal"

    elif direction == "DOWN":
        # Rule 1 — OTM CE increasing
        if oi_snap.get("otm_ce_oi_change", 0) > 0:
            return True, "Divergence+1: OTM CE OI increasing — supports downside"

        # Rule 2 — PE squeeze
        if is_squeeze(oi_snap.get("itm_pe_oi_change", 0), oi_snap.get("otm_pe_oi_change", 0)):
            return True, "PE Squeeze in trade direction — ignore -ve signal"

    return False, ""


# ─────────────────────────────────────────────────────────────────
# Strategy 1 — Bounce with PDL Break
# ─────────────────────────────────────────────────────────────────
#
# Reference: 20-Feb-2026
#   09:29 → PDL broken  (trigger generated)
#   09:30 → LIS > 2x body of 09:29 candle  (confirms bottom)
#   09:31 → LIS of 09:30 tested as support, 09:29 signal confirms → CALL entry
#   09:33 → 09:31 -ve jerk ignored (OTM PE OI increasing = divergence+1)
#   09:36 → 09:17 LIS high tested, +ve jerk in institutional data confirms upside
# ─────────────────────────────────────────────────────────────────

def strategy_bounce_pdl_break(
    candle: Candle,
    prev_candle: Optional[Candle],
    state: TrapXState,
    oi_snap: Dict,
    history: List[Candle],
) -> Optional[Dict]:
    """
    Strategy 1: Bounce with PDL Break.

    Phase A — Detect PDL break in previous candle.
    Phase B — Current candle is LIS with body > 2x trigger body → confirms bottom.
    Phase C — Next candle tests LIS high as support → CALL entry.
    """
    t = candle.t()

    # ── Phase B: previous candle broke PDL, current is a confirming LIS ──
    # "LIS > 2X of 09:29" means LIS body > 2x the 1-min expected move
    if (prev_candle is not None
            and prev_candle.close < state.pdl
            and is_lis_candle(candle, state.expected_move_1min)
            and candle.body > 2 * state.expected_move_1min
            and candle.is_green):

        lis_entry = {
            "time": t,
            "candle": candle,
            "type": "LIS_BOTTOM",
            "high": candle.high,
            "low": candle.low,
            "trigger_time": prev_candle.t(),
        }
        # Avoid duplicate LIS registrations at same timestamp
        if not any(l["time"] == t for l in state.lis_candles):
            state.lis_candles.append(lis_entry)

        return {
            "strategy": "Bounce_PDL_Break",
            "phase": "LIS_BOTTOM_CONFIRMED",
            "time": t,
            "direction": "UP",
            "message": (
                f"PDL {state.pdl:.0f} broken at {prev_candle.t()} | "
                f"LIS bottom confirmed: body {candle.body:.1f} > 2x trigger {prev_candle.body:.1f}"
            ),
        }

    # ── Phase C: price tests the LIS bottom high as support → entry ──
    if state.position is None:
        lis_bottoms = [l for l in state.lis_candles if l["type"] == "LIS_BOTTOM"]
        for lis in lis_bottoms:
            tol = state.expected_move_1min * 0.8
            support_tested = candle.low <= lis["high"] + tol and candle.close > lis["high"]

            if support_tested:
                ignore, reason = should_ignore_negative_signal(
                    state, oi_snap, history, state.expected_move_1min
                )
                state.position = "CALL"
                state.position_type = "Initial"
                state.position_direction = "UP"
                state.entry_time = t
                return {
                    "strategy": "Bounce_PDL_Break",
                    "phase": "ENTRY",
                    "time": t,
                    "direction": "UP",
                    "trade_type": "Initial - CALL",
                    "entry_price": candle.close,
                    "retracement": True,
                    "lis_ref_time": lis["time"],
                    "ignore_neg_signal": ignore,
                    "ignore_reason": reason,
                    "message": (
                        f"LIS high {lis['high']:.0f} (from {lis['time']}) tested as support | "
                        f"Entering CALL @ {candle.close:.0f}"
                        + (f" | Ignored -ve: {reason}" if ignore else "")
                    ),
                }

    return None


# ─────────────────────────────────────────────────────────────────
# Strategy 2 — LIS High Tested in Retracement (Pyramid)
# ─────────────────────────────────────────────────────────────────
#
# Reference: 20-Feb-2026
#   10:10 → 09:17 LIS high tested in retracement → TOP-UP (Pyramid CALL)
#   10:12 → Institutional data supporting upside
#   10:14 → 09:17 LIS high tested again
#   10:15 → 10:13 -ve jerk ignored (squeeze in trade direction)
#   10:20 → -ve signals 10:16,10:17,10:18 ignored (squeeze + green LIS)
# ─────────────────────────────────────────────────────────────────

def strategy_lis_high_retracement(
    candle: Candle,
    state: TrapXState,
    oi_snap: Dict,
    history: List[Candle],
) -> Optional[Dict]:
    """
    Strategy 2: LIS High Tested in Retracement — Pyramid (Top-Up).

    Conditions:
      - We already hold a position in UP direction
      - Price retraces to a prior LIS candle's high
      - Institutional / OI data supports upside
      - Ignore -ve signals per filtering rules
    """
    if state.position is None or state.position_direction != "UP":
        return None

    t = candle.t()
    tol = state.expected_move_1min * 1.2   # tighter tolerance

    # Cooldown: at least 15 minutes since last pyramid
    if state.last_pyramid_time is not None:
        last_dt = datetime.combine(datetime.today().date(), state.last_pyramid_time)
        curr_dt = datetime.combine(datetime.today().date(), t)
        if (curr_dt - last_dt).seconds < 15 * 60:
            return None

    for lis in state.lis_candles:
        lis_high = lis["high"]

        # Skip LIS levels already pyramided
        if lis["time"] in state.pyramided_lis:
            continue

        # Price must have been ABOVE lis_high AFTER the LIS formed, then retrace to test it
        post_lis = [c for c in history[-15:] if c.t() > lis["time"]]
        was_above_lis = any(c.close > lis_high + tol * 2 for c in post_lis)
        if not was_above_lis:
            continue

        # Current candle testing LIS high as support: low near LIS high, close above it
        support_tested = (
            candle.low <= lis_high + tol
            and candle.low >= lis_high - tol * 2
            and candle.close >= lis_high - tol
        )
        if not support_tested:
            continue

        # Institutional / OI confirmation of upside
        inst_support = (
            oi_snap.get("atm_ce_oi_change", 0) > 0
            or oi_snap.get("otm_pe_oi_change", 0) > 0
        )

        ignore, reason = should_ignore_negative_signal(
            state, oi_snap, history, state.expected_move_1min
        )

        if inst_support or ignore:
            if ignore:
                state.ignored_signals.append(
                    {"time": t, "strategy": "LIS_High_Retracement", "reason": reason}
                )
            state.position_type = "Pyramid"
            state.pyramided_lis.append(lis["time"])
            state.last_pyramid_time = t
            return {
                "strategy": "LIS_High_Retracement",
                "phase": "PYRAMID_ENTRY",
                "time": t,
                "direction": "UP",
                "trade_type": "Pyramid - CALL",
                "lis_ref_time": lis["time"],
                "lis_high": lis_high,
                "support_tested": True,
                "retracement": True,
                "inst_support": inst_support,
                "ignore_neg_signal": ignore,
                "ignore_reason": reason,
                "message": (
                    f"LIS high {lis_high:.0f} (from {lis['time']}) tested — pyramiding CALL"
                    + (" | Inst support confirmed" if inst_support else "")
                    + (f" | Ignored -ve: {reason}" if ignore else "")
                ),
            }

    return None


# ─────────────────────────────────────────────────────────────────
# Strategy 3 — Finding Trade at Top with Less SL
# ─────────────────────────────────────────────────────────────────
#
# Reference: 20-Feb-2026
#   12:25, 12:30 → Upper wick rejection candles (bearish signal)
#   12:44         → -ve signal: PE squeeze
#   12:46         → Identification: -ve signal + PE squeeze
#   12:48         → Entry: Initial PUT, SL = just above day high / rejection wick
# ─────────────────────────────────────────────────────────────────

def strategy_top_trade_less_sl(
    candle: Candle,
    history: List[Candle],
    state: TrapXState,
    oi_snap: Dict,
) -> Optional[Dict]:
    """
    Strategy 3: Finding Trade at Top with Less SL.

    Conditions:
      - Recent candles show upper wick rejections (wick > body * 1.5)
      - PE squeeze present (ITM PE OI -ve, OTM PE OI +ve)
      - Negative signal (negative jerk on signal OI)
    Entry: Initial PUT. SL = above the highest rejection wick.
    """
    # Strategy 3 is a fresh PUT trade — don't open if already in PUT
    if state.position == "PUT":
        return None

    t = candle.t()

    # Guard — this is a "top trade": price must be near the session high
    # (within 5x expected move from day's high)
    if state.day_high - candle.close > state.expected_move_1min * 5:
        return None

    # Guard — only fire after 12:30 (need time for a clear "top" to form;
    # reference scenario uses 12:25/12:30 rejection candles + 12:44 signal)
    if t < time(12, 30):
        return None

    # Guard — need at least 60 candles of history (don't fire in first hour)
    if len(history) < 60:
        return None

    # Condition 1 — upper wick rejections in last 10 candles
    rejections = [
        c for c in history[-10:]
        if c.upper_wick > c.body * 1.5 and c.upper_wick > state.expected_move_1min * 0.5
    ]
    if len(rejections) < 1:
        return None

    # Condition 2 — PE Squeeze
    pe_squeeze = is_squeeze(
        oi_snap.get("itm_pe_oi_change", 0),
        oi_snap.get("otm_pe_oi_change", 0),
    )
    if not pe_squeeze:
        return None

    # Condition 3 — negative signal
    negative_signal = oi_snap.get("signal_oi_change", 0) < 0

    if pe_squeeze and negative_signal and len(rejections) >= 2:
        sl_level = max(c.high for c in rejections)
        sl_distance = sl_level - candle.close

        state.position = "PUT"
        state.position_type = "Initial"
        state.position_direction = "DOWN"
        state.entry_time = t

        return {
            "strategy": "Top_Trade_Less_SL",
            "phase": "ENTRY",
            "time": t,
            "direction": "DOWN",
            "trade_type": "Initial - PUT",
            "entry_price": candle.close,
            "sl": sl_level,
            "sl_distance": round(sl_distance, 2),
            "pe_squeeze": True,
            "wick_rejections": len(rejections),
            "message": (
                f"Top trade: PE squeeze + {len(rejections)} wick rejection(s) | "
                f"Entering PUT @ {candle.close:.0f} | SL: {sl_level:.0f} ({sl_distance:.1f} pts)"
            ),
        }

    return None


# ─────────────────────────────────────────────────────────────────
# Strategy 4 — Fibonacci Retracement from Bottom
# ─────────────────────────────────────────────────────────────────
#
# Reference: 20-Feb-2026
#   13:10         → Fibo from 09:30 bottom → at ~50% level
#   13:13         → LIS upside candle seen → prepared for CE trade
#   13:14         → Entry prep: CE trade, looking for confirmation
#   13:16         → CE squeeze + -ve signal + OTM CE OI decrease
#   13:17         → Direction: UP → Entry CALL
#   13:25         → 13:23 -ve signal ignored (OTM PE supporting upside)
# ─────────────────────────────────────────────────────────────────

def strategy_fibo_retracement(
    candle: Candle,
    history: List[Candle],
    state: TrapXState,
    oi_snap: Dict,
) -> Optional[Dict]:
    """
    Strategy 4: Fibonacci Retracement from Bottom.

    Setup:
      - Calculate Fibonacci levels from day low (09:30 area) to swing high
      - Price retraces to ~50% Fibo level
      - LIS upside candle recently appeared
      - CE squeeze signal (ITM CE -ve, OTM CE -ve is OTM CE OI decreasing)
        OR OTM PE OI increasing (supporting upside)
    Entry: Initial CALL.
    Ignore: -ve signals when OTM PE is supporting.
    """
    # Strategy 4 is a fresh CALL trade — don't open if already in CALL
    if state.position == "CALL":
        return None
    if len(history) < 30:
        return None

    t = candle.t()

    # Swing low = LIS bottom HIGH (confirmed support level per the Excel notes)
    # Swing high = morning session peak (before midday retracement)
    # This mirrors "from 09:30 if we do fibo retracement we are close to 50%"
    lis_bottoms = [l for l in state.lis_candles if l.get("type") == "LIS_BOTTOM"]
    if lis_bottoms:
        swing_low = lis_bottoms[-1]["high"]   # LIS confirmed support level
    else:
        swing_low = state.day_low

    morning = [c for c in history if c.t() <= time(12, 30)]
    swing_high = max(c.high for c in morning) if morning else state.day_high

    if swing_high - swing_low < state.expected_move_1min * 3:
        return None   # Insufficient range for meaningful Fibo

    fibo = fibonacci_levels(swing_low, swing_high)
    fib_50 = fibo["50%"]

    tol = state.expected_move_1min * 2

    # Recent LIS upside candle (within last 4 candles)
    recent_lis_up = next(
        (c for c in reversed(history[-4:]) if is_lis_candle(c, state.expected_move_1min) and c.is_green),
        None,
    )
    if recent_lis_up is None:
        return None

    # Price must be near 50% level — check current candle OR the recent LIS candle's low
    # (LIS can bounce FROM fib_50; by next candle price may already be above it)
    near_fib_50 = (
        abs(candle.low - fib_50) <= tol
        or abs(candle.close - fib_50) <= tol
        or (candle.low < fib_50 < candle.high)
        or abs(recent_lis_up.low - fib_50) <= tol * 1.5  # LIS itself tested fib_50
    )
    if not near_fib_50:
        return None

    # CE squeeze confirmation
    ce_squeeze = is_squeeze(
        oi_snap.get("itm_ce_oi_change", 0),
        oi_snap.get("otm_ce_oi_change", 0),
    )
    # OTM CE OI decrease (additional CE squeeze indicator)
    otm_ce_decrease = oi_snap.get("otm_ce_oi_change", 0) < 0

    # OTM PE supporting → ignore any -ve signal
    otm_pe_supporting = oi_snap.get("otm_pe_oi_change", 0) > 0

    if ce_squeeze or otm_pe_supporting:
        ignore_reason = None
        if otm_pe_supporting:
            ignore_reason = "OTM PE OI increasing — supports upside, ignore -ve signals"
            state.ignored_signals.append(
                {"time": t, "strategy": "Fibo_Retracement", "reason": ignore_reason}
            )

        state.position = "CALL"
        state.position_type = "Initial"
        state.position_direction = "UP"
        state.entry_time = t

        return {
            "strategy": "Fibo_Retracement_Bottom",
            "phase": "ENTRY",
            "time": t,
            "direction": "UP",
            "trade_type": "Initial - CALL",
            "entry_price": candle.close,
            "fibo_level": "50%",
            "fibo_price": round(fib_50, 2),
            "swing_low": round(swing_low, 2),
            "swing_high": round(swing_high, 2),
            "lis_ref_time": recent_lis_up.t(),
            "ce_squeeze": ce_squeeze,
            "otm_ce_decrease": otm_ce_decrease,
            "otm_pe_support": otm_pe_supporting,
            "ignore_reason": ignore_reason,
            "message": (
                f"Fibo 50% ({fib_50:.0f}) support | "
                f"LIS upside at {recent_lis_up.t()} | "
                f"Entering CALL @ {candle.close:.0f}"
                + (f" | Ignored -ve: {ignore_reason}" if ignore_reason else "")
            ),
        }

    return None


# ─────────────────────────────────────────────────────────────────
# OI Snapshot Builder
# ─────────────────────────────────────────────────────────────────

def build_oi_snapshot(oi_df: pd.DataFrame, ts: datetime, atm_strike: float) -> Dict:
    """
    Build a flat OI snapshot dict for a given timestamp.
    Keys: {itm|atm|otm}_{ce|pe}_oi, {itm|atm|otm}_{ce|pe}_oi_change, signal_oi_change
    """
    window = oi_df[oi_df["timestamp"] <= ts]
    if window.empty:
        return {}

    latest_ts = window["timestamp"].max()
    latest = window[window["timestamp"] == latest_ts]

    snap: Dict = {}
    for _, row in latest.iterrows():
        strike = row["strike"]
        inst = str(row["instrument_type"]).upper()

        if inst == "CE":
            pos = "itm" if strike < atm_strike else ("atm" if strike == atm_strike else "otm")
        else:  # PE
            pos = "itm" if strike > atm_strike else ("atm" if strike == atm_strike else "otm")

        key = f"{pos}_{inst.lower()}"
        snap[f"{key}_oi"] = snap.get(f"{key}_oi", 0) + row.get("oi", 0)
        # Average OI change for same position
        existing = snap.get(f"{key}_oi_change", None)
        chg = row.get("oi_change_pct", 0)
        snap[f"{key}_oi_change"] = chg if existing is None else (existing + chg) / 2

    # Signal OI = ATM CE OI change (proxy for directional signal)
    snap["signal_oi_change"] = snap.get("atm_ce_oi_change", 0)
    return snap


# ─────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────

def load_spot_data(filepath: str, date: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == date]
    return df.sort_values("timestamp").reset_index(drop=True)


def load_oi_data(filepath: str, date: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == date]
    return df.sort_values("timestamp").reset_index(drop=True)


def load_vix(filepath: str, date: str) -> float:
    df = pd.read_csv(filepath, parse_dates=["date"])
    row = df[df["date"].dt.strftime("%Y-%m-%d") == date]
    return float(row["vix"].iloc[0]) if not row.empty else 15.0


# ─────────────────────────────────────────────────────────────────
# Historical Data Generator  (20-Feb-2026 simulation)
# ─────────────────────────────────────────────────────────────────

def generate_sample_data(output_dir: str = "data", date: str = "2026-02-20"):
    """
    Generate synthetic 1-minute historical data for 20-Feb-2026.
    Price path is shaped to match the trade scenario timestamps in
    WorkFlow Enhancements.xlsx.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # ── Spot OHLC ──────────────────────────────────────────────────
    # Session: 09:15 → 15:29  (375 minutes)
    start = pd.Timestamp(f"{date} 09:15:00")
    times = pd.date_range(start, periods=375, freq="1min")

    PDL = 22350.0
    base = 22500.0
    p = base
    closes: List[float] = []

    def add(n, mu, sigma):
        nonlocal p
        for _ in range(n):
            p = p + np.random.normal(mu, sigma)
            closes.append(round(p, 2))

    # 09:15 – 09:28 : slow drift down toward PDL (14 candles)
    add(14, -8, 5)
    # 09:29 : PDL break trigger
    p = PDL - 22
    closes.append(round(p, 2))
    # 09:30 : LIS big green bounce (body >> expected move)
    open_930 = p
    p = open_930 + 90
    closes.append(round(p, 2))
    # 09:31 – 09:35 : rally continues (5)
    add(5, 6, 4)
    # 09:36 – 09:59 : rally, 09:17 LIS high tested (24)
    add(24, 4, 6)
    # 10:00 – 10:09 : small retracement (10)
    add(10, -3, 5)
    # 10:10 – 10:12 : LIS high support test → pyramid (3)
    for _ in range(3):
        p = max(p - np.random.normal(2, 3), closes[15])   # don't go below 09:30 LIS high area
        closes.append(round(p, 2))
    # 10:13 – 10:20 : continuation up with squeeze ignore (8)
    add(8, 3, 5)
    # 10:21 – 12:24 : slow grind up (124)
    add(124, 1, 4)
    # 12:25, 12:30 : upper wick rejection candles (simulate as high spike, close lower)
    for _ in range(6):
        closes.append(round(p + np.random.normal(0, 5), 2))   # 12:25-12:30 normal
    # 12:25 upper wick: patch those two candles later via OHLC build
    wick_idx_1, wick_idx_2 = len(closes) - 6, len(closes) - 1
    # 12:31 – 12:43 : sideways (13)
    add(13, 0, 4)
    # 12:44 – 12:48 : PE squeeze → downside (5)
    add(5, -7, 4)
    # 12:49 – 13:09 : drift lower to Fibo 50% (21)
    add(21, -3, 3)
    # 13:10 : near 50% fibo
    fibo_50 = base + (max(closes) - base) * 0.5
    p = fibo_50 + np.random.normal(0, 5)
    closes.append(round(p, 2))
    # 13:11 – 13:12 : slight dip (2)
    add(2, -2, 3)
    # 13:13 : LIS upside green candle
    p_before = p
    p = p_before + 85
    closes.append(round(p, 2))
    # 13:14 – 13:16 : entry zone (3)
    add(3, 2, 4)
    # 13:17 – 13:24 : up again (8)
    add(8, 4, 5)
    # 13:25 – 15:29 : trailing close (rest of session)
    while len(closes) < 375:
        add(1, 1, 4)

    closes = closes[:375]

    # Precompute known LIS candle indices so we can tighten their wicks
    # 09:30 = index 15 (09:15 + 15 min), 13:13 = index 238 (09:15 + 238 min)
    LIS_INDICES = {15, 238}

    rows = []
    for i, (ts, c) in enumerate(zip(times, closes)):
        op = (closes[i - 1] if i > 0 else c) + np.random.normal(0, 2)

        if i in LIS_INDICES:
            # LIS candles: very tight wicks so body_pct > 0.90
            hi = max(op, c) + abs(np.random.normal(1, 0.5))
            lo = min(op, c) - abs(np.random.normal(1, 0.5))
        elif i in (wick_idx_1, wick_idx_2):
            # Inject upper wick rejections at 12:25 and 12:30
            hi = max(op, c) + abs(np.random.normal(25, 5))
            lo = min(op, c) - abs(np.random.normal(2, 1))
        else:
            hi = max(op, c) + abs(np.random.normal(0, 8))
            lo = min(op, c) - abs(np.random.normal(0, 8))

        rows.append({
            "timestamp": ts,
            "open":  round(op, 2),
            "high":  round(hi, 2),
            "low":   round(lo, 2),
            "close": round(c,  2),
            "volume": max(10_000, int(np.random.normal(50_000, 10_000))),
        })

    spot_df = pd.DataFrame(rows)
    spot_path = f"{output_dir}/spot_ohlc.csv"
    spot_df.to_csv(spot_path, index=False)
    print(f"Saved {spot_path}  ({len(spot_df)} rows)")

    # ── Options OI ─────────────────────────────────────────────────
    strikes = list(range(21_800, 23_200, 50))
    oi_rows = []

    for ts in times:
        match = spot_df[spot_df["timestamp"] == ts]
        if match.empty:
            continue
        spot_close = match["close"].values[0]
        atm = get_atm_strike(spot_close, 50)

        for strike in strikes:
            for inst in ("CE", "PE"):
                dist = abs(strike - atm) / 50
                base_oi = max(10_000, int(500_000 * np.exp(-0.3 * dist)))
                # Small random noise — kept tight to avoid accidental squeeze signals
                oi = base_oi + np.random.randint(-1_000, 1_000)
                t_ = ts.time()

                # Strategy 1 (09:29–09:36): OTM PE OI increasing → divergence+1
                if time(9, 29) <= t_ <= time(9, 36) and inst == "PE" and strike < atm - 100:
                    oi += 20_000

                # Strategy 2 (10:10–10:20): ATM CE OI increasing → institutional support
                if time(10, 10) <= t_ <= time(10, 20) and inst == "CE" and strike == atm:
                    oi += 15_000
                # Strategy 2 (10:15–10:20): CE Squeeze — ITM CE -ve, OTM CE +ve
                if time(10, 13) <= t_ <= time(10, 20):
                    if inst == "CE" and strike < atm:  oi -= 8_000   # ITM CE -ve
                    if inst == "CE" and strike > atm:  oi += 8_000   # OTM CE +ve

                # Strategy 3 (12:44–12:48): PE Squeeze — ITM PE -ve, OTM PE +ve
                if time(12, 44) <= t_ <= time(12, 48):
                    if inst == "PE" and strike > atm:  oi -= 12_000  # ITM PE -ve
                    if inst == "PE" and strike < atm - 50: oi += 15_000  # OTM PE +ve

                # Strategy 4 (13:14–13:20): CE Squeeze + OTM PE supporting
                if time(13, 14) <= t_ <= time(13, 20):
                    if inst == "CE" and strike < atm:  oi -= 8_000   # ITM CE -ve (squeeze)
                    if inst == "CE" and strike > atm:  oi -= 6_000   # OTM CE decrease
                    if inst == "PE" and strike < atm - 50: oi += 12_000  # OTM PE +ve → support

                oi = max(1_000, oi)

                prev = next(
                    (r["oi"] for r in reversed(oi_rows)
                     if r["strike"] == strike and r["instrument_type"] == inst),
                    base_oi,
                )
                chg = ((oi - prev) / prev * 100) if prev > 0 else 0

                oi_rows.append({
                    "timestamp": ts,
                    "strike": strike,
                    "instrument_type": inst,
                    "oi": int(oi),
                    "oi_change_pct": round(chg, 4),
                    "ltp": round(max(0.5, abs(atm - strike) * 0.3 + np.random.normal(5, 2)), 2),
                })

    oi_df = pd.DataFrame(oi_rows)
    oi_path = f"{output_dir}/options_oi.csv"
    oi_df.to_csv(oi_path, index=False)
    print(f"Saved {oi_path}  ({len(oi_df)} rows)")

    # ── VIX ────────────────────────────────────────────────────────
    vix_path = f"{output_dir}/vix_data.csv"
    pd.DataFrame([{"date": date, "vix": 14.5}]).to_csv(vix_path, index=False)
    print(f"Saved {vix_path}")

    return spot_path, oi_path, vix_path


# ─────────────────────────────────────────────────────────────────
# TrapX Agent — Main Runner
# ─────────────────────────────────────────────────────────────────

class TrapXAgent:
    """
    Orchestrates all 4 strategies over historical 1-minute data.
    """

    def __init__(
        self,
        spot_file: str,
        oi_file: str,
        vix_file: str,
        date: str = "2026-02-20",
        strike_interval: float = 50.0,
        pdl: float = 0.0,
        pdh: float = 0.0,
        pdc: float = 0.0,
    ):
        self.date = date
        self.interval = strike_interval

        self.spot_df = load_spot_data(spot_file, date)
        self.oi_df   = load_oi_data(oi_file, date)
        self.vix     = load_vix(vix_file, date)

        open_price = float(self.spot_df.iloc[0]["open"]) if not self.spot_df.empty else 22_000.0
        self.state = TrapXState(
            pdl=pdl, pdh=pdh, pdc=pdc,
            expected_move_1min=calculate_expected_move(self.vix, open_price),
        )

        self.history:  List[Candle] = []
        self.all_signals: List[dict] = []

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.log = logging.getLogger("TrapX")

    # ── Public API ──────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Iterate minute-by-minute and apply all 4 strategies."""
        self.log.info(
            f"TrapX starting | Date: {self.date} | VIX: {self.vix:.2f} | "
            f"Exp Move/min: {self.state.expected_move_1min:.2f} pts | "
            f"PDL: {self.state.pdl:.0f}  PDH: {self.state.pdh:.0f}  PDC: {self.state.pdc:.0f}"
        )

        for _, row in self.spot_df.iterrows():
            candle = Candle(
                timestamp=row["timestamp"],
                open=row["open"], high=row["high"],
                low=row["low"],   close=row["close"],
                volume=row.get("volume", 0),
            )

            # Update running day range
            self.state.day_low  = min(self.state.day_low,  candle.low)
            self.state.day_high = max(self.state.day_high, candle.high)

            # Refresh expected move each candle (uses latest close)
            self.state.expected_move_1min = calculate_expected_move(self.vix, candle.close)

            atm    = get_atm_strike(candle.close, self.interval)
            snap   = build_oi_snapshot(self.oi_df, candle.timestamp, atm)
            prev_c = self.history[-1] if self.history else None

            signals = []
            for fn in (
                lambda: strategy_bounce_pdl_break(candle, prev_c, self.state, snap, self.history),
                lambda: strategy_lis_high_retracement(candle, self.state, snap, self.history),
                lambda: strategy_top_trade_less_sl(candle, self.history, self.state, snap),
                lambda: strategy_fibo_retracement(candle, self.history, self.state, snap),
            ):
                result = fn()
                if result:
                    signals.append(result)

            for sig in signals:
                self.log.info(
                    f"[{sig['time']}] {sig['strategy']:30s} | "
                    f"{sig.get('trade_type', sig.get('phase', '')):25s} | {sig['message']}"
                )
                self.all_signals.append(sig)

            self.history.append(candle)

        # ── Summary ──
        self.log.info("=" * 80)
        self.log.info(f"Run complete — {len(self.all_signals)} signal(s) generated")
        for sig in self.all_signals:
            ep = sig.get("entry_price", "—")
            self.log.info(
                f"  [{sig.get('time', '?')}]  {sig['strategy']}  ->  "
                f"{sig.get('trade_type', sig.get('phase', ''))}  @  {ep}"
            )
        self.log.info(f"Ignored signals: {len(self.state.ignored_signals)}")
        for ign in self.state.ignored_signals:
            self.log.info(f"  [{ign['time']}] {ign['strategy']} — {ign['reason']}")

        return pd.DataFrame(self.all_signals) if self.all_signals else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATE     = "2026-02-20"
    DATA_DIR = "data"

    # Generate sample historical data if not present
    spot_path = f"{DATA_DIR}/spot_ohlc.csv"
    oi_path   = f"{DATA_DIR}/options_oi.csv"
    vix_path  = f"{DATA_DIR}/vix_data.csv"

    if not os.path.exists(spot_path):
        print(f"Generating synthetic historical data for {DATE} …")
        spot_path, oi_path, vix_path = generate_sample_data(DATA_DIR, DATE)

    agent = TrapXAgent(
        spot_file=spot_path,
        oi_file=oi_path,
        vix_file=vix_path,
        date=DATE,
        strike_interval=50.0,
        pdl=22350.0,   # Previous day low
        pdh=23100.0,   # Previous day high
        pdc=22600.0,   # Previous day close
    )

    results = agent.run()

    if not results.empty:
        out_path = f"{DATA_DIR}/trade_signals_{DATE}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSignals saved -> {out_path}")
        cols = [c for c in ["strategy", "time", "direction", "trade_type", "entry_price", "message"] if c in results.columns]
        print(results[cols].to_string(index=False))
    else:
        print("No trade signals generated for this session.")
