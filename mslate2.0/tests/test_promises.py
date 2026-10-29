"""Behavioural tests for the promise-critical nodes, driven by synthetic bars.

Run:  venv\\Scripts\\python -m pytest tests -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import TrapX_agent as T  # noqa: E402

EM = 17.0


def bar(t, o, h, l, c, vol=0):
    return dict(timestamp=f"2026-02-20 {t}:00", open=o, high=h, low=l, close=c, volume=vol)


@pytest.fixture(autouse=True)
def fresh_globals():
    for g in (T.G_SPOT, T.G_FUT, T.G_SIG):
        g.clear()
    T.G_PDC.clear()
    T.G_PDC.update(prev_day_high=25885.3, prev_day_low=25388.75, prev_day_close=25454.35, vix_close=13.46)
    yield


def state(**kw):
    s = dict(bar_index=10, bar_time="10:00", trigger_time="10:01", expected_move_1min=EM,
             exec_log=[], realized_pnl=0.0, equity=float(T.CFG["capital"]),
             equity_peak=float(T.CFG["capital"]), max_drawdown_pct=0.0, kill_switch=False,
             pos=None, wf_approved=None, wf_fired=[], bo_state={})
    s.update(kw)
    return s


def open_call(entry=25500.0, stop=25450.0, lots=2):
    R = entry - stop
    return dict(strategy="S1", action="S1_INITIAL_CALL", direction="CALL", entry=entry, avg=entry,
                stop=stop, init_stop=stop, target=entry + 2 * R, R=R, lots=lots, base_lots=lots,
                peak=entry, trailing=False, opened_bar="09:30", tier="FULL", score=85)


def step(s, t, o, h, l, c):
    T.G_SPOT.append(bar(t, o, h, l, c))
    s["bar_time"] = t
    return T.position_manager(s)


def exits(s):
    return [e for e in s["exec_log"] if e["type"] == "EXIT"]


# ── Promise 7 · trade management ────────────────────────────────────────────
def test_stop_exit_fills_at_stop():
    s = step(state(pos=open_call()), "10:00", 25480, 25485, 25440, 25445)
    (e,) = exits(s)
    assert e["action"] == "EXIT_STOP" and e["price"] == 25450.0
    assert e["pnl_inr"] == round(-50 * 2 * T.CFG["lot_size"])
    assert s["pos"] is None and s["wf_position"] == "IDLE"


def test_gap_through_stop_fills_at_open_not_stop():
    s = step(state(pos=open_call()), "10:00", 25430, 25435, 25420, 25425)
    assert exits(s)[0]["price"] == 25430          # worse than the 25450 stop: realistic


def test_target_exit_at_2R():
    s = step(state(pos=open_call()), "10:00", 25520, 25610, 25515, 25600)
    (e,) = exits(s)
    assert e["action"] == "EXIT_TARGET" and e["price"] == 25600.0 and e["r_multiple"] == 2.0


def test_stop_checked_before_target_when_both_touch():
    s = step(state(pos=open_call()), "10:00", 25500, 25610, 25440, 25500)
    assert exits(s)[0]["action"] == "EXIT_STOP"   # conservative assumption


def test_trailing_arms_after_1R_and_never_loosens():
    s = step(state(pos=open_call()), "10:00", 25500, 25560, 25495, 25555)   # +60 > 1R(50)
    p = s["pos"]
    assert p["trailing"] and p["stop"] == pytest.approx(max(25560 - 1.5 * EM, 25500))
    tightened = p["stop"]
    s = step(s, "10:01", 25555, 25556, 25540, 25545)                        # pullback, no new high
    assert s["pos"]["stop"] == tightened
    s = step(s, "10:02", 25545, 25546, 25520, 25525)                        # breaches trail
    assert exits(s)[0]["action"] == "EXIT_TRAIL" and exits(s)[0]["pnl_inr"] > 0


def test_eod_squareoff_at_1515():
    s = step(state(pos=open_call()), "15:15", 25505, 25510, 25500, 25507)
    (e,) = exits(s)
    assert e["action"] == "EXIT_EOD" and e["price"] == 25507


# ── Promise 9 · risk ────────────────────────────────────────────────────────
def approved(entry, stop, direction="CALL", tier="FULL", strategy="S1"):
    return dict(strategy=strategy, action=f"{strategy}_INITIAL_{direction}", direction=direction,
                entry=entry, stop=stop, tier=tier, score=85 if tier == "FULL" else 72, trap_ok=True,
                bar_time="10:00", trigger_time="10:01", checklist={})


def test_size_is_risk_based():
    s = step(state(wf_approved=approved(25500, 25450)), "10:00", 25500, 25505, 25495, 25500)
    per_lot = 50 * T.CFG["lot_size"]
    assert s["pos"]["lots"] == int(T.CFG["capital"] * T.CFG["risk_per_trade_pct"] / 100 // per_lot)


def test_half_tier_halves_risk():
    full = step(state(wf_approved=approved(25500, 25450)), "10:00", 25500, 25505, 25495, 25500)["pos"]["lots"]
    T.G_SPOT.clear()
    half = step(state(wf_approved=approved(25500, 25450, tier="HALF")), "10:00", 25500, 25505, 25495, 25500)["pos"]["lots"]
    assert half == full // 2 or half == max(1, full // 2)


def test_skip_when_one_lot_exceeds_max_risk():
    stop = 25500 - (T.CFG["capital"] * T.CFG["max_risk_per_trade_pct"] / 100 / T.CFG["lot_size"]) - 10
    s = step(state(wf_approved=approved(25500, stop)), "10:00", 25500, 25505, 25495, 25500)
    assert s["pos"] is None
    assert any(e["action"] == "S1_RISK_SKIP" for e in s["exec_log"])


def test_kill_switch_flattens_and_blocks_new_entries():
    big = open_call(entry=25500, stop=25000, lots=10)    # wide stop so the kill switch, not the stop, fires
    loss_pts = T.CFG["capital"] * T.CFG["max_daily_loss_pct"] / 100 / (10 * T.CFG["lot_size"]) + 1
    px = 25500 - loss_pts
    s = step(state(pos=big), "10:00", 25500, 25500, px, px)
    assert s["kill_switch"] and s["pos"] is None
    assert exits(s)[0]["action"] == "EXIT_KILL"
    s["wf_approved"] = approved(px, px - 20)
    s = step(s, "10:01", px, px + 5, px - 5, px)
    assert s["pos"] is None                              # nothing else today


def test_reversal_closes_opposite_position_first():
    s = state(pos=open_call(), wf_approved=approved(25520, 25560, direction="PUT", strategy="S3"))
    s = step(s, "13:00", 25515, 25525, 25510, 25520)
    assert exits(s)[0]["action"] == "EXIT_REVERSAL"
    assert s["pos"]["direction"] == "PUT"


def test_no_new_entries_after_1500():
    s = step(state(wf_approved=approved(25500, 25450)), "15:01", 25500, 25505, 25495, 25500)
    assert s["pos"] is None


# ── Promise 1 · breakout validation ─────────────────────────────────────────
def run_bo(bars):
    s = state(bar_index=0, bo_state={}, exec_log=[])
    for i, (t, o, h, l, c) in enumerate(bars):
        T.G_SPOT.append(bar(t, o, h, l, c))
        s.update(bar_index=i, bar_time=t)
        s = T.breakout_validator(s)
    return s


def test_pdl_break_then_reclaim_is_fake():
    pdl = T.G_PDC["prev_day_low"]
    s = run_bo([("09:20", pdl + 5, pdl + 6, pdl - 8, pdl - 4),
                ("09:21", pdl - 4, pdl + 15, pdl - 5, pdl + 12)])
    assert s["bo_state"]["PDL"]["status"] == "FAKE"
    assert [e["action"] for e in s["exec_log"]] == ["PDL_BREAK", "PDL_FAKE"]


def test_pdl_break_that_holds_is_real():
    pdl = T.G_PDC["prev_day_low"]
    bars = [(f"09:{20 + i}", pdl - 5 - i, pdl - 3 - i, pdl - 10 - i, pdl - 6 - i) for i in range(6)]
    s = run_bo(bars)
    assert s["bo_state"]["PDL"]["status"] == "REAL"


def test_pdh_fake_break_mirrors_pdl():
    pdh = T.G_PDC["prev_day_high"]
    s = run_bo([("11:00", pdh - 5, pdh + 8, pdh - 6, pdh + 4),
                ("11:01", pdh + 4, pdh + 5, pdh - 15, pdh - 12)])
    assert s["bo_state"]["PDH"]["status"] == "FAKE"


# ── Promise 3 · nothing below 70 trades ─────────────────────────────────────
def test_gate_blocks_low_scores_and_logs_them():
    T.G_SPOT.append(bar("10:00", 25500, 25505, 25495, 25500))
    T.G_FUT.extend([bar("09:59", 25510, 25512, 25508, 25510, 1000), bar("10:00", 25510, 25512, 25508, 25510, 1100)])
    cand = dict(strategy="S1", action="S1_INITIAL_CALL", direction="CALL", entry=25500, stop=25450,
                trap_ok=False, bar_time="10:00", trigger_time="10:01")
    s = T.conviction_gate(state(wf_candidates=[cand], regime="UNDEFINED", gate_last_skip={}))
    assert s["wf_approved"] is None
    assert s["gate_decisions"][0]["score"] < T.CFG["conviction_threshold"]
    assert s["exec_log"][0]["type"] == "SKIP"
