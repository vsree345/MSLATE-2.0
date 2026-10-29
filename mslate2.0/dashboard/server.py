"""TRAP-X dashboard server.

  POST /api/simulate   start a replay   {"limit": int|null, "speed": "1x"|"4x"|"max"}
  GET  /api/stream     Server-Sent Events: log lines, per-bar records, done summary
  GET  /api/results    last completed (or in-progress) run as JSON
  GET  /               dashboard UI

Run:  venv\\Scripts\\python dashboard\\server.py   →  http://localhost:8000
"""
import asyncio
import json
import os
import sys
import threading
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import TrapX_agent as engine  # noqa: E402

SPEEDS = {"1x": 0.12, "4x": 0.03, "max": 0.0}   # seconds slept per bar


# ─────────────────────────────────────────────────────────────────────────────
# stdout routing: only the replay thread's print() output goes to the run log
# ─────────────────────────────────────────────────────────────────────────────
class _ThreadRouter:
    def __init__(self, real):
        self._real = real
        self.targets = {}            # thread ident -> callable(str)

    def write(self, data):
        sink = self.targets.get(threading.get_ident())
        if sink:
            sink(data)
        else:
            self._real.write(data)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


_router = _ThreadRouter(sys.stdout)
sys.stdout = _router


class Run:
    def __init__(self, limit, speed):
        self.id = f"run-{int(time.time() * 1000)}"
        self.limit, self.speed = limit, speed
        self.events = []             # append-only list of (kind, payload)
        self.bars = []
        self.summary = None
        self.done = False
        self.error = None
        self._buf = ""

    def log(self, data: str):
        self._buf += data
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            if ln.strip():
                self.events.append(("log", ln))

    def bar(self, record):
        self.bars.append(record)
        self.events.append(("bar", record))


_lock = threading.Lock()
_current: Run = None


def _summary(result) -> dict:
    final, pdc = result["final"], result["pdc"]
    return dict(
        bars=len(result["bars"]),
        regime=final.get("regime"), regime_confidence=final.get("regime_confidence"),
        intraday_high=final.get("intraday_high"), intraday_low=final.get("intraday_low"),
        twap_crossings=final.get("vwap_crossings"),
        volume_nodes=len(final.get("volume_nodes", [])),
        expected_move=final.get("expected_move_1min"),
        trade_log=final.get("trade_log", []), wf_trade_log=final.get("wf_trade_log", []),
        wf_position=final.get("wf_position"), wf_entry=final.get("wf_entry"),
        fib_50=final.get("wf_fib_50"), pdc=pdc,
        exec_log=final.get("exec_log", []), realized_pnl=final.get("realized_pnl", 0.0),
        equity=final.get("equity"), max_drawdown_pct=final.get("max_drawdown_pct", 0.0),
        kill_switch=final.get("kill_switch", False), ai_opinions=final.get("ai_opinions", []),
        late_ai=result.get("late_ai", []), latency=result.get("latency", {}),
        ai_status=result.get("ai_status", {}),
        cfg=dict(capital=engine.CFG["capital"], lot_size=engine.CFG["lot_size"],
                 max_daily_loss_pct=engine.CFG["max_daily_loss_pct"],
                 conviction_threshold=engine.CFG["conviction_threshold"]),
    )


def _worker(run: Run):
    _router.targets[threading.get_ident()] = run.log
    try:
        result = engine.run_replay(limit=run.limit, on_bar=run.bar,
                                   delay=SPEEDS.get(run.speed, 0.03))
        run.summary = _summary(result)
        engine.print_session_summary(result["final"])
    except Exception as e:  # surface engine failures to the UI
        run.error = f"{type(e).__name__}: {e}"
        run.events.append(("log", f"ERROR {run.error}"))
    finally:
        _router.targets.pop(threading.get_ident(), None)
        run.done = True
        run.events.append(("done", {"summary": run.summary, "error": run.error}))
        _lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="TRAP-X")


class SimulateRequest(BaseModel):
    limit: int | None = None
    speed: str = "4x"


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    global _current
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "A replay is already running")
    _current = Run(req.limit, req.speed)
    threading.Thread(target=_worker, args=(_current,), daemon=True).start()
    return {"run_id": _current.id}


@app.get("/api/stream")
async def stream(cursor: int = 0, last_event_id: str | None = Header(None)):
    run = _current
    if run is None:
        raise HTTPException(404, "No replay has been started")
    if last_event_id and last_event_id.isdigit():   # EventSource auto-reconnect
        cursor = int(last_event_id)

    async def gen():
        i = cursor
        yield f"event: hello\ndata: {json.dumps({'run_id': run.id})}\n\n"
        while True:
            while i < len(run.events):
                kind, payload = run.events[i]
                i += 1
                yield f"id: {i}\nevent: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"
                if kind == "done":
                    return
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/results")
def results():
    run = _current
    if run is None:
        return {"status": "empty"}
    return json.loads(json.dumps(dict(
        status="done" if run.done else "running", run_id=run.id,
        bars=run.bars, summary=run.summary, error=run.error,
        log=[p for k, p in run.events if k == "log"][-2000:],
    ), default=str))


app.mount("/static", StaticFiles(directory=_HERE), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_HERE, "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("  TRAP-X dashboard → http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
