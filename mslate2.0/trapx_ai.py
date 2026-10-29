"""TRAP-X AI analyst — an LLM second opinion on the session (Promise 8).

Runs alongside the replay/live loop: briefs are answered on worker threads,
so a slow model never delays a bar. Opinions are advisory; the engine's own
rules and conviction gate still decide every trade.

Providers (TRAPX_AI_PROVIDER = auto | gemini | claude-cli | anthropic):
  gemini      Google Gemini REST API, key from GEMINI_API_KEY (env or .env)
  claude-cli  headless Claude Code (`claude -p`), uses the local login
  anthropic   Anthropic SDK, needs ANTHROPIC_API_KEY or an `ant` profile
`auto` tries them in that order, and drops from Gemini to Claude Code if
Gemini rejects the key or the model. TRAPX_AI=off disables the analyst.

Back-pressure: at most `max_workers` calls in flight; while busy only the
newest brief is kept, so a fast replay never builds a backlog.
"""
import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait as _wait
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

GEMINI_MODEL    = "gemini-flash-lite-latest"
CLAUDE_MODEL    = "claude-opus-5-5"
GEMINI_TIMEOUT  = 90
CLI_TIMEOUT     = 120

SYSTEM = """You are an independent intraday analyst giving a second opinion to a \
rule-based NIFTY 50 trap-detection engine. You receive the last 30 one-minute \
candles, previous-day levels (PDH/PDL/PDC), India VIX, the 1-minute expected move, \
TWAP, the engine's regime label, breakout validation states, an options-flow \
snapshot, the engine's current position and any setups it just scored.

Judge the next 15-30 minutes only. Say LEAN_BUY, LEAN_SELL or WAIT. Prefer WAIT \
when the evidence is mixed — a calibrated WAIT is more useful than a forced call. \
Confidence is your probability that the lean is right (0.5 = coin flip). In \
reasoning, cite the specific evidence (levels, candles, flow) in two or three \
sentences, and say whether you agree with the engine's position or setup."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict":    {"type": "string", "enum": ["LEAN_BUY", "LEAN_SELL", "WAIT"]},
        "confidence": {"type": "number"},
        "agrees_with_engine": {"type": "boolean"},
        "reasoning":  {"type": "string"},
    },
    "required": ["verdict", "confidence", "agrees_with_engine", "reasoning"],
    "additionalProperties": False,
}


class ProviderUnavailable(Exception):
    """The provider can't serve at all (bad key, retired model) — try the next one."""


def _env(name: str) -> Optional[str]:
    """Environment first, then the project's gitignored .env file."""
    if os.environ.get(name):
        return os.environ[name]
    path = os.path.join(_HERE, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k == name and v:
                return v.strip().strip('"').strip("'")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Providers — each returns a dict matching SCHEMA or raises
# ─────────────────────────────────────────────────────────────────────────────
class GeminiProvider:
    name = "gemini"

    def __init__(self):
        self.key = _env("GEMINI_API_KEY")
        if not self.key:
            raise ProviderUnavailable("no GEMINI_API_KEY")
        self.model = os.environ.get("TRAPX_AI_MODEL", GEMINI_MODEL)

    def ask(self, brief: Dict) -> Dict:
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(brief, separators=(",", ":"), default=str)}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA,
                                 "thinkingConfig": {"thinkingLevel": "low"}},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT))
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            if e.code in (400, 401, 403, 404) and ("API_KEY" in detail or "PERMISSION" in detail
                                                   or e.code in (401, 403, 404)):
                raise ProviderUnavailable(f"Gemini HTTP {e.code}: {detail}") from e
            raise RuntimeError(f"Gemini HTTP {e.code}") from e
        cand = (resp.get("candidates") or [{}])[0]
        if cand.get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
            raise RuntimeError(f"Gemini finish {cand.get('finishReason')}")
        text = cand["content"]["parts"][-1]["text"]
        out = json.loads(text)
        out["model"] = resp.get("modelVersion", self.model)
        return out


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(self):
        self.exe = shutil.which("claude")
        if not self.exe:
            raise ProviderUnavailable("`claude` CLI not on PATH")
        self.model = os.environ.get("TRAPX_AI_CLAUDE_MODEL", CLAUDE_MODEL)

    def ask(self, brief: Dict) -> Dict:
        cmd = [self.exe, "-p", "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
               "--tools", "", "--no-session-persistence", "--model", self.model,
               "--system-prompt", SYSTEM]
        proc = subprocess.run(cmd, input=json.dumps(brief, default=str), capture_output=True,
                              text=True, encoding="utf-8", timeout=CLI_TIMEOUT,
                              cwd=os.path.expanduser("~"))
        try:
            data = json.loads(proc.stdout)
        except ValueError as e:
            raise RuntimeError(f"claude CLI exit {proc.returncode}: {proc.stderr[:200]}") from e
        if data.get("is_error") or not data.get("structured_output"):
            raise RuntimeError(f"claude CLI: {data.get('subtype')} {str(data.get('result'))[:200]}")
        out = dict(data["structured_output"])
        out["model"] = next(iter(data.get("modelUsage") or {self.model: 0}))
        return out


class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        if not (_env("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE")):
            raise ProviderUnavailable("no Anthropic credentials")
        import anthropic
        self.a = anthropic
        self.client = anthropic.Anthropic(api_key=_env("ANTHROPIC_API_KEY"), max_retries=2, timeout=60.0)
        self.model = os.environ.get("TRAPX_AI_CLAUDE_MODEL", CLAUDE_MODEL)

    def ask(self, brief: Dict) -> Dict:
        a = self.a
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=4000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": json.dumps(brief, separators=(",", ":"), default=str)}],
            )
        except a.AuthenticationError as e:
            raise ProviderUnavailable("invalid Anthropic API key") from e
        except a.NotFoundError as e:
            raise ProviderUnavailable(f"model {self.model} not found") from e
        except a.RateLimitError as e:
            raise RuntimeError("rate limited") from e
        except a.APIStatusError as e:
            raise RuntimeError(f"API error {e.status_code}") from e
        except a.APIConnectionError as e:
            raise RuntimeError("network error") from e
        if resp.stop_reason == "refusal":
            return dict(verdict="WAIT", confidence=0.0, agrees_with_engine=False,
                        reasoning="Model declined to give an opinion on this brief.", model=resp.model)
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
        out["model"] = resp.model
        return out


PROVIDERS = {"gemini": GeminiProvider, "claude-cli": ClaudeCLIProvider, "anthropic": AnthropicProvider}


# ─────────────────────────────────────────────────────────────────────────────
# Analyst
# ─────────────────────────────────────────────────────────────────────────────
class AIAnalyst:
    def __init__(self, max_workers: int = 2):
        self.enabled, self.reason, self.provider = False, "", None
        self._done: List[Dict] = []
        self._lock = threading.Lock()
        self._inflight, self._futures, self._queued = 0, [], None
        self.dropped = 0
        self._max = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="trapx-ai")
        if os.environ.get("TRAPX_AI", "on").lower() in ("off", "0", "false"):
            self.reason = "disabled by TRAPX_AI=off"
            return
        choice = os.environ.get("TRAPX_AI_PROVIDER", "auto").lower()
        self._order = list(PROVIDERS) if choice == "auto" else [choice]
        self._next_provider()

    def _next_provider(self) -> bool:
        reasons = [self.reason] if self.reason else []
        while self._order:
            name = self._order.pop(0)
            try:
                self.provider = PROVIDERS[name]()
                self.enabled, self.reason = True, ""
                return True
            except ProviderUnavailable as e:
                reasons.append(f"{name}: {e}")
            except Exception as e:  # SDK missing etc.
                reasons.append(f"{name}: {type(e).__name__}")
        self.enabled, self.provider = False, None
        self.reason = "; ".join(reasons) or "no AI provider available"
        return False

    # ── public ───────────────────────────────────────────────────────────────
    def submit(self, bar_time: str, brief: Dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._inflight >= self._max:
                if self._queued:
                    self.dropped += 1
                self._queued = (bar_time, brief)        # newest brief wins
                return
            self._inflight += 1
        self._futures.append(self._pool.submit(self._run, bar_time, brief))

    def drain(self) -> List[Dict]:
        with self._lock:
            out, self._done = self._done, []
        return out

    def wait(self, timeout: float = 45.0) -> List[Dict]:
        """At the end of a replay: give in-flight calls `timeout` seconds,
        drop anything still queued, and return what arrived."""
        with self._lock:
            if self._queued:
                self.dropped += 1
            self._queued = None
        _wait(list(self._futures), timeout=timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)
        return self.drain()

    @property
    def status(self) -> Dict:
        return dict(enabled=self.enabled, provider=self.provider.name if self.provider else None,
                    model=getattr(self.provider, "model", None), reason=self.reason,
                    inflight=self._inflight, dropped=self.dropped)

    # ── worker ───────────────────────────────────────────────────────────────
    def _run(self, bar_time: str, brief: Dict) -> None:
        opinion = dict(bar_time=bar_time)
        while True:
            provider = self.provider
            if provider is None:
                opinion.update(verdict="WAIT", confidence=0.0, agrees_with_engine=False, error=True,
                               reasoning="AI analyst unavailable: " + self.reason)
                break
            try:
                data = provider.ask(brief)
                data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0))))
                data.setdefault("agrees_with_engine", False)
                opinion.update(data, provider=provider.name)
                break
            except ProviderUnavailable as e:
                with self._lock:
                    if self.provider is provider:      # first worker to notice switches provider
                        self.reason = f"{provider.name}: {e}"
                        self._next_provider()
                continue                               # retry this brief on the next provider
            except Exception as e:
                opinion.update(verdict="WAIT", confidence=0.0, agrees_with_engine=False, error=True,
                               provider=provider.name,
                               reasoning=f"{provider.name} error ({type(e).__name__}: {str(e)[:120]}) — defaulted to WAIT.")
                break
        with self._lock:
            self._done.append(opinion)
            self._inflight -= 1
            nxt, self._queued = self._queued, None
            if nxt:
                self._inflight += 1
        if nxt:
            try:
                self._futures.append(self._pool.submit(self._run, *nxt))
            except RuntimeError:                       # pool already shut down
                with self._lock:
                    self._inflight -= 1
