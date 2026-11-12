"""Driver: run the eval suite once under a token-per-minute pacer.

RETAINED, not deleted: ``evals/results.md`` cites this file by name as the
pacing harness both recorded measurements used, so it is now methodology
documentation rather than scratch (this supersedes the closing line below).

Not part of the product. It exists because the provider in use (Groq free tier,
OpenAI-compatible endpoint) enforces 8,000 tokens per MINUTE, while the shipped
adapter's mandated retry budget is 1s/2s/4s = 7s total -- far too short to
outwait a bucket that needs ~25s to refill. The result was an invalid 25% run
whose failures were rate limits, not agent behaviour.

Nothing under src/agent_lab/ is modified. Two documented seams are used:

* ``run_suite(..., model_client_factory=...)`` -- one client per task.
* ``OpenAIClient(..., max_retries=..., retry_delays=...)`` -- both are
  constructor parameters, so a generous backstop schedule needs no edit.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Any, Mapping, Sequence

from agent_lab.config import Config, load_config
from agent_lab.evals.runner import run_suite
from agent_lab.evals.schema import EvalTask
from agent_lab.model_client import Message, ModelReply
from agent_lab.providers.openai_client import OpenAIClient

# --------------------------------------------------------------------------- #
# Pacing policy
# --------------------------------------------------------------------------- #

WINDOW_SECONDS = 60.0
#: Headroom below the measured x-ratelimit-limit-tokens of 8000 per minute.
TOKEN_BUDGET = 7_000
#: What one upcoming call is assumed to cost before it has been made. Raised to
#: match reality if a call turns out bigger than this.
INITIAL_RESERVE = 4_000
RESERVE_CEILING = 6_000

#: The backstop: a 429 that still slips past the pacer is outwaited instead of
#: burning 7 seconds and giving up.
BACKSTOP_RETRIES = 6
BACKSTOP_DELAYS = (20.0, 25.0, 30.0, 35.0, 40.0, 60.0)


class TokenPacer:
    """Trailing-60-second token accounting, shared by every task of the suite.

    The bucket is per-minute and per-credential, so it does not reset between
    tasks; one pacer therefore spans the whole suite.
    """

    def __init__(self) -> None:
        self._window: deque[tuple[float, int]] = deque()
        self._reserve = INITIAL_RESERVE
        self.calls = 0
        self.total_tokens = 0
        self.total_slept = 0.0

    def _prune(self, now: float) -> None:
        while self._window and now - self._window[0][0] >= WINDOW_SECONDS:
            self._window.popleft()

    def _trailing(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens in self._window)

    def wait_turn(self) -> None:
        """Sleep until the next call fits inside the trailing-minute budget."""
        while True:
            now = time.monotonic()
            used = self._trailing(now)
            if used + self._reserve <= TOKEN_BUDGET or not self._window:
                return
            oldest = self._window[0][0]
            wait = max(1.0, WINDOW_SECONDS - (now - oldest) + 0.5)
            _log(
                f"pacing: {used} tok in trailing 60s + {self._reserve} reserved "
                f"> {TOKEN_BUDGET}; sleeping {wait:.1f}s"
            )
            time.sleep(wait)
            self.total_slept += wait

    def record(self, tokens: int) -> None:
        self._window.append((time.monotonic(), tokens))
        self.calls += 1
        self.total_tokens += tokens
        if tokens > self._reserve:
            self._reserve = min(RESERVE_CEILING, tokens)


class PacedClient:
    """``ModelClient`` wrapper that paces, then delegates unchanged.

    Satisfies the protocol with exactly ``model_id`` and ``complete``; the
    ``ModelReply`` is returned as the adapter produced it.
    """

    def __init__(self, inner: OpenAIClient, pacer: TokenPacer, label: str) -> None:
        self._inner = inner
        self._pacer = pacer
        self._label = label
        self.model_id = inner.model_id

    def complete(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]] = (),
    ) -> ModelReply:
        self._pacer.wait_turn()
        started = time.monotonic()
        reply = self._inner.complete(system_prompt, messages, tool_schemas)
        tokens = reply.usage.input_tokens + reply.usage.output_tokens
        self._pacer.record(tokens)
        _log(
            f"call {self._pacer.calls} [{self._label}] "
            f"in={reply.usage.input_tokens} out={reply.usage.output_tokens} "
            f"{'estimated' if reply.usage.estimated else 'reported'} "
            f"{time.monotonic() - started:.1f}s  "
            f"suite_tokens={self._pacer.total_tokens}  "
            f"tool_calls={len(reply.tool_calls)}"
        )
        return reply


def _log(message: str) -> None:
    print(f"    . {message}", flush=True)


def _build_sdk_client(api_key: str, base_url: str | None) -> Any:
    from openai import OpenAI

    # max_retries=0: the one retry policy in the process stays the adapter's.
    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def main() -> int:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not in this process environment.", file=sys.stderr)
        return 2

    # Req 15.2: the suite must not be able to block on a prompt.
    config = load_config(os.environ, interactive=False, probe=False)
    print(
        f"model={config.model_id}  base_url={base_url or '<sdk default>'}  "
        f"budget={TOKEN_BUDGET} tok/60s  backstop={BACKSTOP_DELAYS}",
        flush=True,
    )

    pacer = TokenPacer()

    def factory(task: EvalTask, task_config: Config) -> PacedClient:
        inner = OpenAIClient(
            task_config.model_id,
            client=_build_sdk_client(api_key, base_url),
            max_retries=BACKSTOP_RETRIES,
            retry_delays=BACKSTOP_DELAYS,
        )
        print(f"\n== task {task.id} (max_steps={task.max_steps})", flush=True)
        return PacedClient(inner, pacer, task.id)

    started = time.monotonic()
    report = run_suite(
        config,
        model_client_factory=factory,
        write_report=True,
        progress=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.monotonic() - started

    print("\n" + "=" * 72, flush=True)
    print(
        f"pass {report.passed_count}/{report.suite_size}  "
        f"({report.pass_rate:.1f}%)  mean_steps={report.mean_steps:.2f}  "
        f"total_cost=${report.total_cost_usd:.4f}",
        flush=True,
    )
    print(
        f"provider calls={pacer.calls}  tokens={pacer.total_tokens}  "
        f"paced sleep={pacer.total_slept:.0f}s  wall={elapsed / 60:.1f} min",
        flush=True,
    )
    for result in report.results:
        print(
            f"  {result.task_id:38s} {result.outcome.value:8s} "
            f"steps={result.steps} run={getattr(result.run_outcome, 'value', '-')} "
            f"detail={result.detail}",
            flush=True,
        )
    print(f"report written to: {report.report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
