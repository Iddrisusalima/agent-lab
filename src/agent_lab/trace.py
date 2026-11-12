"""The step trace: one JSON object per line, and a sink that cannot break a run.

Requirement 11 asks for a machine-readable record of every step. Three decisions
shape this module, and each one is a direct reading of a requirement rather than
a taste preference.

**One line per completed step, whatever happened inside that step.** A step with
three tool calls collapses into the ``tool_calls`` array of a single record
(Req 11.7), which is what makes "line count equals step count" a property a test
can check (Req 11.8). ``json.dumps`` escapes embedded newlines, so no recorded
value -- a document chunk, a traceback, a note containing blank lines -- can
split a record across two lines (Req 11.9). The file is opened in append mode
with an explicit ``newline="\\n"``, so existing lines are never rewritten
(Req 11.1) and the terminator is exactly one byte on Windows as well as on
POSIX.

**The logger is a sink, never a control-flow participant.** Req 11.11 is
explicit: a failed append prints one warning naming the path and the reason,
discards that record, and the run ends with the same outcome and the same final
answer it would have reached had the append succeeded. So nothing here raises at
a caller -- not a serialisation failure, not a full disk, not a record the loop
filled in wrongly. Values are coerced into something JSON can hold instead of
being validated with an exception, which is the one place this module
deliberately departs from the fail-loudly style of ``tools/base.py`` and
``model_client.py``: those types guard developer invariants, while a broken
trace line must never cost an operator their answer.

**Redaction is applied to the serialised line.** One call, after serialisation,
catches the key wherever it reached -- a tool input, a result, assistant text, an
error message quoted inside any of them (Req 11.5, Req 1.11). The escaped form
of the key is redacted as well, because a key containing a quote or a backslash
appears in the serialised line escaped and would otherwise slip past a literal
search for it.

The run id is a ``uuid4`` hex generated once per run and carried on every record
of that run, which is what distinguishes the runs that share one trace file
(Req 11.3). ``run_outcome`` is set on the final record of a run only (Req 11.4);
the loop knows which step is final, and this module does not second-guess it.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import IO, Any, Iterable, Mapping, Protocol, runtime_checkable
from uuid import uuid4

from agent_lab.model_client import ToolCall, Usage
from agent_lab.redaction import redact
from agent_lab.tools.base import Approval, ToolResult

# Req 11.2, Req 11.6: tool inputs, tool results and assistant text are recorded
# to at most this many characters. Fixed at 4,000 by the requirement itself, so
# unlike the tool result limit (Req 5.12) it is not configurable: the trace is a
# debugging artefact with a documented shape, not an operator-tuned output.
TRACE_MAX_CHARS = 4000

# Req 11.2: the marker that states truncation occurred, appended wherever this
# module shortened a value. It carries the pre-truncation character count for the
# same reason the dispatcher's marker does -- it is the only way a reader can
# tell a complete value from its first 4,000 characters.
TRUNCATION_MARKER = "[trace truncated: {original} characters, first {limit} recorded]"

# How deep a nested tool-argument structure is walked before it is recorded as
# text. Arguments arrive as decoded JSON, so this bound is never reached in
# practice; it exists so that a pathological payload cannot turn the sink into a
# recursion error.
_MAX_DEPTH = 20

# The keys of a trace record, in the order the design fixes for them. Exported so
# a test can assert the shape against one list rather than fifteen literals.
RECORD_FIELDS: tuple[str, ...] = (
    "run_id",
    "step",
    "started_at",
    "duration_ms",
    "tool_name",
    "tool_input",
    "tool_result",
    "tool_calls",
    "assistant_text",
    "input_tokens",
    "output_tokens",
    "tokens_estimated",
    "cost_usd",
    "approval",
    "run_outcome",
)

# The keys of one entry of the ``tool_calls`` array, in order.
CALL_FIELDS: tuple[str, ...] = (
    "index",
    "name",
    "input",
    "result",
    "status",
    "truncated",
    "original_length",
    "approval",
)


# --------------------------------------------------------------------------- #
# Helpers the loop and the tests both use
# --------------------------------------------------------------------------- #


def new_run_id() -> str:
    """A fresh run identifier (Req 11.3).

    ``uuid4().hex`` rather than a timestamp or a counter: two runs started in the
    same millisecond, or two processes appending to one trace file, must still
    produce different identifiers, and only randomness gives that without
    coordination.
    """
    return uuid4().hex


def iso_utc_millis(moment: datetime | None = None) -> str:
    """``moment`` as an ISO 8601 UTC timestamp with millisecond precision.

    The exact shape Req 11.2 asks for, ``2026-03-04T18:22:41.507Z``. A naive
    datetime is read as UTC, an aware one is converted to it, and ``None`` means
    now. Sub-millisecond digits are dropped rather than rounded, so the recorded
    time never sits after the instant it describes.
    """
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    utc = moment.astimezone(timezone.utc)
    return f"{utc.strftime('%Y-%m-%dT%H:%M:%S')}.{utc.microsecond // 1000:03d}Z"


def elapsed_ms(start: float, end: float | None = None) -> int:
    """Whole milliseconds between two ``time.monotonic()`` readings (Req 11.2).

    The loop takes ``start`` immediately before the model call of a step and
    reads the clock again after the last tool of that step completes, which is
    precisely the interval Req 11.2 defines. ``time.monotonic`` and not
    ``time.time``: a clock adjustment mid-step must not produce a negative
    duration. The result is clamped at zero anyway, because a duration is never
    negative and a trace field should not be the place that argument is had.
    """
    if end is None:
        end = time.monotonic()
    return max(0, round((end - start) * 1000))


def truncate(text: str, limit: int = TRACE_MAX_CHARS) -> str:
    """``text`` cut to ``limit`` characters with the truncation marker appended.

    Returns ``text`` unchanged when it already fits, so the marker appears
    exactly where truncation was applied and nowhere else (Req 11.2, Req 11.6).
    """
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER.format(original=len(text), limit=limit)


# --------------------------------------------------------------------------- #
# Record types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TracedCall:
    """One tool request and its result, as the trace records it (Req 11.7).

    ``index``
        The position of this call in the step's dispatch order. Recorded
        explicitly so the order survives a reader that sorts or filters the
        array.
    ``name``
        The tool name as the model asked for it, including a name the registry
        does not know -- the trace records what happened, not what should have.
    ``input``
        The argument mapping; string values are truncated with a marker when the
        serialiser builds the line.
    ``result``
        The result content the model observed, truncated the same way.
    ``status``
        ``"success"`` or ``"error"``, so a reader can count failed calls without
        parsing result text.
    ``truncated`` / ``original_length``
        What the *dispatcher* reported (Req 5.12), not what this module did.
        Two different truncations, so two different places to look.
    ``approval``
        The confirmation decision for a write (Req 9.3, Req 9.4, Req 9.6). Kept
        per call as well as on the record, because a step containing two write
        requests took two decisions and a single field could only hold one.
    """

    index: int
    name: str | None
    input: Mapping[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str | None = None
    truncated: bool = False
    original_length: int | None = None
    approval: Approval | None = None

    @classmethod
    def from_dispatch(cls, index: int, call: ToolCall, result: ToolResult) -> TracedCall:
        """Build an entry from what the dispatcher returned for one call.

        The loop already holds both values, so this keeps the mapping between a
        ``ToolCall`` / ``ToolResult`` pair and its recorded shape in one place
        instead of inline at the call site.
        """
        return cls(
            index=index,
            name=getattr(call, "name", None),
            input=dict(getattr(call, "arguments", {}) or {}),
            result=getattr(result, "content", None),
            status=getattr(result, "status", None),
            truncated=bool(getattr(result, "truncated", False)),
            original_length=getattr(result, "original_length", None),
            approval=getattr(result, "approval", None),
        )


@dataclass(frozen=True)
class TraceRecord:
    """Everything the trace knows about one completed step (Req 11.2).

    ``run_id`` is absent on purpose: the logger stamps every record it writes
    with its own run id, so no caller can hand the file a record belonging to a
    different run (Req 11.3).

    ``calls`` is empty on a step that requested no tool; the serialiser then
    records a null ``tool_name``, an empty ``tool_input`` and the assistant text
    (Req 11.6). ``run_outcome`` is set by the loop on the final record of the run
    and left ``None`` on every other (Req 11.4).
    """

    step: int
    started_at: datetime | str | None = None
    duration_ms: int = 0
    calls: tuple[TracedCall, ...] = ()
    assistant_text: str | None = None
    usage: Usage | None = None
    cost_usd: float = 0.0
    approval: Approval | None = None
    run_outcome: Any = None

    @property
    def tool_calls(self) -> tuple[TracedCall, ...]:
        """``calls`` under the name the serialised field uses."""
        return tuple(self.calls)

    @property
    def step_approval(self) -> Approval | None:
        """The decision this step recorded, preferring an explicit one.

        A step's ``approval`` field mirrors the first call that asked for a
        decision rather than the first call outright: a step that searched and
        then wrote would otherwise report ``null`` at the top level and hide the
        operator's answer, while ``tool_calls`` keeps every decision in order.
        """
        if self.approval is not None:
            return self.approval
        for call in self.calls:
            if call.approval is not None:
                return call.approval
        return None


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def _json_safe(value: Any, depth: int = 0) -> Any:
    """``value`` reduced to something ``json.dumps`` can write, with truncation.

    Strings are truncated at :data:`TRACE_MAX_CHARS` with a marker. Non-finite
    floats become ``null``: ``json.loads`` accepts ``NaN`` and ``Infinity``, so a
    provider's argument payload can legitimately carry one, and writing it back
    out would produce a line that no strict JSON reader can parse (Req 11.9).
    Anything else without a JSON counterpart is recorded as its ``repr``, which
    keeps a surprising value visible instead of losing the whole record.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return truncate(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth >= _MAX_DEPTH:
        return truncate(repr(value))
    if isinstance(value, Mapping):
        return {
            (key if isinstance(key, str) else truncate(str(key))): _json_safe(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth + 1) for item in value]
    return truncate(repr(value))


def _text_or_none(value: Any) -> str | None:
    """A recorded text field: truncated text, or ``null`` when there is none."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return truncate(text) if text else None


def _outcome_text(outcome: Any) -> str | None:
    """The run outcome as the string the record carries (Req 11.4).

    Accepts the loop's enum member or a plain string, so this module does not
    import the agent loop and the loop is free to define ``RunOutcome`` however
    it likes.
    """
    if outcome is None:
        return None
    if isinstance(outcome, Enum):
        return str(outcome.value)
    return str(outcome)


def _whole(value: Any, minimum: int = 0) -> int:
    """``value`` as an integer at or above ``minimum``, never raising."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, number)


def _cost(value: Any) -> float:
    """The per-call cost rounded to 6 decimal places (Req 11.2).

    Six places is what Req 3.2 retains through the accounting and what Req 3.7's
    0.000001-dollar tolerance is stated against, so rounding here loses nothing
    the totals depend on. A value that is not a finite number is recorded as
    ``0.0`` -- the field stays a number, which is what every reader of the trace
    expects of it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, 6)


def record_object(record: TraceRecord, run_id: str) -> dict[str, Any]:
    """The trace record as a plain JSON-ready dictionary (Req 11.2).

    Pure, so the whole recorded shape can be asserted without a filesystem.

    ``tool_calls`` is authoritative: it holds one entry per request in dispatch
    order (Req 11.7). ``tool_name``, ``tool_input`` and ``tool_result`` mirror
    the first call for convenient reading, and are ``null`` / ``{}`` / ``null``
    on a step that requested no tool (Req 11.6).
    """
    calls = tuple(record.calls)
    first = calls[0] if calls else None
    usage = record.usage if isinstance(record.usage, Usage) else None

    return {
        "run_id": run_id,
        "step": _whole(record.step),
        "started_at": (
            record.started_at
            if isinstance(record.started_at, str)
            else iso_utc_millis(record.started_at)
        ),
        "duration_ms": _whole(record.duration_ms),
        "tool_name": first.name if first is not None else None,
        "tool_input": _json_safe(dict(first.input)) if first is not None else {},
        "tool_result": _text_or_none(first.result) if first is not None else None,
        "tool_calls": [
            {
                "index": _whole(call.index),
                "name": call.name,
                "input": _json_safe(dict(call.input or {})),
                "result": _text_or_none(call.result),
                "status": call.status,
                "truncated": bool(call.truncated),
                "original_length": (
                    None if call.original_length is None else _whole(call.original_length)
                ),
                "approval": call.approval,
            }
            for call in calls
        ],
        "assistant_text": _text_or_none(record.assistant_text),
        "input_tokens": _whole(usage.input_tokens) if usage is not None else 0,
        "output_tokens": _whole(usage.output_tokens) if usage is not None else 0,
        "tokens_estimated": bool(usage.estimated) if usage is not None else False,
        "cost_usd": _cost(record.cost_usd),
        "approval": record.step_approval,
        "run_outcome": _outcome_text(record.run_outcome),
    }


def serialise_record(record: TraceRecord, run_id: str, secret: str | None = None) -> str:
    """One trace record as a single redacted JSON line, without its terminator.

    ``ensure_ascii=False`` keeps non-ASCII document text readable in the file
    rather than escaping it into ``\\uXXXX`` sequences; ``json.dumps`` still
    escapes every newline, quote and backslash, so the result is one line that
    parses as one object (Req 11.1, Req 11.9).

    ``allow_nan=False`` is a guard rather than a working path: :func:`_json_safe`
    has already replaced any non-finite number, so this can only fire on a value
    that walk missed, and a raised error discards one record instead of writing a
    line no reader can parse.

    Redaction runs last, on the finished line, which is the one place that sees
    every recorded value at once (Req 11.5). Both the key and its JSON-escaped
    form are substituted: a key containing a quote or a backslash is written
    escaped, and a literal search for it would miss exactly the case that
    matters.
    """
    line = json.dumps(record_object(record, run_id), ensure_ascii=False, allow_nan=False)
    line = redact(line, secret)
    if secret:
        escaped = json.dumps(secret, ensure_ascii=False)[1:-1]
        if escaped != secret:
            line = redact(line, escaped)
    return line


# --------------------------------------------------------------------------- #
# The sink
# --------------------------------------------------------------------------- #


@runtime_checkable
class TraceSink(Protocol):
    """Where completed steps go.

    ``run_id`` and ``path`` are part of the protocol because the loop reports
    both in its ``RunResult`` and neither should be re-derived from the sink's
    concrete type.
    """

    run_id: str

    @property
    def path(self) -> Path: ...

    def append(self, record: TraceRecord) -> None: ...


class JsonlTraceLogger:
    """Appends one JSON line per completed step to a newline-delimited file.

    Constructed once per run, which is what makes "one run id on every record of
    this run" true by construction (Req 11.3): callers cannot pass a run id to
    :meth:`append`, so no record can be stamped with another run's identity.

    The file and any missing parent directory are created at construction, which
    is run start, and before any record is appended (Req 11.10). A failure there
    warns and is remembered rather than raised, for the same reason a failed
    append warns: this object must not be able to end a run.
    """

    __slots__ = ("run_id", "_path", "_secret", "_stream")

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str | None = None,
        api_key: str | None = None,
        stream: IO[str] | None = None,
    ) -> None:
        """
        :param path: the configured trace file (``Config.trace_file``).
        :param run_id: the run identifier; a fresh one is generated when absent.
        :param api_key: the provider key to redact from every line (Req 11.5).
        :param stream: where warnings go; ``sys.stderr`` when absent, looked up
            at write time so a test that replaces ``sys.stderr`` is still heard.
        """
        self.run_id = run_id or new_run_id()
        self._path = Path(path)
        self._secret = api_key
        self._stream = stream
        self._create()

    @property
    def path(self) -> Path:
        """The trace file this logger appends to."""
        return self._path

    def append(self, record: TraceRecord) -> None:
        """Append one record as a single line, or warn and discard it (Req 11.1).

        Serialisation and the write are guarded separately so the warning can say
        which half failed. Either way exactly one warning is printed, the record
        is dropped, and this returns normally: the loop goes on to the same
        outcome and the same final answer it would have reached (Req 11.11).
        """
        step = getattr(record, "step", "?")
        try:
            line = serialise_record(record, self.run_id, self._secret)
        except Exception as exc:  # noqa: BLE001 - Req 11.11 is the reason
            self._warn(step, f"the record could not be serialised ({_reason(exc)})")
            return

        try:
            # newline="\n" matters: the default translates "\n" to "\r\n" on
            # Windows, and Req 11.1 asks for exactly one newline character.
            # Append mode means the write can only add bytes to the end, so
            # every line already in the file -- including other runs' lines --
            # is left untouched.
            with open(self._path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - Req 11.11 is the reason
            self._warn(step, _reason(exc))

    def append_all(self, records: Iterable[TraceRecord]) -> None:
        """Append several records, each on its own line and each independently.

        Used by nothing in the run path; it exists so a replay or a test can hand
        over a sequence without looping at the call site. One failing record does
        not stop the rest, which is the same tolerance :meth:`append` has.
        """
        for record in records:
            self.append(record)

    # -- internals ------------------------------------------------------------

    def _create(self) -> None:
        """Create the parent directory and an empty trace file (Req 11.10).

        ``touch`` leaves an existing file exactly as it is, so a trace file
        holding earlier runs is never emptied by starting a new one (Req 11.1,
        Req 11.3).
        """
        try:
            parent = self._path.parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
            self._path.touch(exist_ok=True)
        except Exception as exc:  # noqa: BLE001 - a sink may not end a run
            self._warn(None, f"the file could not be created ({_reason(exc)})")

    def _warn(self, step: object, reason: str) -> None:
        """Print the one warning Req 11.11 asks for: the path and the reason.

        Redacted like every other output line (Req 1.11) -- the reason text comes
        from an operating-system error whose message quotes a configured path,
        and a path is as capable of containing a secret as anything else.
        """
        where = f"step {step}" if step not in (None, "?") else "the trace file"
        message = (
            f"warning: could not write {where} to the trace file "
            f"{self._path}: {reason}. The record was discarded; the run continues."
        )
        stream = self._stream if self._stream is not None else sys.stderr
        try:
            print(redact(message, self._secret), file=stream)
        except Exception:  # noqa: BLE001 - a closed stream must not end a run
            pass


def _reason(exc: BaseException) -> str:
    """An error as a short human reason, preferring the OS message."""
    detail = getattr(exc, "strerror", None) or str(exc) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {detail}"


__all__ = [
    "CALL_FIELDS",
    "RECORD_FIELDS",
    "TRACE_MAX_CHARS",
    "TRUNCATION_MARKER",
    "JsonlTraceLogger",
    "TraceRecord",
    "TraceSink",
    "TracedCall",
    "elapsed_ms",
    "iso_utc_millis",
    "new_run_id",
    "record_object",
    "serialise_record",
    "truncate",
]
