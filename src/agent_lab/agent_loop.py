"""The agent loop: think, act, observe, bounded twice (Req 4).

One model call plus the tool executions that call asked for is a *step*
(glossary, Req 4.2). :meth:`AgentLoop.run` performs steps until one of four
things is true, and returns a single :class:`RunResult` saying which (Req 4.14).

**It imports no adapter and no provider SDK** (Req 1.1). Every collaborator
arrives through the constructor as a protocol: a ``ModelClient``, a dispatcher,
a registry, a ``CostTracker``, a ``TraceSink``, a ``Renderer`` and the system
prompt text. That is what lets the whole of Requirement 4 be tested offline
against a scripted fake, and it is why this module can be imported without
importing ``cli.py``.

**The step order is the design.** Three requirements are statements about
*where* a check sits rather than about what it does, so the order below is
load-bearing and is written out once, in :meth:`AgentLoop.run`, in exactly this
sequence::

    1. completed_steps >= MAX_STEPS        -> end STEP_LIMIT   (Req 4.6, 4.7)
    2. cost_tracker.total_usd >= COST_CAP  -> end COST_CAP     (Req 4.8, 4.9)
    3. step_no = completed_steps + 1                           (Req 4.4)
    4. reply = model_client.complete(...)                      (Req 4.5)
         typed error                       -> end MODEL_ERROR  (Req 4.13)
    5. cost_tracker.record(reply.usage, model_id)              (Req 3.1, 3.2)
    6. append the assistant message                            (Req 2.13)
    7. no tool calls: text        -> end FINAL_ANSWER          (Req 4.3)
                      no text     -> end MODEL_ERROR           (Req 4.13)
    8. more than 8 tool calls     -> end MODEL_ERROR           (Req 4.13)
    9. dispatch each call in reply order, appending each result (Req 4.2, 5.11)
   10. write the step's trace line                              (Req 11.1)
   11. completed_steps += 1

Both bounds are checked *before* the model call of the step. That placement is
the whole mechanism: the step count cannot exceed ``MAX_STEPS`` because the
check precedes the call that would make step 9 happen (Property 1), and the
total cost cannot exceed ``COST_CAP`` by more than one call because the call
that crossed the cap was already in flight when the cap was crossed
(Property 2). Checking after the call would make both bounds advisory.

Step 10 runs once per step whatever happened inside it -- every tool call
failed, the model asked for nothing, the model's text was empty -- which is what
keeps the trace line count equal to the step count (Req 11.8, Property 11).

**Who prints what.** The loop narrates steps and tool calls through the
renderer, and prints the abnormal-termination message of Req 4.7, Req 4.8,
Req 4.12 and Req 4.13 through ``renderer.error`` to stderr. It does *not* print
the final answer or the run summary: those are the CLI's, under Req 13.4 and
Req 3.9, and the CLI has the cost tracker to read. The same message the loop
printed is also carried on ``RunResult.error_message`` so the eval runner can
read it without parsing a terminal; a CLI that printed that field again would
double the line.

**A failed model call performs no step.** When ``complete`` raises its typed
error there is no reply, no usage and no result, so nothing is recorded: the
conversation keeps exactly the messages it had, the cost tracker is untouched,
and no trace line is written for the attempt (Req 2.7, Req 4.13). The run ends
with the model-error outcome and ``steps`` equal to the number of steps that
did complete, which keeps the cost tracker's step count, the trace line count
and ``RunResult.steps`` equal to each other (Req 3.7, Req 11.8).

**Why the trace line of a step is written one beat late.** Req 11.4 puts the
run outcome on the trace record of the *final* step, and Req 11.8 wants exactly
one line per step. For a run that ends at a bound check, the outcome becomes
known only *after* the final step has completed -- so a record appended the
instant its step finished could never carry it, and appending an extra record
for the step that never ran would break the line count. This module therefore
holds the most recently completed step's record in a one-slot buffer: it is
appended when the next step completes, or, at the end of the run, stamped with
the outcome and appended then. Records are still written once each, in
ascending step order, and the final one carries the outcome on all four paths.
An unexpected exception out of :meth:`AgentLoop.run` flushes the held record in
a ``finally``, so a crash costs the outcome stamp rather than the line.

**Grounding, and the one place this module departs from the design's
pseudocode.** Req 12.8 requires text from ``search_docs`` or ``read_notes`` to
enter the conversation enclosed in the untrusted-data delimiters with any
forged delimiter inside it removed. The design's pseudocode achieves that with
an unconditional ``grounding.wrap`` on every result from those two tools, but
both tools now delimit their own document text -- ``search_docs`` per *entry*,
so that each chunk's source name sits immediately before its own quoted block
(Req 8.2, Req 8.4), and ``read_notes`` around the note body only, keeping its
own header and omission marker outside the block (Req 10.1). Since ``wrap``
sanitises what it wraps, wrapping an already-delimited result would replace
every inner marker with a removal notice and destroy precisely the boundaries
Req 8.4 asks for. :meth:`AgentLoop._ground` therefore *ensures* the invariant
instead of blindly re-establishing it:

* a successful result from one of those tools that already carries the
  canonical delimiters is passed through unchanged -- it is already enclosed
  and already sanitised;
* a successful result that carries none is wrapped, so a tool result that is
  document text still cannot reach the model unenclosed;
* an error result is sanitised but not wrapped. Error text is this
  application's own words, written to be acted on so the model can correct its
  call (Req 5.11); enclosing it in delimiters would tell the model -- via the
  system prompt's data-not-instructions rule (Req 12.2) -- to report it as
  something a document said rather than fix the call. Document text never
  appears in an error result from either tool: a failed read reports the reason
  and the path, and a rejected query performs no retrieval. Sanitising still
  removes any delimiter echoed back from model-supplied input, so no error
  result can forge a data boundary either.

The invariant that holds in every branch is the one Req 12.8 exists for: no
text entering the conversation from either document tool contains a forged
delimiter, and all document text entering the conversation is enclosed.

**The optional memory file.** Req 18.4 lets a configured file be read into the
initial conversation state. It is off unless a memory source is passed to the
constructor, it enters the conversation through the same ``grounding.wrap`` as
document text, and a configured path that cannot be read warns and the run
continues (Req 18.8). See :meth:`AgentLoop._memory_message`.

**Nothing in a tool result can change a bound.** ``MAX_STEPS``, ``COST_CAP``,
the registry and the gate mode are fixed at construction and have no setter
reachable from result text, which is Req 12.9 held by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from agent_lab import exit_codes
from agent_lab.cost import CostTracker
from agent_lab.grounding import BEGIN, END, sanitize, wrap
from agent_lab.model_client import (
    Message,
    ModelClient,
    ModelClientError,
    ModelReply,
    ToolCall,
)
from agent_lab.render import Renderer
from agent_lab.tools.base import ToolResult
from agent_lab.tools.dispatcher import ToolDispatcher
from agent_lab.tools.registry import ToolRegistry
from agent_lab.trace import (
    TracedCall,
    TraceRecord,
    TraceSink,
    elapsed_ms,
    iso_utc_millis,
    new_run_id,
)

# --------------------------------------------------------------------------- #
# The two bounds (Req 4.11)
# --------------------------------------------------------------------------- #

# Req 4.11: both constants live in this module, each with the reason for its
# value, and both apply to every run. No CLI flag overrides either one per run;
# `cli.py` passes the configured values of Req 1.6/Req 1.7, which default to
# exactly these numbers, and `.env.example` documents them as such.

MAX_STEPS = 8      # 8 is enough for search -> read -> compute -> answer with slack;
                   # beyond that a stuck model burns money without converging.
COST_CAP = 0.25    # a quarter of a dollar per task keeps a 12-task suite near $3
                   # and makes an accidental loop cheap.

# Req 4.2, Req 4.13: at most 8 tool requests may be executed in one step. A
# reply asking for more is a model error rather than a truncated dispatch: the
# model is working from a plan the loop cannot carry out, and executing the
# first 8 of 20 calls would feed it an observation of a plan that half ran.
MAX_TOOL_CALLS_PER_STEP = 8

# Req 4.1, Req 4.12: the accepted task length after trimming leading and
# trailing whitespace.
MIN_TASK_CHARS = 1
MAX_TASK_CHARS = 4_000

# Req 18.4: how many characters of the optional memory file are read. The value
# is the documented default of TOOL_RESULT_MAX_CHARS, duplicated here as a
# number rather than imported, because this module reads no configuration (it
# takes every bound as a constructor argument, Req 1.1). `cli.py` passes
# `config.tool_result_max_chars`, so an operator who changes that variable
# changes this limit too; this constant only decides what a direct construction
# that names no limit gets.
MEMORY_READ_MAX_CHARS = 4_000

# Req 18.8: the environment variable named in the failure message, so an
# operator reading it knows which setting to correct.
MEMORY_FILE_VAR = "MEMORY_FILE"

# What the failure message calls a memory source injected as a callable with no
# path beside it. Only a test does that; the CLI always passes the path.
_MEMORY_SOURCE_LABEL = "(injected memory source)"

# Req 12.8: the two tools whose results carry document text, and whose results
# therefore enter the conversation enclosed in the untrusted-data delimiters.
# `save_note` and `calculator` return this application's own words, so there is
# nothing untrusted to enclose.
GROUNDED_TOOLS = frozenset({"search_docs", "read_notes"})


# --------------------------------------------------------------------------- #
# Outcomes (Req 4.14) and their exit codes (Req 4.10, Req 4.12)
# --------------------------------------------------------------------------- #


class RunOutcome(str, Enum):
    """The terminal classification of one run. Exactly one per run (Req 4.14).

    The first four are the Run_Outcome values of the glossary.
    :attr:`INVALID_TASK` is not one of them and is deliberately distinct: Req
    4.12 says that a task string which is empty, whitespace-only or over-long
    starts *no run at all*, so there is no run to classify -- but
    :meth:`AgentLoop.run` still has to return something, and returning a
    ``RunResult`` carrying this value is better than raising an exception the
    CLI and the eval runner would each have to catch.

    A ``str`` subclass rather than ``enum.StrEnum`` because the project supports
    Python 3.10 and ``StrEnum`` arrived in 3.11. The behaviour that matters --
    ``RunOutcome.FINAL_ANSWER == "final_answer"``, and a plain string in JSON --
    is the same either way.
    """

    FINAL_ANSWER = "final_answer"
    STEP_LIMIT = "step_limit_reached"
    COST_CAP = "cost_cap_reached"
    MODEL_ERROR = "model_error"
    INVALID_TASK = "invalid_task"

    def __str__(self) -> str:  # pragma: no cover - convenience for messages
        return self.value


# Req 4.10, Req 4.12: one distinct, documented exit code per outcome. The codes
# themselves live in `exit_codes.py` and are listed in the README; this mapping
# is the single place that joins an outcome to its code, so the CLI does not
# re-derive it with a chain of comparisons.
EXIT_CODES: dict[RunOutcome, int] = {
    RunOutcome.FINAL_ANSWER: exit_codes.FINAL_ANSWER,
    RunOutcome.STEP_LIMIT: exit_codes.STEP_LIMIT,
    RunOutcome.COST_CAP: exit_codes.COST_CAP,
    RunOutcome.MODEL_ERROR: exit_codes.MODEL_ERROR,
    RunOutcome.INVALID_TASK: exit_codes.INVALID_TASK,
}


def exit_code_for(outcome: RunOutcome | str) -> int:
    """The documented process exit code for ``outcome`` (Req 4.10, Req 4.12).

    An unrecognised value maps to the model-error code rather than to success:
    whatever produced it, the run did not demonstrably answer the question, and
    a caller's ``if exit_code == 0`` must not read it as one.
    """
    try:
        return EXIT_CODES[RunOutcome(outcome)]
    except (KeyError, ValueError):
        return exit_codes.MODEL_ERROR


# --------------------------------------------------------------------------- #
# The run result
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    """Everything one run produced: the single value the CLI and evals consume.

    The eval checks are pure functions of this object rather than of live
    process state, which is what lets each check type be unit-tested from a
    literal fixture (design decision 12, Req 16.13).

    ``run_id``
        The trace run identifier, so a report can point at the lines of this
        run in a file holding many (Req 11.3).
    ``outcome``
        Exactly one :class:`RunOutcome` (Req 4.14).
    ``final_answer``
        The assistant text of the reply that ended the run, and ``None`` on
        every other outcome -- a step-limited or cost-capped run produced no
        final answer, and Req 4.7 requires saying so rather than offering the
        last thing the model happened to write.
    ``steps``
        Completed steps, i.e. model calls that returned a reply. Equal to the
        cost tracker's step count and to the number of trace lines this run
        appended (Req 3.7, Req 11.8), and at most ``max_steps`` (Req 4.6).
    ``input_tokens`` / ``output_tokens`` / ``cost_usd``
        The run totals as the cost tracker reported them at the end (Req 3.9).
    ``tools_called``
        Every tool name the loop dispatched, in execution order, with
        duplicates kept -- a step that asked for the same tool twice appears
        twice (Req 4.2). The name is the one the model requested, including a
        name the registry does not know, because the field records what was
        attempted.
    ``notes_text_at_end``
        The notes file's text when the run ended, as the file holds it: a
        snapshot for the ``notes_contains`` / ``notes_not_contains`` checks
        (Req 14.3). Empty when no notes path was injected or the file does not
        exist.
    ``trace_path``
        The configured trace file, for a report that names where to look.
    ``error_message``
        Why the run ended abnormally, for the step-limit, cost-cap, model-error
        and invalid-task outcomes; ``None`` for a final answer. The loop has
        already printed this through ``renderer.error``, so a caller should
        report it, not reprint it.
    ``warnings``
        The cost tracker's warning lines (Req 3.3, Req 3.4) plus any the loop
        itself accumulated, in that order.
    ``messages``
        The conversation as it stood when the run ended. Carried because
        Req 4.13 requires the conversation state to be *retained* when a model
        call fails, which is only checkable if it survives the return.
    """

    run_id: str
    outcome: RunOutcome
    final_answer: str | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    notes_text_at_end: str = ""
    trace_path: str = ""
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    messages: tuple[Message, ...] = ()

    @property
    def total_tokens(self) -> int:
        """Input plus output, the fourth token figure of the summary (Req 3.9)."""
        return self.input_tokens + self.output_tokens

    @property
    def exit_code(self) -> int:
        """The documented exit code for this run's outcome (Req 4.10, Req 4.12)."""
        return exit_code_for(self.outcome)

    @property
    def ended_with_final_answer(self) -> bool:
        """Whether the run produced a final answer (an eval check reads this)."""
        return self.outcome is RunOutcome.FINAL_ANSWER


# --------------------------------------------------------------------------- #
# Task validation (Req 4.1, Req 4.12)
# --------------------------------------------------------------------------- #


def validate_task(task: Any) -> tuple[str, str | None]:
    """Return ``(trimmed_task, error_message)`` for a submitted task string.

    ``error_message`` is ``None`` when the trimmed string is between
    :data:`MIN_TASK_CHARS` and :data:`MAX_TASK_CHARS` characters (Req 4.1), and
    otherwise states that the task is empty or exceeds the 4,000-character
    limit, which is what Req 4.12 asks the message to say.

    A module-level pure function, so the rule can be tested without a loop, a
    model or a trace file, and so the CLI can apply the identical check before
    it builds anything if it wants to.
    """
    if not isinstance(task, str):
        return "", (
            f"The task string is invalid: a task must be text between "
            f"{MIN_TASK_CHARS} and {MAX_TASK_CHARS} characters after trimming, but a "
            f"value of type {type(task).__name__} was received. No run was started."
        )

    trimmed = task.strip()
    if len(trimmed) < MIN_TASK_CHARS:
        return trimmed, (
            "The task string is empty: a task must contain at least one "
            "non-whitespace character and at most "
            f"{MAX_TASK_CHARS} characters after trimming. No run was started."
        )
    if len(trimmed) > MAX_TASK_CHARS:
        return trimmed, (
            f"The task string is {len(trimmed)} characters after trimming, which "
            f"exceeds the {MAX_TASK_CHARS}-character limit. Shorten the task. "
            f"No run was started."
        )
    return trimmed, None


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


class AgentLoop:
    """Runs one bounded think-act-observe loop per :meth:`run` call (Req 4).

    :param model_client: anything satisfying
        :class:`~agent_lab.model_client.ModelClient` -- the real adapter in
        production, a scripted fake in the tests (Req 2.1, Req 16.8).
    :param dispatcher: the single path into tool code. Called for its
        ``dispatch(call) -> ToolResult``, which never raises (Req 5.9), so no
        dispatch here is wrapped in a ``try``.
    :param registry: the registered tools. ``schemas()`` is read afresh for
        every model call, so exactly the registered declarations are offered on
        every call of every step (Req 4.5).
    :param cost_tracker: the run's accounting. ``record`` is called once per
        completed call and ``total_usd`` is read before each call (Req 3.1,
        Req 4.8).
    :param trace_sink: where completed steps go (Req 11.1). Optional: pass
        ``None`` to run without tracing, which the eval runner's dry paths and
        some unit tests want.
    :param renderer: step narration and the abnormal-termination messages
        (Req 13.1, Req 13.2, Req 13.8). Optional; silent when absent.
    :param system_prompt: the prompt text, loaded from its versioned resource
        by ``prompts.py`` and injected here, so this module holds no prompt
        text of its own (Req 12.10).
    :param max_steps: the step bound, defaulting to :data:`MAX_STEPS`. Supplied
        by configuration (Req 1.6, Req 1.7); never by a per-run CLI flag
        (Req 4.11).
    :param cost_cap: the spend bound in dollars, defaulting to
        :data:`COST_CAP`, under the same rule.
    :param notes_file: the configured notes path, read once at the end of a run
        for ``RunResult.notes_text_at_end`` (Req 14.3).
    :param notes_reader: an alternative to ``notes_file`` -- a callable
        returning the notes text. Lets the eval runner supply a snapshot it
        already holds.
    :param memory_file: the optional memory file of Req 18.4, read at the start
        of every run. ``None`` -- the default -- is the disabled state, in which
        nothing below runs at all (Req 18.6, Req 18.7).
    :param memory_reader: an alternative source for the same text -- a callable
        returning it -- in the style of ``notes_reader``, so a test needs no
        real file. Pass ``memory_file`` alongside it to have the failure message
        name a path.
    :param memory_max_chars: the read limit of Req 18.4, defaulting to
        :data:`MEMORY_READ_MAX_CHARS`. The CLI passes the configured
        tool-result maximum.

    An instance is cheap and is built per run in practice, because the cost
    tracker and the trace sink it is given are themselves per-run. Nothing is
    stored on the instance during a run: all per-run state is local to
    :meth:`run`, so a second call on the same instance starts clean.
    """

    def __init__(
        self,
        model_client: ModelClient,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
        cost_tracker: CostTracker,
        trace_sink: TraceSink | None = None,
        renderer: Renderer | None = None,
        system_prompt: str = "",
        max_steps: int = MAX_STEPS,
        cost_cap: float = COST_CAP,
        *,
        notes_file: Path | str | None = None,
        notes_reader: Callable[[], str] | None = None,
        memory_file: Path | str | None = None,
        memory_reader: Callable[[], str] | None = None,
        memory_max_chars: int = MEMORY_READ_MAX_CHARS,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            # Req 4.6 requires at least one step, so a bound below 1 describes a
            # run that cannot exist. Configuration already refuses anything
            # outside 1-50 (Req 1.9); this catches a direct construction.
            raise ValueError(f"max_steps must be an integer of 1 or more, received {max_steps!r}")
        if isinstance(cost_cap, bool) or not isinstance(cost_cap, (int, float)) or cost_cap <= 0:
            raise ValueError(f"cost_cap must be a positive number of dollars, received {cost_cap!r}")
        if not isinstance(system_prompt, str):
            raise ValueError(
                f"system_prompt must be a string, received {type(system_prompt).__name__}"
            )

        self._client = model_client
        self._dispatcher = dispatcher
        self._registry = registry
        self._cost = cost_tracker
        self._sink = trace_sink
        self._renderer = renderer
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._cost_cap = float(cost_cap)
        self._notes_file = Path(notes_file) if notes_file is not None else None
        self._notes_reader = notes_reader
        # Req 18.4, Req 18.6: both of these default to None, and None in both is
        # the whole of the disabled state -- `_memory_message` returns before it
        # touches anything when neither was supplied.
        self._memory_file = Path(memory_file) if memory_file is not None else None
        self._memory_reader = memory_reader
        self._memory_max_chars = max(0, int(memory_max_chars))

    # -- accessors -------------------------------------------------------- #

    @property
    def max_steps(self) -> int:
        """The step bound in force for every run of this loop (Req 4.6)."""
        return self._max_steps

    @property
    def cost_cap(self) -> float:
        """The spend bound in dollars in force for every run (Req 4.8)."""
        return self._cost_cap

    @property
    def run_id(self) -> str:
        """The trace run identifier, or a fresh one when there is no sink."""
        identifier = getattr(self._sink, "run_id", None)
        return identifier if isinstance(identifier, str) and identifier else ""

    # -- the run ---------------------------------------------------------- #

    def run(self, task: str) -> RunResult:
        """Perform steps until the run ends, and return its one result (Req 4).

        The body is the step order of the module docstring, in that order and
        nowhere else, because three requirements are claims about where a check
        sits. A reader checking Property 1, 2 or 3 against the code should be
        able to read this method top to bottom and see the claim.
        """
        run_id = self.run_id or new_run_id()
        warnings: list[str] = []
        trace = _PendingTrace(self._sink, warnings)
        messages: list[Message] = []
        tools_called: list[str] = []
        completed_steps = 0

        # Req 4.1, Req 4.12: trim and bound the task before anything else. No
        # run starts, no model call is issued, and no trace line is written.
        trimmed, task_error = validate_task(task)
        if task_error is not None:
            self._error(task_error)
            return self._result(
                run_id,
                RunOutcome.INVALID_TASK,
                steps=0,
                messages=(),
                tools_called=tools_called,
                error_message=task_error,
                warnings=warnings,
            )

        # Req 4.1: the system prompt is submitted by the client on every call
        # (Req 2.12), so the conversation itself starts with the task alone.
        messages.append(Message(role="user", content=trimmed))

        # Req 18.4: the optional memory file, as a second message of the
        # *initial* conversation state -- appended before step 1, and after the
        # task, which leaves the task string itself untouched and keeps it the
        # first message of the conversation for every reader that relies on
        # that. Req 18.8's failure path appends nothing and warns instead.
        # With no memory source injected this call returns None without
        # touching the filesystem, the renderer or `warnings`, so a run with the
        # feature unconfigured has exactly the conversation it had before this
        # block existed (Req 18.6, Req 18.7).
        memory_message = self._memory_message(warnings)
        if memory_message is not None:
            messages.append(memory_message)

        try:
            while True:
                # 1. Req 4.6, Req 4.7: the step bound, before the model call.
                if completed_steps >= self._max_steps:
                    message = (
                        f"Run ended: the step limit of {self._max_steps} "
                        f"{_plural('step', self._max_steps)} was reached and no final "
                        f"answer was produced."
                    )
                    self._error(message)
                    trace.flush(RunOutcome.STEP_LIMIT)
                    return self._result(
                        run_id,
                        RunOutcome.STEP_LIMIT,
                        steps=completed_steps,
                        messages=messages,
                        tools_called=tools_called,
                        error_message=message,
                        warnings=warnings,
                    )

                # 2. Req 4.8, Req 4.9: the spend bound, before the model call.
                # Reading the total here and not after the call is what makes
                # the worst case "the cap plus one call" (Property 2).
                total_usd = self._cost.total_usd
                if total_usd >= self._cost_cap:
                    message = (
                        f"Run ended: the cost cap of ${self._cost_cap:.4f} was reached. "
                        f"The run cost ${total_usd:.4f} after {completed_steps} "
                        f"{_plural('step', completed_steps)}, so no further model call "
                        f"was made and no final answer was produced."
                    )
                    self._error(message)
                    trace.flush(RunOutcome.COST_CAP)
                    return self._result(
                        run_id,
                        RunOutcome.COST_CAP,
                        steps=completed_steps,
                        messages=messages,
                        tools_called=tools_called,
                        error_message=message,
                        warnings=warnings,
                    )

                # 3. Req 4.4: steps are numbered from 1, one number per call.
                step_no = completed_steps + 1
                self._step_start(step_no)
                started_at = iso_utc_millis(datetime.now(timezone.utc))
                started = time.monotonic()

                # 4. Req 4.5: exactly the registry's schemas, on every call.
                try:
                    reply = self._client.complete(
                        self._system_prompt,
                        tuple(messages),
                        self._registry.schemas(),
                    )
                except ModelClientError as exc:
                    # Req 4.13, Req 2.7: the typed error after the client's own
                    # retries. No reply, so no usage to record, nothing to
                    # append, and no trace line for an attempt that produced no
                    # step. The conversation and the records of the completed
                    # steps are retained exactly as they were.
                    message = (
                        f"Run ended because the model call failed: {exc} "
                        f"{completed_steps} {_plural('step', completed_steps)} completed, "
                        f"total cost ${self._cost.total_usd:.4f}."
                    )
                    self._error(message)
                    trace.flush(RunOutcome.MODEL_ERROR)
                    return self._result(
                        run_id,
                        RunOutcome.MODEL_ERROR,
                        steps=completed_steps,
                        messages=messages,
                        tools_called=tools_called,
                        error_message=message,
                        warnings=warnings,
                    )

                # 5. Req 3.1, Req 3.2: the call is complete, so it is accounted
                # for before anything else can end the run -- a step that ends
                # the run still cost what it cost.
                cost_usd = self._cost.record(reply.usage, getattr(self._client, "model_id", ""))

                # 6. Req 2.13: the assistant message carries the text *and* the
                # tool calls; neither is discarded, and it enters the
                # conversation before the results do (Req 2.12).
                messages.append(reply.as_assistant_message())

                # 7. Req 4.3, Req 4.13: a reply with no tool request either is
                # the final answer or is a reply the loop cannot act on.
                if not reply.has_tool_calls:
                    record = TraceRecord(
                        step=step_no,
                        started_at=started_at,
                        duration_ms=elapsed_ms(started),
                        calls=(),
                        assistant_text=reply.text,
                        usage=reply.usage,
                        cost_usd=cost_usd,
                    )
                    trace.hold(record)
                    completed_steps += 1

                    if reply.has_text:
                        trace.flush(RunOutcome.FINAL_ANSWER)
                        return self._result(
                            run_id,
                            RunOutcome.FINAL_ANSWER,
                            final_answer=reply.text,
                            steps=completed_steps,
                            messages=messages,
                            tools_called=tools_called,
                            warnings=warnings,
                        )

                    message = (
                        f"Run ended because the model call failed: the reply contained "
                        f"neither a tool request nor any text. {completed_steps} "
                        f"{_plural('step', completed_steps)} completed, total cost "
                        f"${self._cost.total_usd:.4f}."
                    )
                    self._error(message)
                    trace.flush(RunOutcome.MODEL_ERROR)
                    return self._result(
                        run_id,
                        RunOutcome.MODEL_ERROR,
                        steps=completed_steps,
                        messages=messages,
                        tools_called=tools_called,
                        error_message=message,
                        warnings=warnings,
                    )

                # 8. Req 4.2, Req 4.13: more than 8 requests in one step, or a
                # reply in which no requested name exists at all, is a reply the
                # loop cannot act on. A reply mixing known and unknown names is
                # dispatched: each unknown one becomes an error result the model
                # can correct from (Req 5.4, Req 5.11), which is the better
                # answer whenever there is still real work in the step.
                model_error = self._unactionable(reply, completed_steps)
                if model_error is not None:
                    record = TraceRecord(
                        step=step_no,
                        started_at=started_at,
                        duration_ms=elapsed_ms(started),
                        calls=tuple(
                            TracedCall(index=index, name=call.name, input=dict(call.arguments))
                            for index, call in enumerate(reply.tool_calls)
                        ),
                        assistant_text=reply.text,
                        usage=reply.usage,
                        cost_usd=cost_usd,
                    )
                    trace.hold(record)
                    completed_steps += 1
                    self._error(model_error)
                    trace.flush(RunOutcome.MODEL_ERROR)
                    return self._result(
                        run_id,
                        RunOutcome.MODEL_ERROR,
                        steps=completed_steps,
                        messages=messages,
                        tools_called=tools_called,
                        error_message=model_error,
                        warnings=warnings,
                    )

                # 9. Req 4.2, Req 5.11: dispatch in reply order, every call
                # separately -- two identical requests are two executions and
                # are never deduplicated -- appending each result, error or not,
                # inside this step and before the next model call.
                traced: list[TracedCall] = []
                for index, call in enumerate(reply.tool_calls):
                    self._tool_call(call)
                    result = self._ground(call, self._dispatcher.dispatch(call))
                    self._tool_result(result)
                    tools_called.append(call.name)
                    messages.append(
                        Message(role="tool", content=result.content, tool_call_id=call.id)
                    )
                    traced.append(TracedCall.from_dispatch(index, call, result))

                # 10. Req 11.1, Req 11.7: one record per step, carrying every
                # call of that step in dispatch order, written whatever happened
                # inside the step.
                trace.hold(
                    TraceRecord(
                        step=step_no,
                        started_at=started_at,
                        duration_ms=elapsed_ms(started),
                        calls=tuple(traced),
                        assistant_text=reply.text,
                        usage=reply.usage,
                        cost_usd=cost_usd,
                    )
                )

                # 11. The step is complete; the bounds are re-checked at the top.
                completed_steps += 1
        finally:
            # An unexpected exception on the way out would otherwise lose the
            # held record. Flushed without an outcome, because in that case the
            # run reached none.
            trace.flush()

    # -- classification --------------------------------------------------- #

    def _unactionable(self, reply: ModelReply, completed_steps: int) -> str | None:
        """The Req 4.13 message for a reply the loop cannot act on, or ``None``.

        Two conditions, both read off the reply before any tool runs:

        * more than :data:`MAX_TOOL_CALLS_PER_STEP` requests in one step;
        * every requested name absent from the registry.

        The second is the design's reading of Req 4.13's "names a Tool that is
        absent from the Tool_Registry" against Req 5.4 and Req 5.11, which
        require an unknown name to come back as an error result the model can
        correct from. Both hold here: a step with at least one real tool call
        proceeds and the unknown ones are reported as errors, while a step whose
        every request is unknown has no work in it at all and ends the run
        rather than spending a step feeding the model its own mistake.
        """
        count = len(reply.tool_calls)
        if count > MAX_TOOL_CALLS_PER_STEP:
            return (
                f"Run ended because the model call failed: the reply requested {count} "
                f"tool calls in one step, and at most {MAX_TOOL_CALLS_PER_STEP} are "
                f"executed per step. {completed_steps + 1} "
                f"{_plural('step', completed_steps + 1)} completed, total cost "
                f"${self._cost.total_usd:.4f}."
            )

        unknown = [call.name for call in reply.tool_calls if self._registry.resolve(call.name) is None]
        if reply.tool_calls and len(unknown) == count:
            names = ", ".join(repr(name) for name in unknown)
            registered = ", ".join(self._registry.names()) or "none are registered"
            return (
                f"Run ended because the model call failed: the reply requested only "
                f"tools that are not registered ({names}). Registered tools: "
                f"{registered}. {completed_steps + 1} "
                f"{_plural('step', completed_steps + 1)} completed, total cost "
                f"${self._cost.total_usd:.4f}."
            )
        return None

    # -- grounding (Req 12.8) --------------------------------------------- #

    def _ground(self, call: ToolCall, result: ToolResult) -> ToolResult:
        """Ensure a document tool's result is delimited before it is appended.

        The reasoning for each of the three branches is in the module docstring
        under "Grounding". In one line: document text is always enclosed and
        nothing from either tool can forge a delimiter, while a result the tool
        already delimited keeps its own per-entry boundaries and an error result
        stays readable as the correction it is.

        Returns the result unchanged whenever the content does not change, so
        the conversation, the renderer and the trace record all see one value.
        """
        if call.name not in GROUNDED_TOOLS:
            return result

        content = result.content
        if result.is_error:
            # Application text the model is meant to act on (Req 5.11); only a
            # forged delimiter echoed from model input is removed.
            grounded = sanitize(content)
        elif BEGIN in content and END in content:
            # Already enclosed by the tool, which sanitised what it enclosed.
            # Re-wrapping would strip those markers and destroy the per-entry
            # boundaries of a search result (Req 8.4).
            grounded = content
        else:
            grounded = wrap(content)

        if grounded == content:
            return result
        return replace(result, content=grounded)

    # -- the optional memory file (Req 18.4, Req 18.8) -------------------- #
    #
    # One optional feature, behind one gate. Everything below returns
    # immediately unless a memory source was injected, which is what makes
    # Req 18.6 and Req 18.7 structural here: with the feature unconfigured no
    # file is opened, no message is printed, no warning is recorded and no
    # conversation entry is added, so the run is the run that existed before.
    #
    # The text is wrapped by `grounding.wrap`, the same function the document
    # tools use. A memory file is untrusted input in exactly the way a document
    # is -- it is usually a previous run's own output, which is to say text a
    # model wrote -- so it is enclosed in the untrusted-data delimiters and any
    # forged delimiter inside it is removed (Req 12.8). Without the sanitising
    # half, a memory file could close its own quoted block and have the rest of
    # its content read as trusted conversation.

    def _memory_message(self, warnings: list[str]) -> Message | None:
        """The initial-conversation memory message, or ``None`` (Req 18.4).

        ``None`` means "append nothing", and covers both the disabled state and
        the Req 18.8 failure path, which differ only in whether a message was
        printed: a configured path that is absent or unreadable produces one
        warning line naming the path and the reason, and the run continues with
        the system prompt and the task string only.
        """
        if self._memory_file is None and self._memory_reader is None:
            return None

        text, reason = self._read_memory()
        if reason is not None:
            line = (
                f"warning: {MEMORY_FILE_VAR}: the memory file could not be read: "
                f'path "{self._memory_file or _MEMORY_SOURCE_LABEL}" ({reason}). '
                f"The run started with the system prompt and the task only, and "
                f"continued."
            )
            # Recorded as well as printed, for the same reason the access-check
            # warnings are (Req 1.10): the run summary repeats them (Req 3.9).
            warnings.append(line)
            self._warning(line)
            return None

        # A present-but-empty file is not a failure and is not special-cased:
        # it yields an empty delimited block, which says "the memory file exists
        # and holds nothing" rather than silently saying nothing at all.
        return Message(role="user", content=wrap(text or ""))

    def _read_memory(self) -> tuple[str | None, str | None]:
        """Return ``(text, None)`` or ``(None, reason)`` for the memory source.

        At most :attr:`_memory_max_chars` characters are returned, from either
        source: the limit is Req 18.4's and belongs to the feature, not to the
        filesystem, so an injected reader is truncated the same way a file is.
        The read is a bounded ``read(limit)`` rather than a whole-file read
        followed by a slice, so a memory file far larger than the limit costs
        the limit.
        """
        limit = self._memory_max_chars

        if self._memory_reader is not None:
            try:
                text = self._memory_reader()
            except Exception as exc:  # noqa: BLE001 - Req 18.8: the run continues
                return None, _failure_reason(exc)
            if not isinstance(text, str):
                return None, (
                    f"the memory source returned {type(text).__name__}, not text"
                )
            return text[:limit], None

        path = self._memory_file
        assert path is not None  # the caller's gate guarantees one or the other
        try:
            if not path.exists():
                # Named separately from the OSError branch because this is the
                # one failure an operator causes by typing a path that is not
                # there, and "the path does not exist" is the phrasing the
                # configuration access check already uses for it (Req 1.10).
                return None, "the path does not exist"
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read(limit), None
        except Exception as exc:  # noqa: BLE001 - a directory, a denial, a device
            return None, _failure_reason(exc)

    # -- the result ------------------------------------------------------- #

    def _result(
        self,
        run_id: str,
        outcome: RunOutcome,
        *,
        steps: int,
        messages: Sequence[Message],
        tools_called: Sequence[str],
        final_answer: str | None = None,
        error_message: str | None = None,
        warnings: Sequence[str] = (),
    ) -> RunResult:
        """Build the one :class:`RunResult` a run returns (Req 4.14).

        Every return path in :meth:`run` goes through here, so the totals, the
        warnings and the notes snapshot cannot be read on one path and forgotten
        on another, and no path can return two outcomes or none.
        """
        totals = self._cost.totals()
        return RunResult(
            run_id=run_id,
            outcome=outcome,
            final_answer=final_answer,
            steps=steps,
            input_tokens=totals.input_tokens,
            output_tokens=totals.output_tokens,
            cost_usd=totals.usd,
            tools_called=list(tools_called),
            notes_text_at_end=self._notes_snapshot(),
            trace_path=self._trace_path(),
            error_message=error_message,
            warnings=[*self._cost.warnings(), *warnings],
            messages=tuple(messages),
        )

    def _notes_snapshot(self) -> str:
        """The notes file's text at the end of the run, or ``""`` (Req 14.3).

        Read with ``errors="replace"`` and guarded against every filesystem
        condition: an absent file is the normal state of a run that saved
        nothing, and a run must not fail at its last step because a snapshot
        taken for a report could not be read.

        Plain text, not the delimited rendering ``read_notes`` returns: this is
        a snapshot for the eval checks to search, not something a model reads.
        """
        if self._notes_reader is not None:
            try:
                text = self._notes_reader()
            except Exception:  # noqa: BLE001 - a snapshot may not end a run
                return ""
            return text if isinstance(text, str) else ""
        if self._notes_file is None:
            return ""
        try:
            return self._notes_file.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - absent, denied, or not a file
            return ""

    def _trace_path(self) -> str:
        """The configured trace file as text, or ``""`` when there is no sink."""
        path = getattr(self._sink, "path", None)
        return "" if path is None else str(path)

    # -- narration (Req 13.1, Req 13.2, Req 13.8) ------------------------- #
    #
    # Four one-line wrappers, each guarded. Narration is a report on the work,
    # never the work: a renderer that cannot write must not be the reason an
    # answer is lost. The shipped renderer already swallows stream failures;
    # these guards cover a renderer that does not.

    def _step_start(self, step_no: int) -> None:
        if self._renderer is None:
            return
        try:
            self._renderer.step_start(step_no)
        except Exception:  # noqa: BLE001
            pass

    def _tool_call(self, call: ToolCall) -> None:
        """Narrated *before* the dispatch, so a slow tool shows as a wait."""
        if self._renderer is None:
            return
        try:
            self._renderer.tool_call(call)
        except Exception:  # noqa: BLE001
            pass

    def _tool_result(self, result: ToolResult) -> None:
        if self._renderer is None:
            return
        try:
            self._renderer.tool_result(result)
        except Exception:  # noqa: BLE001
            pass

    def _warning(self, message: str) -> None:
        """A non-fatal notice, to stderr (Req 13.8). Req 18.8 is its one user."""
        if self._renderer is None:
            return
        try:
            self._renderer.warning(message)
        except Exception:  # noqa: BLE001
            pass

    def _error(self, message: str) -> None:
        """The abnormal-termination message, to stderr (Req 4.7, 4.8, 4.12, 4.13)."""
        if self._renderer is None:
            return
        try:
            self._renderer.error(message)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# The one-slot trace buffer
# --------------------------------------------------------------------------- #


class _PendingTrace:
    """Holds one completed step's record until the next step or the run's end.

    The reason is in the module docstring: Req 11.4 puts the run outcome on the
    final step's record, and on the two bound-check paths the outcome is not
    known until after that step has completed. Holding the record for one beat
    is what lets the outcome reach it while keeping exactly one line per step in
    ascending order (Req 11.8).

    Appends are guarded even though :class:`~agent_lab.trace.JsonlTraceLogger`
    never raises, because a caller may inject any sink and Req 11.11 requires
    the run to continue to the same outcome and the same final answer when an
    append fails. A failure adds one warning line to the run's warnings.
    """

    __slots__ = ("_sink", "_warnings", "_record")

    def __init__(self, sink: TraceSink | None, warnings: list[str]) -> None:
        self._sink = sink
        self._warnings = warnings
        self._record: TraceRecord | None = None

    def hold(self, record: TraceRecord) -> None:
        """Append whatever was held, then hold ``record``."""
        self.flush()
        self._record = record

    def flush(self, outcome: RunOutcome | None = None) -> None:
        """Append the held record, stamped with ``outcome`` when one is given.

        Idempotent: the slot is cleared first, so the ``finally`` in
        :meth:`AgentLoop.run` cannot write a second copy of a record an
        outcome-stamped flush has already written.
        """
        record = self._record
        self._record = None
        if record is None or self._sink is None:
            return
        if outcome is not None:
            record = replace(record, run_outcome=outcome)
        try:
            self._sink.append(record)
        except Exception as exc:  # noqa: BLE001 - Req 11.11: the run continues
            self._warnings.append(
                f"Warning: step {getattr(record, 'step', '?')} could not be written to "
                f"the trace file: {type(exc).__name__}: {exc}. The record was "
                f"discarded; the run continued."
            )


def _plural(noun: str, count: int) -> str:
    """``"step"`` or ``"steps"``, so the printed messages read correctly."""
    return noun if count == 1 else f"{noun}s"


def _failure_reason(exc: BaseException) -> str:
    """The reason clause of the Req 18.8 message, from the exception that caused it.

    ``OSError.strerror`` is preferred over ``str(exc)`` because the latter
    repeats the filename the message already quotes; the exception type is kept
    in front of it, so a reader can tell a denial from a missing device.
    """
    detail = getattr(exc, "strerror", None) or str(exc) or "no further detail"
    return f"{type(exc).__name__}: {detail}"


__all__ = [
    "COST_CAP",
    "EXIT_CODES",
    "GROUNDED_TOOLS",
    "MAX_STEPS",
    "MEMORY_FILE_VAR",
    "MEMORY_READ_MAX_CHARS",
    "MAX_TASK_CHARS",
    "MAX_TOOL_CALLS_PER_STEP",
    "MIN_TASK_CHARS",
    "AgentLoop",
    "RunOutcome",
    "RunResult",
    "exit_code_for",
    "validate_task",
]
