"""Terminal output: step narration, the final answer, and the run summary (Req 13).

Everything an operator sees during a run is written from here. The loop and the
CLI hold a :class:`Renderer` and call its methods; they never touch
``sys.stdout`` or ``sys.stderr`` themselves. That is what makes Req 13.9
checkable: this module has exactly one write function, that function applies
:func:`~agent_lab.redaction.redact` to every string before it reaches a stream,
and no other code in the presentation path formats output.

Five decisions shape the module.

**The two renderers differ in colour and nothing else.** Req 13.6 requires the
plain rendering to carry the same step numbers, tool names, tool inputs,
shortened results, success and failure markers, final answer and summary fields
as the `rich` rendering. The way that is guaranteed here is that the *text* is
built by module-level functions -- :func:`format_step_heading`,
:func:`format_tool_call`, :func:`format_tool_result`,
:func:`format_final_answer`, :func:`format_run_summary` -- which both renderers
call. :class:`RichRenderer` overrides only how a finished line reaches the
stream, passing it to `rich` with a style; it never composes a different line
and it never injects markup into the text. So the two renderings are the same
characters, one of them wrapped in colour escapes.

**`rich` is optional and its absence is not an error.** The import is deferred
to the first write and guarded, so importing this module -- or constructing a
:class:`RichRenderer` -- works on an installation without `rich`.
A :class:`RichRenderer` that cannot import `rich` writes plain text rather than
failing, and :func:`build_renderer` picks the plain renderer up front when
`rich` is unavailable, which is what the CLI calls (Req 13.5, Req 13.6).

**Flushing is not an optimisation question.** Req 13.1 wants the step heading on
screen before the step's first tool runs and within 100 ms of the step
beginning, and Req 13.2 wants the tool lines out before the next step begins.
Every write here flushes its stream immediately, so both hold for any call order
the loop uses: the heading is on screen when :meth:`Renderer.step_start`
returns, and a tool line is on screen when :meth:`Renderer.tool_result`
returns. There is no buffer to drain and no background writer.

**A sink may not end a run.** A closed or broken stream -- a piped `head`, a
redirect to a full disk -- raises on write. Narration is not the work; losing a
line is better than losing the run, so writes are guarded and a failed write is
silently dropped, exactly as the trace logger and the confirmation gate do.

**The display limit is validated here as well as in configuration.**
:func:`~agent_lab.config.load_config` already rejects a ``DISPLAY_MAX_CHARS``
outside 80-10,000, but a renderer can be constructed directly by the eval runner
or a test, and Req 13.3 asks for the rejection to name the permitted range
wherever the value comes from. :class:`DisplayLimitError` subclasses
:class:`~agent_lab.config.ConfigError` so the CLI's existing single catch and
its configuration exit code cover this case too.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import IO, Any, Protocol, runtime_checkable

from agent_lab.config import Config, ConfigError
from agent_lab.cost import DISPLAY_DECIMALS, Totals
from agent_lab.model_client import ToolCall
from agent_lab.redaction import redact
from agent_lab.tools.base import ToolResult

# --------------------------------------------------------------------------- #
# The display limit (Req 13.3)
# --------------------------------------------------------------------------- #

# Req 13.3: the accepted range and the default. The same numbers as
# ``config.INT_RANGES["DISPLAY_MAX_CHARS"]`` and ``config.DEFAULTS``; they are
# named here too because this module is the one that applies them, and a test
# asserts the two agree rather than trusting a comment.
DISPLAY_MIN_CHARS = 80
DISPLAY_MAX_CHARS = 10_000
DEFAULT_DISPLAY_MAX_CHARS = 500

# The name of the environment variable an operator would have set, quoted in the
# rejection message so the message says what to go and change.
DISPLAY_LIMIT_VAR = "DISPLAY_MAX_CHARS"


class DisplayLimitError(ConfigError):
    """A display limit outside the permitted range (Req 13.3).

    A :class:`~agent_lab.config.ConfigError`, so the CLI's existing handling --
    message on stderr, configuration exit code, before the first step -- applies
    without a second catch clause.
    """


def validate_display_max_chars(value: Any) -> int:
    """Return ``value`` as an accepted display limit, or raise (Req 13.3).

    The message names the offending value and the permitted range, which is the
    whole of what Req 13.3 asks a rejection to say.

    ``bool`` is refused explicitly: ``True`` is an ``int`` in Python and would
    otherwise be accepted as the limit 1 and then fail the range check with a
    confusing message about the number one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise DisplayLimitError(
            f"{DISPLAY_LIMIT_VAR} must be an integer between {DISPLAY_MIN_CHARS} "
            f"and {DISPLAY_MAX_CHARS} inclusive, but received {value!r}."
        )
    if not DISPLAY_MIN_CHARS <= value <= DISPLAY_MAX_CHARS:
        raise DisplayLimitError(
            f"{DISPLAY_LIMIT_VAR} must be an integer between {DISPLAY_MIN_CHARS} "
            f"and {DISPLAY_MAX_CHARS} inclusive, but received {value}. "
            f"The default is {DEFAULT_DISPLAY_MAX_CHARS}."
        )
    return value


# --------------------------------------------------------------------------- #
# The fixed vocabulary of the output (Req 13.6)
# --------------------------------------------------------------------------- #

# Req 13.7: two markers, textually distinct, so an error result is
# distinguishable from a successful one without colour. Distinct in more than
# case: a reader piping output through a tool that folds case, or reading it in
# a terminal with no colour, still sees two different words.
SUCCESS_MARKER = "[ok]"
FAILURE_MARKER = "[FAILED]"

# Req 13.4: the labelled heading the final answer is written under.
FINAL_ANSWER_HEADING = "Final answer:"

# Req 3.9: the first token of the run summary line.
SUMMARY_PREFIX = "Run summary:"

# Req 18.2: the line printed on stderr when text that had already been streamed
# turned out to belong to a reply that then requested a tool, so it was not the
# final answer after all. See :class:`AnswerStreamer` for why that can happen.
STREAM_RETRACTED_MESSAGE = (
    "warning: the text streamed above was not the final answer -- the model "
    "went on to request a tool in the same reply. The run continues."
)

# The indentation of a tool line under its step heading. Two spaces, ASCII: the
# Windows console's default code page is not UTF-8, and a box-drawing character
# here would raise ``UnicodeEncodeError`` on the target host for the sake of
# decoration. Every literal in this module is ASCII for that reason.
INDENT = "  "


def shorten(text: str, limit: int) -> tuple[str, bool]:
    """Shorten ``text`` to ``limit`` characters, appending an indicator (Req 13.3).

    Returns the text to display and whether it was shortened. The indicator
    states both the kept and the original character count, so a reader can tell
    how much was withheld and raise the limit deliberately rather than guess.

    The indicator is appended *after* the ``limit`` characters rather than
    counted inside them. The limit governs how much of the tool result is shown,
    and an indicator that ate into the result it describes would make a limit of
    80 show almost nothing.
    """
    if len(text) <= limit:
        return text, False
    kept = text[:limit]
    return (
        f"{kept}... [truncated for display: showing the first {limit} of "
        f"{len(text)} characters]"
    ), True


def split_on_partial_secret(text: str, secret: str | None) -> tuple[str, str]:
    """Split ``text`` into the part safe to write now and the part to hold back.

    This exists for one problem that only arises when output is written in
    fragments rather than lines (Req 18.2). :func:`~agent_lab.redaction.redact`
    replaces occurrences of the whole key in the string it is given, so a key
    that arrives split across two provider chunks -- ``"sk-abc"`` then
    ``"def"`` -- is not an occurrence in either fragment and would reach the
    terminal intact, defeating Req 13.9.

    The fix is to withhold, at the end of each fragment, the longest suffix that
    is a proper prefix of the secret, and to prepend it to the next fragment
    before redacting again. Then any occurrence of the key, however it was
    chopped up, is a complete occurrence in some joined buffer, and is replaced
    there. What is written is never a prefix of the key followed later by its
    remainder.

    Two consequences, both deliberate and both bounded:

    * **The hold-back is only ever an ambiguous tail.** A fragment whose end
      cannot begin the secret is released in full and immediately, so the 1
      second of Req 18.2 is met for every fragment of real model prose. Only a
      tail that genuinely looks like the start of the key waits for the next
      fragment -- and :meth:`PlainRenderer.end_stream` flushes whatever is still
      held as soon as the stream ends, so nothing is withheld indefinitely.
    * **The guarantee is about the configured key and nothing else.** This
      cannot recognise a credential it was not given, and it is not a secret
      scanner. It makes the streamed path exactly as safe as the line-oriented
      path, no safer.

    Returns ``(release, hold)``, whose concatenation is always ``text``.
    """
    if not secret or not text:
        return text, ""
    # A suffix as long as the secret itself would be a complete occurrence,
    # which `redact` has already replaced; only shorter ones are ambiguous.
    longest = min(len(secret) - 1, len(text))
    for size in range(longest, 0, -1):
        if secret.startswith(text[-size:]):
            return text[:-size], text[-size:]
    return text, ""


def format_step_heading(step: int) -> str:
    """The step heading line, carrying the step number (Req 13.1)."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(f"step must be an int, received {step!r}")
    if step < 1:
        # Req 4.4 numbers steps from 1. A zero or negative number here means the
        # caller's bookkeeping is wrong, and narration that misreports the step
        # number is worse than no narration.
        raise ValueError(f"step must be 1 or greater, received {step}")
    return f"Step {step}"


def format_arguments(arguments: Mapping[str, Any] | None) -> str:
    """The tool inputs as one line of JSON (Req 13.2).

    JSON rather than ``repr`` so the inputs read the same way here, in the trace
    file and in the provider's request: one representation an operator can
    compare across all three.

    ``default=str`` and the fallback keep this total. The arguments come from
    model output and have been through ``json.loads`` by the time the dispatcher
    sees them, so they are ordinarily serialisable -- but narration must not be
    able to raise over an argument value, so an unserialisable mapping falls
    back to ``repr``.
    """
    if not arguments:
        return "{}"
    try:
        return json.dumps(dict(arguments), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - narration may not raise over its input
        return repr(dict(arguments))


def format_tool_call(
    name: str,
    arguments: Mapping[str, Any] | None,
    limit: int = DEFAULT_DISPLAY_MAX_CHARS,
) -> str:
    """The line naming the tool and its inputs (Req 13.2).

    The display limit is applied to the rendered inputs as well as to the
    result. ``save_note`` can be asked to write several thousand characters, and
    a narration line that reproduced the whole note would bury the step it is
    meant to describe. The limit is a display concern and the inputs are
    display; the conversation and the trace record keep their own full copies.
    """
    rendered, _ = shorten(format_arguments(arguments), limit)
    return f"{INDENT}tool: {name}  inputs: {rendered}"


def format_tool_result(
    result: ToolResult,
    limit: int = DEFAULT_DISPLAY_MAX_CHARS,
) -> str:
    """The marker and the shortened result content (Req 13.2, Req 13.3, Req 13.7).

    An error result is preceded by :data:`FAILURE_MARKER` and a successful one by
    :data:`SUCCESS_MARKER`. Nothing about an error stops the line being written
    or the next step being narrated -- Req 13.7 requires the run to continue, and
    the way that is honoured here is that an error is a different marker and
    nothing else.

    A result the dispatcher already truncated (Req 5.12) carries its own marker
    inside ``content``; that marker is part of what the model saw, so it is
    shown. This function's own indicator is about *display* and says so, which
    is why the two can appear together without ambiguity.
    """
    marker = FAILURE_MARKER if result.is_error else SUCCESS_MARKER
    content, _ = shorten(result.content, limit)
    return f"{INDENT}{marker} {content}"


def format_final_answer(text: str) -> str:
    """The final answer under its labelled heading (Req 13.4).

    Never shortened. The display limit governs narration of tool results; the
    final answer is the product of the run, and truncating it would make the
    output of a long answer useless.
    """
    return f"{FINAL_ANSWER_HEADING}\n{text}"


def format_run_summary(totals: Totals) -> str:
    """The one-line run summary (Req 3.9).

    The exact shape, with two spaces between fields::

        Run summary: steps=3  input_tokens=4210  output_tokens=388  total_tokens=4598  cost=$0.0137

    ``total_tokens`` is read from the snapshot's property rather than added here,
    so the printed figure cannot disagree with its two parts. The cost is
    rounded to :data:`~agent_lab.cost.DISPLAY_DECIMALS` (4) places for display
    only; the tracker keeps its own fuller precision (Req 3.2).
    """
    return (
        f"{SUMMARY_PREFIX} steps={totals.steps}"
        f"  input_tokens={totals.input_tokens}"
        f"  output_tokens={totals.output_tokens}"
        f"  total_tokens={totals.total_tokens}"
        f"  cost=${totals.usd:.{DISPLAY_DECIMALS}f}"
    )


# --------------------------------------------------------------------------- #
# The protocol (Req 13)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Renderer(Protocol):
    """What the loop and the CLI call to put text on a terminal.

    The call order the loop uses, and the one the flush guarantees are written
    for::

        renderer.step_start(1)                      # before the first tool runs
        renderer.tool_call("calculator", args)      # before the tool executes
        renderer.tool_result(result)                # after it returns
        ...
        renderer.final_answer(text)                 # at the end of the run
        renderer.run_summary(tracker)               # immediately after it

    :meth:`warning` and :meth:`error` may be called at any point and write to a
    different stream (Req 13.8), so they never interrupt that sequence.
    """

    def step_start(self, step: int) -> None: ...

    def tool_call(
        self, call: ToolCall | str, arguments: Mapping[str, Any] | None = None
    ) -> None: ...

    def tool_result(self, result: ToolResult) -> None: ...

    def final_answer(self, text: str) -> None: ...

    def run_summary(self, totals: Totals | Any, warnings: Sequence[str] | None = None) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...


# --------------------------------------------------------------------------- #
# The plain renderer (Req 13.6)
# --------------------------------------------------------------------------- #


class PlainRenderer:
    """Writes the Req 13 field set as plain ASCII text.

    The reference rendering. :class:`RichRenderer` is this class plus colour, so
    anything true of the text here is true of both (Req 13.6).

    Streams are injectable and resolved at write time. Injectable, so the eval
    runner and the unit tests can capture output by passing a buffer rather than
    by replacing process-wide state; resolved at write time, so a caller that
    passes nothing and later replaces ``sys.stdout`` still has its replacement
    written to.
    """

    __slots__ = ("_limit", "_secret", "_stdout", "_stderr", "_narrate", "_carry")

    def __init__(
        self,
        *,
        display_max_chars: int = DEFAULT_DISPLAY_MAX_CHARS,
        api_key: str | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        narrate: bool = True,
    ) -> None:
        """
        :param display_max_chars: the display limit, 80-10,000 (Req 13.3).
        :param api_key: the provider key to redact from every write (Req 13.9).
            ``None`` is accepted and redacts nothing, for the eval runner and
            for tests that configure no credential.
        :param stdout: where narration, the final answer and the summary go;
            ``sys.stdout`` when absent (Req 13.8).
        :param stderr: where warnings and errors go; ``sys.stderr`` when absent
            (Req 13.8).
        :param narrate: whether step narration is emitted. ``False`` suppresses
            step headings and tool lines only -- the final answer and the run
            summary print either way (Req 13.4).
        :raises DisplayLimitError: if ``display_max_chars`` is out of range.
        """
        self._limit = validate_display_max_chars(display_max_chars)
        self._secret = api_key
        self._stdout = stdout
        self._stderr = stderr
        self._narrate = bool(narrate)
        # The streamed-output hold-back of Req 18.2, empty except between two
        # fragments of one stream. See :func:`split_on_partial_secret`.
        self._carry = ""

    # -- construction from configuration --------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        narrate: bool = True,
    ) -> PlainRenderer:
        """Build a renderer from a validated :class:`~agent_lab.config.Config`.

        Takes the display limit and the key to redact from the one object that
        has already validated both, so the CLI has no opportunity to wire a
        renderer with no secret and leak the key through narration.
        """
        return cls(
            display_max_chars=config.display_max_chars,
            api_key=config.api_key,
            stdout=stdout,
            stderr=stderr,
            narrate=narrate,
        )

    # -- accessors ------------------------------------------------------------

    @property
    def display_max_chars(self) -> int:
        """The display limit in force (Req 13.3)."""
        return self._limit

    @property
    def narrates(self) -> bool:
        """Whether step narration is emitted."""
        return self._narrate

    # -- the Req 13 surface ---------------------------------------------------

    def step_start(self, step: int) -> None:
        """Write the step heading and flush it (Req 13.1).

        Called by the loop at the top of a step, before the first tool of that
        step is invoked. The flush happens inside this call, so the heading is on
        screen by the time it returns -- well inside the 100 ms Req 13.1 allows,
        since the only work is one formatted line and one stream flush.
        """
        if not self._narrate:
            return
        self._emit(format_step_heading(step), style="step")

    def tool_call(
        self, call: ToolCall | str, arguments: Mapping[str, Any] | None = None
    ) -> None:
        """Write the tool name and its inputs (Req 13.2).

        Accepts either the :class:`~agent_lab.model_client.ToolCall` the loop is
        about to dispatch, or a name and an argument mapping. The loop has a
        ``ToolCall``; a test usually has the two pieces.

        Written before the tool executes, so an operator watching a run sees
        which tool is being waited on rather than a pause.
        """
        if not self._narrate:
            return
        name, args = _normalise_call(call, arguments)
        self._emit(format_tool_call(name, args, self._limit), style="tool")

    def tool_result(self, result: ToolResult) -> None:
        """Write the marker and the shortened result, and flush (Req 13.2, 13.7).

        The flush is what Req 13.2 asks for: the tool lines of this step are on
        screen before the next step begins, whatever the loop does next. An error
        result gets the distinct failure marker and the run continues (Req 13.7).
        """
        if not self._narrate:
            return
        style = "failure" if getattr(result, "is_error", False) else "success"
        self._emit(format_tool_result(result, self._limit), style=style)

    def final_answer(self, text: str) -> None:
        """Write the final answer under its labelled heading (Req 13.4).

        Emitted whether or not narration is suppressed, because Req 13.4 says
        "regardless", and the run summary follows it immediately.
        """
        self._emit(format_final_answer(text), style="answer")

    # -- the streamed final answer (Req 18.2, optional) -----------------------
    #
    # Three methods, used only when `STREAM_OUTPUT=on`. With streaming off they
    # are never called and :meth:`final_answer` above is the one write, exactly
    # as before the three existed (Req 18.6, Req 18.7).
    #
    # The characters they put on stdout are the characters :meth:`final_answer`
    # would have put there: the heading line, the answer, one newline. So a run
    # whose stream equalled the answer is byte-identical to the same run with
    # streaming off, and the CLI can therefore suppress its own single write
    # rather than print the answer twice.

    def begin_stream(self) -> None:
        """Write the heading the streamed answer appears under (Req 13.4, 18.2).

        Called on the first token rather than before the request, so a reply
        that never produces answer text prints no heading.
        """
        self._carry = ""
        self._emit(FINAL_ANSWER_HEADING, style="answer")

    def stream_fragment(self, text: str) -> None:
        """Write one received fragment of the answer, with no newline (Req 18.2).

        Redacted and flushed on arrival, so the operator sees the token within
        the 1 second Req 18.2 allows -- there is no buffering here beyond the
        ambiguous-tail hold-back that :func:`split_on_partial_secret` explains,
        and nothing waits for a line to complete.
        """
        if not text:
            return
        joined = self._carry + text
        # Redact the *joined* buffer, so a key split across two fragments is a
        # complete occurrence here and is replaced (Req 13.9).
        release, self._carry = split_on_partial_secret(redact(joined, self._secret), self._secret)
        if release:
            self._emit(release, style="answer", newline=False)

    def end_stream(self) -> None:
        """Flush the held tail and close the streamed answer's line (Req 18.2).

        Whatever is still held cannot be the key -- it is a strict prefix of it
        and the stream has ended -- so it is written. The trailing newline is
        the one :meth:`final_answer` would have written.
        """
        tail, self._carry = self._carry, ""
        self._emit(tail + "\n", style="answer", newline=False)

    def run_summary(
        self,
        totals: Totals | Any,
        warnings: Sequence[str] | None = None,
    ) -> None:
        """Write the run summary line, then one line per accumulated warning.

        :param totals: a :class:`~agent_lab.cost.Totals` snapshot, or the
            :class:`~agent_lab.cost.CostTracker` itself -- in which case the
            snapshot and, unless ``warnings`` is given, the tracker's own
            warning lines are read from it. Accepting the tracker keeps the one
            call site in the CLI from having to remember to pass both halves.
        :param warnings: the warning lines to print under the summary. Defaults
            to the tracker's own (Req 3.3, Req 3.4); pass a sequence to add the
            configuration access warnings of Req 1.10 alongside them.

        The summary goes to stdout (Req 3.9, Req 13.4) and the warning lines to
        stderr, because they are warnings and Req 13.8 admits no exception for a
        warning that happens to be printed late. On a terminal both streams
        interleave in call order, so the operator still reads the warnings under
        the summary.
        """
        snapshot, own_warnings = _as_totals(totals)
        self._emit(format_run_summary(snapshot), style="summary")
        lines = own_warnings if warnings is None else warnings
        for line in lines:
            self.warning(line)

    def warning(self, message: str) -> None:
        """Write a warning to stderr (Req 13.8).

        The message is printed as given. The warning lines the cost tracker and
        the configuration access check produce already begin with their own
        label, and a second prefix added here would double it.
        """
        self._emit(message, to_err=True, style="warning")

    def error(self, message: str) -> None:
        """Write an error message to stderr (Req 13.8).

        The abnormal-termination messages of Req 4.7, Req 4.8 and Req 4.13 go
        here. Narration, the final answer and the run summary continue to go to
        stdout, which is the split Req 13.8 asks for.
        """
        self._emit(message, to_err=True, style="error")

    # -- the single write path (Req 13.9) -------------------------------------

    def _emit(
        self,
        text: str,
        *,
        to_err: bool = False,
        style: str | None = None,
        newline: bool = True,
    ) -> None:
        """Redact, write, flush. The only place this module touches a stream.

        Req 13.9 requires every occurrence of the provider API key value in
        anything written to stdout or stderr to be replaced by the redaction
        marker. One function reaches a stream, and that function redacts first,
        so there is no second path to audit. Subclasses override :meth:`_write`,
        which receives the already-redacted text and cannot reach the stream
        before this line has run.

        ``newline=False`` is the streamed-answer path of Req 18.2, and it is a
        parameter here rather than a second write function for exactly the
        reason above: a fragment written anywhere else would be a sink this
        docstring's claim does not cover. A fragment is written to the stream
        directly rather than through :meth:`_write`, because `rich` styles whole
        lines and would insert a line break after every token; the characters
        written are the same either way, which is all Req 13.6 asks.

        The write is guarded. A closed or broken stream must not end a run:
        narration is a report on the work, not the work.
        """
        safe = redact(text, self._secret)
        stream = self._resolve(to_err)
        if stream is None:  # pragma: no cover - sys.stdout can be None on Windows GUI hosts
            return
        try:
            if newline:
                self._write(safe, stream, style)
            else:
                stream.write(safe)
            stream.flush()
        except Exception:  # noqa: BLE001 - a sink may not end a run
            pass

    def _write(self, text: str, stream: IO[str], style: str | None) -> None:
        """Put one already-redacted line on the stream. Overridden for colour.

        ``style`` is the name of a semantic role, not a colour: the plain
        renderer ignores it entirely, which is the whole of the difference
        Req 13.6 permits between the two renderings.
        """
        stream.write(text + "\n")

    def _resolve(self, to_err: bool) -> IO[str] | None:
        """The stream to write to, looked up now rather than at construction."""
        if to_err:
            return self._stderr if self._stderr is not None else sys.stderr
        return self._stdout if self._stdout is not None else sys.stdout


# --------------------------------------------------------------------------- #
# The rich renderer (Req 13.5)
# --------------------------------------------------------------------------- #

# The style per semantic role. Values are `rich` style strings and are the only
# thing in this module that `rich` interprets. They are applied to a whole
# finished line through the ``style=`` argument, never as markup inside the
# text, so the characters written are the plain renderer's characters and the
# difference is colour (Req 13.6).
RICH_STYLES: dict[str, str] = {
    "step": "bold cyan",
    "tool": "cyan",
    "success": "green",
    "failure": "bold red",
    "answer": "bold white",
    "summary": "bold",
    "warning": "yellow",
    "error": "bold red",
}


def rich_available() -> bool:
    """Whether the optional `rich` package can be imported (Req 13.5, Req 13.6).

    Asked by :func:`build_renderer` and by the CLI. Imported and discarded here
    rather than cached, because the answer is a property of the installation and
    this is called once per run.
    """
    try:
        import rich.console  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


class RichRenderer(PlainRenderer):
    """The plain renderer's text, with colour and emphasis (Req 13.5).

    Only :meth:`_write` is overridden. The lines come from the same formatting
    functions, the redaction in :meth:`PlainRenderer._emit` has already run, and
    `rich` is handed a finished string with ``markup`` and ``highlight`` off --
    so it colours the line and changes nothing in it (Req 13.6).

    Degrades rather than fails. `rich` is an optional extra, the import is
    deferred to the first write, and if it cannot be imported this renderer
    writes exactly what :class:`PlainRenderer` would. Constructing a
    ``RichRenderer`` on an installation without `rich` is therefore safe; the CLI
    still prefers :func:`build_renderer`, which chooses the plain renderer up
    front so the class in use reflects the installation.
    """

    # One console per stream, built on first use. ``__slots__`` has to be
    # restated because the base class defines it; adding to it keeps instances
    # dict-free.
    __slots__ = ("_console_for", "_rich_console")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The stream the cached console was built for, compared by identity: the
        # streams are resolved late, so a caller that replaces ``sys.stdout``
        # mid-run must not keep writing to a console wrapping the old one.
        self._console_for: IO[str] | None = None
        self._rich_console: Any | None = None

    def _write(self, text: str, stream: IO[str], style: str | None) -> None:
        """Print through `rich` when it is available, plainly when it is not."""
        console = self._console(stream)
        if console is None:
            super()._write(text, stream, style)
            return
        console.print(
            text,
            style=RICH_STYLES.get(style or "", None),
            # No wrapping: `rich` would otherwise reflow a long tool result to
            # the terminal width, which would make the shortened text of
            # Req 13.3 differ from the plain renderer's by inserted newlines.
            soft_wrap=True,
        )

    def _console(self, stream: IO[str]) -> Any | None:
        """The `rich` console for ``stream``, or ``None`` if `rich` is absent."""
        if self._rich_console is not None and self._console_for is stream:
            return self._rich_console
        try:
            from rich.console import Console
        except Exception:  # noqa: BLE001 - the optional extra is not installed
            return None
        try:
            console = Console(
                file=stream,
                # Req 13.6: the text must be the plain renderer's text. Markup
                # off, so a result containing square brackets is not read as a
                # style tag; highlighting and emoji off, so no number, path or
                # ``:name:`` sequence in model output is rewritten.
                markup=False,
                highlight=False,
                emoji=False,
                soft_wrap=True,
            )
        except Exception:  # noqa: BLE001 - fall back rather than fail
            return None
        self._console_for = stream
        self._rich_console = console
        return console


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_renderer(
    config: Config | None = None,
    *,
    display_max_chars: int | None = None,
    api_key: str | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    narrate: bool = True,
    prefer_rich: bool = True,
) -> PlainRenderer:
    """The renderer the CLI uses: rich where available, plain otherwise.

    Req 13.5 asks for colour where `rich` is installed and Req 13.6 for the same
    fields as plain text where it is not. This is the one decision point for
    that, so no caller has to repeat the availability check.

    :param config: a validated configuration to read the display limit and the
        key to redact from. Either ``config`` or the explicit values.
    :param display_max_chars: overrides the configured limit; defaults to 500
        when neither is given (Req 13.3).
    :param api_key: overrides the configured key to redact (Req 13.9).
    :param prefer_rich: pass ``False`` to force the plain renderer -- for an
        eval run writing to a file, or a test asserting the plain rendering.
    :raises DisplayLimitError: if the resulting limit is out of range.
    """
    limit = display_max_chars
    if limit is None:
        limit = config.display_max_chars if config is not None else DEFAULT_DISPLAY_MAX_CHARS
    secret = api_key
    if secret is None and config is not None:
        secret = config.api_key

    cls = RichRenderer if (prefer_rich and rich_available()) else PlainRenderer
    return cls(
        display_max_chars=limit,
        api_key=secret,
        stdout=stdout,
        stderr=stderr,
        narrate=narrate,
    )


# --------------------------------------------------------------------------- #
# The streamed answer (Req 18.2, Req 18.9 -- optional, off by default)
# --------------------------------------------------------------------------- #


class AnswerStreamer:
    """Writes answer tokens to a renderer as the provider emits them (Req 18.2).

    Constructed by the CLI only when ``STREAM_OUTPUT=on``, and handed to the
    provider adapter as a token sink. :meth:`token` is called once per received
    content fragment, :meth:`retract` if the reply those fragments belonged to
    went on to request a tool, and :meth:`finish` once after the run.

    **What this class guarantees, and what it cannot.** Req 18.2 requires the
    concatenation of what was written to equal the final answer the loop
    returns. The loop's final answer is the text of the first reply that
    requests no tool, so the tokens of *that* reply are the ones to write, and
    the tokens of a reply that asks for a tool must not be presented as an
    answer. The adapter stops forwarding the moment it sees a tool-call delta,
    which covers every reply that asks for a tool before saying anything and
    every reply that asks in the same chunk as it speaks.

    One case remains, and this class does not pretend otherwise: a reply that
    emits prose first and only then asks for a tool. Those tokens are already on
    the terminal by the time the tool request arrives -- they were written
    within a second of arriving, which is what streaming means -- and no code
    can recall them. :meth:`retract` is what honesty looks like there: the
    stream is closed, a line on stderr says the text above was not the answer,
    and the streamer resets so a later step can still stream the real one. The
    CLI then prints the loop's answer in its own single write, so the answer the
    operator reads under the heading is always the loop's.

    The streamer never decides what the answer *is*; it reports what it wrote,
    and the CLI compares that with ``RunResult.final_answer``.
    """

    __slots__ = ("_renderer", "_parts", "_open", "_wrote", "_retractions", "_supported")

    def __init__(self, renderer: Renderer | PlainRenderer) -> None:
        self._renderer = renderer
        self._parts: list[str] = []
        self._open = False
        self._wrote = False
        self._retractions = 0
        # A renderer that does not offer the fragment surface -- a test double,
        # or a future renderer -- degrades to no streaming at all rather than
        # raising. `streamed` is then empty, the CLI's comparison fails, and the
        # answer is printed by the ordinary single write.
        self._supported = all(
            callable(getattr(renderer, name, None))
            for name in ("begin_stream", "stream_fragment", "end_stream")
        )

    # -- the sink the adapter calls ------------------------------------------

    def token(self, text: str) -> None:
        """Write one received output token (Req 18.2)."""
        if not text or not self._supported:
            return
        if not self._open:
            self._renderer.begin_stream()  # type: ignore[union-attr]
            self._open = True
        self._parts.append(text)
        self._wrote = True
        self._renderer.stream_fragment(text)  # type: ignore[union-attr]

    def retract(self) -> None:
        """Disown what was streamed for a reply that then requested a tool."""
        self._retractions += 1
        if not self._open:
            self._parts.clear()
            self._wrote = False
            return
        self._renderer.end_stream()  # type: ignore[union-attr]
        self._open = False
        self._parts.clear()
        self._wrote = False
        self._renderer.warning(STREAM_RETRACTED_MESSAGE)

    # -- the CLI's side -------------------------------------------------------

    def finish(self) -> str:
        """Close an open stream and return everything streamed as the answer."""
        if self._open:
            self._renderer.end_stream()  # type: ignore[union-attr]
            self._open = False
        return self.streamed

    @property
    def streamed(self) -> str:
        """The concatenation of the tokens written as the answer (Req 18.2)."""
        return "".join(self._parts)

    @property
    def wrote_answer(self) -> bool:
        """Whether any token was written under the final-answer heading."""
        return self._wrote

    @property
    def retractions(self) -> int:
        """How many times streamed text turned out not to be the answer."""
        return self._retractions

    def __repr__(self) -> str:
        return (
            f"AnswerStreamer(wrote_answer={self._wrote}, "
            f"characters={len(self.streamed)}, retractions={self._retractions})"
        )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _normalise_call(
    call: ToolCall | str, arguments: Mapping[str, Any] | None
) -> tuple[str, Mapping[str, Any]]:
    """Accept either a ``ToolCall`` or a name and a mapping (Req 13.2)."""
    if isinstance(call, ToolCall):
        return call.name, call.arguments
    if isinstance(call, str):
        return call, arguments or {}
    # Duck-typed last resort: anything carrying a name and arguments narrates
    # correctly, which keeps a test double from having to be a real ToolCall.
    name = getattr(call, "name", None)
    if isinstance(name, str):
        return name, getattr(call, "arguments", None) or {}
    raise ValueError(
        f"tool_call expects a ToolCall or a tool name, received {type(call).__name__}"
    )


def _as_totals(totals: Totals | Any) -> tuple[Totals, Sequence[str]]:
    """Accept a snapshot or a cost tracker, returning the snapshot and warnings.

    The tracker is detected by its two reading methods rather than by its type,
    so the eval runner can pass an equivalent accumulator without importing
    :class:`~agent_lab.cost.CostTracker`.
    """
    if isinstance(totals, Totals):
        return totals, ()
    snapshot = getattr(totals, "totals", None)
    if callable(snapshot):
        resolved = snapshot()
        if not isinstance(resolved, Totals):
            raise ValueError(
                f"totals() must return a Totals, received {type(resolved).__name__}"
            )
        warnings = getattr(totals, "warnings", None)
        return resolved, tuple(warnings()) if callable(warnings) else ()
    raise ValueError(
        f"run_summary expects a Totals or a cost tracker, received {type(totals).__name__}"
    )


__all__ = [
    "DEFAULT_DISPLAY_MAX_CHARS",
    "DISPLAY_LIMIT_VAR",
    "DISPLAY_MAX_CHARS",
    "DISPLAY_MIN_CHARS",
    "FAILURE_MARKER",
    "FINAL_ANSWER_HEADING",
    "INDENT",
    "RICH_STYLES",
    "STREAM_RETRACTED_MESSAGE",
    "SUCCESS_MARKER",
    "SUMMARY_PREFIX",
    "AnswerStreamer",
    "DisplayLimitError",
    "PlainRenderer",
    "Renderer",
    "RichRenderer",
    "build_renderer",
    "format_arguments",
    "format_final_answer",
    "format_run_summary",
    "format_step_heading",
    "format_tool_call",
    "format_tool_result",
    "rich_available",
    "shorten",
    "split_on_partial_secret",
    "validate_display_max_chars",
]
