"""Unit tests for `render.py`, the terminal output path (Req 16.1, Req 13).

The six cases task 6.4 names are covered here: the plain and the `rich`
renderer emit the same field set (Req 13.6), a display limit outside 80-10,000
is rejected naming the range (Req 13.3), a result longer than the limit carries
a truncation indicator (Req 13.3), an error result carries a textually distinct
failure marker and the run goes on (Req 13.7), narration goes to stdout while
warnings and errors go to stderr (Req 13.8), and the provider API key value
appears in neither stream (Req 13.9).

Streams are injected rather than captured from the process. `PlainRenderer`
resolves its streams at write time, so passing an `io.StringIO` captures
exactly what a terminal would have received, with no global state replaced and
no interference between the two renderers compared in one test.
"""

from __future__ import annotations

import io

import pytest

from agent_lab.config import DEFAULTS, INT_RANGES, ConfigError, load_config
from agent_lab.cost import CostTracker, Totals
from agent_lab.model_client import ToolCall, Usage
from agent_lab.pricing import ModelPrice, PriceTable
from agent_lab.redaction import REDACTION_MARKER
from agent_lab.render import (
    DEFAULT_DISPLAY_MAX_CHARS,
    DISPLAY_LIMIT_VAR,
    DISPLAY_MAX_CHARS,
    DISPLAY_MIN_CHARS,
    FAILURE_MARKER,
    FINAL_ANSWER_HEADING,
    SUCCESS_MARKER,
    SUMMARY_PREFIX,
    DisplayLimitError,
    PlainRenderer,
    RichRenderer,
    build_renderer,
    format_run_summary,
    rich_available,
    shorten,
)
from agent_lab.tools.base import ToolResult

# A key-shaped string, long enough and distinctive enough that a surviving
# fragment in either stream would be unambiguous (Req 13.9).
SECRET = "sk-test-4b19c0ffee0123456789abcdef"

# Both renderer classes, so every field-set, marker and redaction assertion is
# made against each rendering rather than only against the plain one (Req 13.6).
RENDERERS = [PlainRenderer, RichRenderer]
RENDERER_IDS = ["plain", "rich"]

TOTALS = Totals(steps=3, input_tokens=4210, output_tokens=388, usd=0.0137)

# The exact Req 3.9 line, two spaces between fields, asserted literally rather
# than reassembled from the same f-string the implementation uses.
SUMMARY_LINE = (
    "Run summary: steps=3  input_tokens=4210  output_tokens=388  "
    "total_tokens=4598  cost=$0.0137"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def streams(cls, **kwargs) -> tuple[PlainRenderer, io.StringIO, io.StringIO]:
    """A renderer of class ``cls`` writing to two fresh buffers."""
    out, err = io.StringIO(), io.StringIO()
    kwargs.setdefault("api_key", SECRET)
    renderer = cls(stdout=out, stderr=err, **kwargs)
    return renderer, out, err


def run_script(cls, **kwargs) -> tuple[str, str]:
    """Drive one renderer through the whole Req 13 surface, in loop order.

    One step with a successful tool, a second step with a failing one, the
    final answer, the run summary, then a warning and an error. Returns what
    reached stdout and what reached stderr.
    """
    renderer, out, err = streams(cls, **kwargs)
    renderer.step_start(1)
    renderer.tool_call(
        ToolCall(
            id="call-1-0",
            name="calculator",
            arguments={"operation": "add", "a": 2, "b": 3},
        )
    )
    renderer.tool_result(ToolResult(status="success", content="5"))
    renderer.step_start(2)
    renderer.tool_call("search_docs", {"query": "retention policy", "k": 3})
    renderer.tool_result(ToolResult(status="error", content="corpus directory not found"))
    renderer.final_answer("The retention period is 30 days.\nSource: policy.md")
    renderer.run_summary(TOTALS)
    renderer.warning("Warning: token counts were unavailable for 1 model call.")
    renderer.error("The run stopped after reaching MAX_STEPS.")
    return out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Req 13.6: the plain and the rich rendering carry the same fields
# --------------------------------------------------------------------------- #


def test_plain_and_rich_write_identical_text() -> None:
    """The two renderings differ in colour only, so to a buffer they are equal.

    `rich` writes no escape sequences to a non-terminal file, and the renderer
    hands it finished lines with markup, highlighting and emoji off. What is
    left is the plain renderer's characters, which is the strongest available
    form of "the same step numbers, tool names, tool inputs, shortened results,
    markers, final answer and summary fields" (Req 13.6).
    """
    assert rich_available(), "the dev extra installs rich; the rich path must be exercised"

    plain_out, plain_err = run_script(PlainRenderer, display_max_chars=80)
    rich_out, rich_err = run_script(RichRenderer, display_max_chars=80)

    assert rich_out == plain_out
    assert rich_err == plain_err


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_both_renderings_carry_every_req_13_6_field(cls) -> None:
    """Step numbers, tool names, inputs, results, markers, answer, summary."""
    out, _ = run_script(cls)

    assert "Step 1" in out
    assert "Step 2" in out

    assert "calculator" in out
    assert "search_docs" in out

    # Tool inputs, rendered as JSON.
    assert '"operation": "add"' in out
    assert '"query": "retention policy"' in out

    # Both results, each behind its own marker.
    assert f"{SUCCESS_MARKER} 5" in out
    assert f"{FAILURE_MARKER} corpus directory not found" in out

    # The final answer under its labelled heading, not shortened.
    assert FINAL_ANSWER_HEADING in out
    assert "The retention period is 30 days." in out
    assert "Source: policy.md" in out

    # Every summary field (Req 3.9).
    assert SUMMARY_LINE in out


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_final_answer_then_summary_in_that_order(cls) -> None:
    """Req 13.4: the summary follows the answer, and narration precedes both."""
    out, _ = run_script(cls)
    assert out.index("Step 1") < out.index(FINAL_ANSWER_HEADING) < out.index(SUMMARY_PREFIX)


def test_build_renderer_can_force_the_plain_rendering() -> None:
    """The plain renderer is reachable even where `rich` is installed.

    Req 13.6 is about what an installation without `rich` sees; forcing the
    plain class is how the eval runner and these tests get that rendering
    without uninstalling the package.
    """
    assert type(build_renderer(prefer_rich=False, api_key=SECRET)) is PlainRenderer
    if rich_available():
        assert type(build_renderer(prefer_rich=True, api_key=SECRET)) is RichRenderer


def test_summary_formatter_matches_the_documented_line() -> None:
    """The pure formatter produces the exact Req 3.9 shape on its own."""
    assert format_run_summary(TOTALS) == SUMMARY_LINE


# --------------------------------------------------------------------------- #
# Req 13.3: the display limit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [79, 0, -1, 10_001, 1_000_000, "500", None, 500.0, True],
    ids=["below", "zero", "negative", "above", "far-above", "string", "none", "float", "bool"],
)
def test_out_of_range_display_limit_is_rejected(bad) -> None:
    """A value outside 80-10,000 raises, and the message names the range."""
    with pytest.raises(DisplayLimitError) as excinfo:
        PlainRenderer(display_max_chars=bad)

    message = str(excinfo.value)
    assert str(DISPLAY_MIN_CHARS) in message
    assert "10000" in message or "10,000" in message
    assert DISPLAY_LIMIT_VAR in message


def test_display_limit_error_is_a_config_error() -> None:
    """So the CLI's single configuration catch and exit code already cover it."""
    assert issubclass(DisplayLimitError, ConfigError)


@pytest.mark.parametrize("good", [DISPLAY_MIN_CHARS, 500, DISPLAY_MAX_CHARS])
def test_in_range_display_limit_is_accepted(good) -> None:
    assert PlainRenderer(display_max_chars=good).display_max_chars == good


def test_display_limit_defaults_to_500() -> None:
    assert PlainRenderer().display_max_chars == DEFAULT_DISPLAY_MAX_CHARS == 500


def test_renderer_and_config_agree_on_the_range_and_the_default() -> None:
    """The two modules name the same numbers; asserted rather than commented."""
    assert INT_RANGES[DISPLAY_LIMIT_VAR] == (DISPLAY_MIN_CHARS, DISPLAY_MAX_CHARS)
    assert DEFAULTS[DISPLAY_LIMIT_VAR] == str(DEFAULT_DISPLAY_MAX_CHARS)


def test_build_renderer_rejects_an_out_of_range_limit() -> None:
    with pytest.raises(DisplayLimitError):
        build_renderer(display_max_chars=12, api_key=SECRET)


# --- the truncation indicator ---------------------------------------------- #


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_long_tool_result_gets_a_truncation_indicator(cls) -> None:
    """A result longer than the limit is shortened and says so, with counts."""
    renderer, out, _ = streams(cls, display_max_chars=100)
    renderer.tool_result(ToolResult(status="success", content="y" * 250))

    line = out.getvalue()
    assert "truncated for display" in line
    assert "first 100 of 250 characters" in line
    # The limit governs the result text, not the indicator: 100 result
    # characters are kept and the indicator is appended after them.
    assert "y" * 100 in line
    assert "y" * 101 not in line


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_short_tool_result_gets_no_indicator(cls) -> None:
    """At or below the limit nothing is withheld, so nothing is announced."""
    renderer, out, _ = streams(cls, display_max_chars=80)
    renderer.tool_result(ToolResult(status="success", content="z" * 80))

    line = out.getvalue()
    assert "truncated for display" not in line
    assert "z" * 80 in line


def test_shorten_reports_whether_it_shortened() -> None:
    kept, cut = shorten("a" * 80, 80)
    assert (kept, cut) == ("a" * 80, False)

    kept, cut = shorten("a" * 81, 80)
    assert cut is True
    assert kept.startswith("a" * 80)
    assert "first 80 of 81 characters" in kept


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_tool_inputs_are_shortened_too(cls) -> None:
    """A multi-thousand-character argument must not bury the step it describes."""
    renderer, out, _ = streams(cls, display_max_chars=80)
    renderer.tool_call("save_note", {"text": "w" * 4000})

    line = out.getvalue()
    assert "save_note" in line
    assert "truncated for display" in line
    assert "w" * 4000 not in line


# --------------------------------------------------------------------------- #
# Req 13.7: the failure marker
# --------------------------------------------------------------------------- #


def test_the_two_markers_are_textually_distinct() -> None:
    """Distinct in more than case, so a case-folding pipe still shows two words."""
    assert SUCCESS_MARKER != FAILURE_MARKER
    assert SUCCESS_MARKER.casefold() != FAILURE_MARKER.casefold()
    assert SUCCESS_MARKER not in FAILURE_MARKER
    assert FAILURE_MARKER not in SUCCESS_MARKER


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_error_result_takes_the_failure_marker_and_the_run_continues(cls) -> None:
    """An error is a different marker and nothing else: step 2 still narrates."""
    renderer, out, err = streams(cls, display_max_chars=200)
    renderer.step_start(1)
    renderer.tool_result(ToolResult(status="error", content="division by zero"))
    renderer.step_start(2)
    renderer.tool_result(ToolResult(status="success", content="7"))
    renderer.final_answer("7")

    text = out.getvalue()
    assert f"{FAILURE_MARKER} division by zero" in text
    assert SUCCESS_MARKER not in text.split("Step 2")[0]
    assert f"{SUCCESS_MARKER} 7" in text
    assert text.index("Step 2") > text.index(FAILURE_MARKER)
    assert FINAL_ANSWER_HEADING in text
    # An error result is not a diagnostic: it is narration, so it stays on
    # stdout (Req 13.8).
    assert err.getvalue() == ""


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_long_error_result_carries_both_the_marker_and_the_indicator(cls) -> None:
    renderer, out, _ = streams(cls, display_max_chars=80)
    renderer.tool_result(ToolResult(status="error", content="E" * 300))

    line = out.getvalue()
    assert line.lstrip().startswith(FAILURE_MARKER)
    assert "truncated for display" in line


# --------------------------------------------------------------------------- #
# Req 13.8: the stream split
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_narration_goes_to_stdout_and_diagnostics_to_stderr(cls) -> None:
    out, err = run_script(cls)

    for narration in ("Step 1", "calculator", SUCCESS_MARKER, FINAL_ANSWER_HEADING, SUMMARY_LINE):
        assert narration in out
        assert narration not in err

    assert "token counts were unavailable" in err
    assert "token counts were unavailable" not in out
    assert "reaching MAX_STEPS" in err
    assert "reaching MAX_STEPS" not in out


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_summary_on_stdout_and_its_warnings_on_stderr(cls) -> None:
    """The tracker's own warning lines accompany the summary but are warnings."""
    tracker = CostTracker(prices=PriceTable({"m": ModelPrice("m", 1.0, 2.0, "2026-10-31")}))
    tracker.record(Usage(100, 50), "m")
    tracker.record(Usage(10, 5), "mystery-model-9000")

    renderer, out, err = streams(cls)
    renderer.run_summary(tracker)

    assert out.getvalue().startswith(SUMMARY_PREFIX)
    assert "steps=2" in out.getvalue()
    assert "mystery-model-9000" in err.getvalue()
    assert "mystery-model-9000" not in out.getvalue()


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_supplied_warnings_replace_the_trackers_own(cls) -> None:
    """The CLI adds the Req 1.10 access warnings alongside the cost ones."""
    renderer, out, err = streams(cls)
    renderer.run_summary(TOTALS, warnings=["Warning: DOCS_DIR is not readable."])

    assert SUMMARY_LINE in out.getvalue()
    assert err.getvalue() == "Warning: DOCS_DIR is not readable.\n"


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_suppressed_narration_still_prints_the_answer_and_the_summary(cls) -> None:
    """Req 13.4 says "regardless of whether step narration is suppressed"."""
    renderer, out, _ = streams(cls, narrate=False)
    renderer.step_start(1)
    renderer.tool_call("calculator", {"operation": "add"})
    renderer.tool_result(ToolResult(status="success", content="5"))
    renderer.final_answer("5")
    renderer.run_summary(TOTALS)

    text = out.getvalue()
    assert "Step 1" not in text
    assert "calculator" not in text
    assert text == f"{FINAL_ANSWER_HEADING}\n5\n{SUMMARY_LINE}\n"


# --------------------------------------------------------------------------- #
# Req 13.9: the key value reaches neither stream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_api_key_value_appears_in_neither_stream(cls) -> None:
    """Every write path redacts: narration, answer, summary, warning, error."""
    renderer, out, err = streams(cls, display_max_chars=500)
    renderer.step_start(1)
    renderer.tool_call("search_docs", {"query": f"what is {SECRET}"})
    renderer.tool_result(ToolResult(status="success", content=f"found {SECRET} in notes"))
    renderer.tool_result(ToolResult(status="error", content=f"auth failed for {SECRET}"))
    renderer.final_answer(f"The key is {SECRET}.")
    renderer.run_summary(TOTALS, warnings=[f"Warning: {SECRET} was rejected."])
    renderer.error(f"Request with {SECRET} failed.")

    stdout, stderr = out.getvalue(), err.getvalue()
    assert SECRET not in stdout
    assert SECRET not in stderr
    # Substituted in place rather than dropped, so the surrounding text still
    # reads (Req 1.11).
    # Four redacted stdout writes -- tool inputs, two results, the answer --
    # and two redacted stderr writes, the warning and the error.
    assert stdout.count(REDACTION_MARKER) == 4
    assert stderr.count(REDACTION_MARKER) == 2
    assert f"found {REDACTION_MARKER} in notes" in stdout
    assert f"Request with {REDACTION_MARKER} failed." in stderr


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_no_api_key_configured_leaves_text_untouched(cls) -> None:
    """An absent credential redacts nothing rather than shredding the output."""
    renderer, out, _ = streams(cls, api_key=None)
    renderer.final_answer("plain text")

    assert out.getvalue() == f"{FINAL_ANSWER_HEADING}\nplain text\n"


def test_from_config_takes_the_key_and_the_limit_from_the_configuration() -> None:
    """The CLI cannot wire a renderer with no secret (Req 13.9, Req 13.3)."""
    config = load_config(
        {"OPENAI_API_KEY": SECRET, "DISPLAY_MAX_CHARS": "120"},
        probe=False,
    )
    out, err = io.StringIO(), io.StringIO()
    renderer = PlainRenderer.from_config(config, stdout=out, stderr=err)

    assert renderer.display_max_chars == 120
    renderer.final_answer(f"leaked {SECRET}")
    assert SECRET not in out.getvalue()
    assert REDACTION_MARKER in out.getvalue()


# --------------------------------------------------------------------------- #
# A sink may not end a run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", RENDERERS, ids=RENDERER_IDS)
def test_a_closed_stream_does_not_raise(cls) -> None:
    """Narration is a report on the work, not the work (Req 13.1, Req 13.2)."""
    out, err = io.StringIO(), io.StringIO()
    out.close()
    err.close()
    renderer = cls(api_key=SECRET, stdout=out, stderr=err)

    renderer.step_start(1)
    renderer.tool_result(ToolResult(status="success", content="5"))
    renderer.final_answer("5")
    renderer.run_summary(TOTALS)
    renderer.error("boom")
