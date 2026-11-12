"""Unit tests for `trace.py`, the step trace sink (Req 16.11).

The five cases Req 16.11 names are covered here: a step that requested one tool,
a step that requested more than one, a step that requested none (Req 11.6), an
absent trace file whose parent directory does not exist either (Req 11.10), and
a failing append that warns and lets the run go on (Req 11.11). The line shape
Req 11.1 fixes -- one JSON object, exactly one trailing newline, append only --
is asserted alongside them, since every one of those cases depends on it.

Property 11 (line count equals step count across several runs sharing one file)
belongs to `tests/test_properties.py` and is not duplicated here.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_lab.model_client import ToolCall, Usage
from agent_lab.redaction import REDACTION_MARKER
from agent_lab.tools.base import ToolResult
from agent_lab.trace import (
    CALL_FIELDS,
    RECORD_FIELDS,
    TRACE_MAX_CHARS,
    JsonlTraceLogger,
    TraceRecord,
    TracedCall,
)

STARTED_AT = datetime(2026, 3, 4, 18, 22, 41, 507_000, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def dispatched(index: int, name: str, arguments: dict, content: str, status: str = "success"):
    """One `TracedCall` built the way the loop builds it, from a call and a result."""
    return TracedCall.from_dispatch(
        index,
        ToolCall(id=f"call-1-{index}", name=name, arguments=arguments),
        ToolResult(status=status, content=content),
    )


def logger_with_warnings(path: Path, **kwargs) -> tuple[JsonlTraceLogger, io.StringIO]:
    """A logger whose warnings are captured, with construction-time ones discarded.

    Construction can warn on its own (Req 11.10 failing), and a test about
    appends should count only the warnings its appends produced.
    """
    warnings = io.StringIO()
    logger = JsonlTraceLogger(path, stream=warnings, **kwargs)
    warnings.seek(0)
    warnings.truncate(0)
    return logger, warnings


def lines_of(path: Path) -> list[str]:
    """The trace file's lines, asserting the terminator Req 11.1 asks for."""
    raw = path.read_bytes()
    assert raw.endswith(b"\n"), "every record ends with a newline"
    assert b"\r\n" not in raw, "exactly one newline character, not a CRLF pair"
    assert not raw.endswith(b"\n\n"), "exactly one newline per record"
    return raw.decode("utf-8").rstrip("\n").split("\n")


def records_in(path: Path) -> list[dict]:
    """Every line of the trace file parsed as one JSON object (Req 11.9)."""
    return [json.loads(line) for line in lines_of(path)]


# --------------------------------------------------------------------------- #
# A step that requested one tool (Req 16.11, Req 11.1, Req 11.2)
# --------------------------------------------------------------------------- #


def test_single_tool_step_writes_one_line_with_the_recorded_shape(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path, run_id="run-abc")

    logger.append(
        TraceRecord(
            step=1,
            started_at=STARTED_AT,
            duration_ms=1234,
            calls=(dispatched(0, "calculator", {"a": 1843, "b": 27, "operation": "multiply"},
                              "49761"),),
            assistant_text="Let me multiply those.",
            usage=Usage(input_tokens=120, output_tokens=18),
            cost_usd=0.0001234567,
        )
    )

    records = records_in(path)
    assert len(records) == 1

    record = records[0]
    assert tuple(record) == RECORD_FIELDS
    assert record["run_id"] == "run-abc"
    assert record["step"] == 1
    assert record["started_at"] == "2026-03-04T18:22:41.507Z"
    assert record["duration_ms"] == 1234
    assert record["tool_name"] == "calculator"
    assert record["tool_input"] == {"a": 1843, "b": 27, "operation": "multiply"}
    assert record["tool_result"] == "49761"
    assert record["assistant_text"] == "Let me multiply those."
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 18
    assert record["tokens_estimated"] is False
    assert record["cost_usd"] == 0.000123  # 6 decimal places (Req 11.2)
    assert record["run_outcome"] is None

    assert len(record["tool_calls"]) == 1
    call = record["tool_calls"][0]
    assert tuple(call) == CALL_FIELDS
    assert call["index"] == 0
    assert call["name"] == "calculator"
    assert call["result"] == "49761"
    assert call["status"] == "success"
    assert call["truncated"] is False


def test_run_outcome_is_recorded_on_the_final_record_only(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)

    logger.append(TraceRecord(step=1, calls=(dispatched(0, "read_notes", {}, "no notes yet"),)))
    logger.append(TraceRecord(step=2, assistant_text="Done.", run_outcome="final_answer"))

    first, final = records_in(path)
    assert first["run_outcome"] is None
    assert final["run_outcome"] == "final_answer"
    # Req 11.3: one run id on every record of the run.
    assert first["run_id"] == final["run_id"] == logger.run_id


# --------------------------------------------------------------------------- #
# A step that requested more than one tool (Req 16.11, Req 11.7)
# --------------------------------------------------------------------------- #


def test_multi_tool_step_collapses_into_one_line_in_dispatch_order(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)

    logger.append(
        TraceRecord(
            step=3,
            started_at=STARTED_AT,
            calls=(
                dispatched(0, "search_docs", {"query": "retention"}, "chunk one"),
                dispatched(1, "calculator", {"a": 2, "b": 3, "operation": "add"}, "5"),
                dispatched(2, "read_notes", {}, "boom", status="error"),
            ),
            assistant_text="Three things at once.",
        )
    )

    # One step, one line, whatever happened inside it (Req 11.1, Req 11.7).
    records = records_in(path)
    assert len(records) == 1
    record = records[0]

    calls = record["tool_calls"]
    assert [call["name"] for call in calls] == ["search_docs", "calculator", "read_notes"]
    assert [call["index"] for call in calls] == [0, 1, 2]
    assert [call["result"] for call in calls] == ["chunk one", "5", "boom"]
    assert [call["status"] for call in calls] == ["success", "success", "error"]

    # The mirrored top-level fields report the first call of the step.
    assert record["tool_name"] == "search_docs"
    assert record["tool_input"] == {"query": "retention"}
    assert record["tool_result"] == "chunk one"


def test_repeated_identical_calls_are_each_recorded(tmp_path):
    # Req 11.7 records every request of the step; nothing is deduplicated.
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)
    arguments = {"a": 2, "b": 2, "operation": "add"}

    logger.append(
        TraceRecord(
            step=1,
            calls=(
                dispatched(0, "calculator", arguments, "4"),
                dispatched(1, "calculator", arguments, "4"),
            ),
        )
    )

    (record,) = records_in(path)
    assert len(record["tool_calls"]) == 2
    assert [call["index"] for call in record["tool_calls"]] == [0, 1]


def test_per_call_approvals_are_kept_for_every_write_of_the_step(tmp_path):
    # Two writes in one step took two decisions; a single field could hold one.
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)

    logger.append(
        TraceRecord(
            step=1,
            calls=(
                TracedCall.from_dispatch(
                    0,
                    ToolCall(id="c0", name="save_note", arguments={"text": "first"}),
                    ToolResult(status="success", content="appended 5 characters",
                               approval="granted"),
                ),
                TracedCall.from_dispatch(
                    1,
                    ToolCall(id="c1", name="save_note", arguments={"text": "second"}),
                    ToolResult(status="error", content="the write was declined",
                               approval="declined"),
                ),
            ),
        )
    )

    (record,) = records_in(path)
    assert [call["approval"] for call in record["tool_calls"]] == ["granted", "declined"]
    assert record["approval"] == "granted"


# --------------------------------------------------------------------------- #
# A step that requested no tool (Req 16.11, Req 11.6)
# --------------------------------------------------------------------------- #


def test_no_tool_step_records_a_null_tool_name_and_the_assistant_text(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)

    logger.append(
        TraceRecord(
            step=4,
            started_at=STARTED_AT,
            duration_ms=812,
            assistant_text="1843 multiplied by 27 is 49,761.",
            usage=Usage(input_tokens=90, output_tokens=12, estimated=True),
            run_outcome="final_answer",
        )
    )

    (record,) = records_in(path)
    assert record["tool_name"] is None
    assert record["tool_input"] == {}
    assert record["tool_result"] is None
    assert record["tool_calls"] == []
    assert record["assistant_text"] == "1843 multiplied by 27 is 49,761."
    assert record["tokens_estimated"] is True
    assert record["approval"] is None


def test_no_tool_step_truncates_long_assistant_text_with_a_marker(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)
    text = "y" * (TRACE_MAX_CHARS + 500)

    logger.append(TraceRecord(step=1, assistant_text=text))

    (record,) = records_in(path)
    recorded = record["assistant_text"]
    assert recorded.startswith("y" * TRACE_MAX_CHARS)
    assert len(recorded) > TRACE_MAX_CHARS  # the marker is appended
    assert str(TRACE_MAX_CHARS + 500) in recorded  # the pre-truncation count
    assert "truncated" in recorded


def test_values_containing_newlines_do_not_split_the_record(tmp_path):
    # Req 11.9: an embedded newline is escaped, so one step is still one line.
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)

    logger.append(
        TraceRecord(
            step=1,
            calls=(dispatched(0, "read_notes", {}, "first\n\nsecond\r\nthird"),),
            assistant_text='a "quoted" \\ line\nand another\n',
        )
    )

    assert len(lines_of(path)) == 1
    (record,) = records_in(path)
    assert record["tool_result"] == "first\n\nsecond\r\nthird"
    assert record["assistant_text"] == 'a "quoted" \\ line\nand another\n'


# --------------------------------------------------------------------------- #
# File and parent directory creation (Req 16.11, Req 11.10)
# --------------------------------------------------------------------------- #


def test_file_and_missing_parent_directories_are_created_at_run_start(tmp_path):
    path = tmp_path / "traces" / "nested" / "run.jsonl"
    assert not path.parent.exists()

    logger = JsonlTraceLogger(path)

    # Created at construction -- that is run start, before the first record.
    assert path.parent.is_dir()
    assert path.is_file()
    assert path.read_bytes() == b""
    assert logger.path == path

    logger.append(TraceRecord(step=1, assistant_text="hello"))
    assert len(records_in(path)) == 1


def test_an_existing_trace_file_is_appended_to_and_never_rewritten(tmp_path):
    # Req 11.1: lines already present -- including an earlier run's -- survive.
    path = tmp_path / "run.jsonl"
    earlier = '{"run_id": "earlier-run", "step": 1}'
    path.write_text(earlier + "\n", encoding="utf-8", newline="\n")

    first = JsonlTraceLogger(path, run_id="run-one")
    first.append(TraceRecord(step=1, assistant_text="one"))
    second = JsonlTraceLogger(path, run_id="run-two")
    second.append(TraceRecord(step=1, assistant_text="two"))

    lines = lines_of(path)
    assert len(lines) == 3
    assert lines[0] == earlier
    assert [json.loads(line)["run_id"] for line in lines] == [
        "earlier-run",
        "run-one",
        "run-two",
    ]


def test_each_appended_record_is_its_own_line(tmp_path):
    path = tmp_path / "run.jsonl"
    logger = JsonlTraceLogger(path)

    logger.append_all(TraceRecord(step=n, assistant_text=f"step {n}") for n in (1, 2, 3))

    records = records_in(path)
    assert [record["step"] for record in records] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# A failing append (Req 16.11, Req 11.11)
# --------------------------------------------------------------------------- #


@pytest.fixture
def blocked_path(tmp_path):
    """A trace path occupied by a directory, so every append to it fails."""
    path = tmp_path / "run.jsonl"
    path.mkdir()
    return path


def test_failing_append_prints_one_warning_naming_the_path_and_returns(blocked_path):
    logger, warnings = logger_with_warnings(blocked_path)

    logger.append(TraceRecord(step=2, assistant_text="still answered"))

    printed = warnings.getvalue()
    assert printed.count("\n") == 1, f"exactly one warning, got: {printed!r}"
    assert str(blocked_path) in printed
    assert "step 2" in printed
    assert "discarded" in printed and "continues" in printed


def test_a_run_continues_across_repeated_append_failures(blocked_path):
    logger, warnings = logger_with_warnings(blocked_path)

    # Every step of the run is still attempted and none of them raises: the
    # sink cannot change the outcome the loop reaches (Req 11.11).
    for step in (1, 2, 3):
        logger.append(TraceRecord(step=step, assistant_text=f"step {step}"))

    assert warnings.getvalue().count("\n") == 3
    assert logger.run_id  # the logger is still usable


def test_file_creation_failure_warns_instead_of_raising(tmp_path):
    # Req 11.10 cannot be satisfied when a regular file occupies the configured
    # parent directory; construction must still not end the run.
    occupied = tmp_path / "traces"
    occupied.write_text("not a directory", encoding="utf-8")
    path = occupied / "run.jsonl"
    warnings = io.StringIO()

    logger = JsonlTraceLogger(path, stream=warnings)

    printed = warnings.getvalue()
    assert printed.count("\n") == 1
    assert str(path) in printed
    assert "could not be created" in printed
    assert logger.path == path


def test_an_unserialisable_record_is_discarded_with_one_warning(tmp_path):
    # The loop filled in `started_at` wrongly. One warning, record dropped,
    # nothing raised, and no partial line written (Req 11.11).
    path = tmp_path / "run.jsonl"
    logger, warnings = logger_with_warnings(path)

    logger.append(TraceRecord(step=7, started_at=12345, assistant_text="lost"))

    printed = warnings.getvalue()
    assert printed.count("\n") == 1
    assert str(path) in printed
    assert "serialised" in printed
    assert path.read_bytes() == b""

    # The next, well-formed record is still written.
    logger.append(TraceRecord(step=8, assistant_text="kept"))
    (record,) = records_in(path)
    assert record["step"] == 8


def test_a_failing_warning_stream_does_not_end_the_run(blocked_path):
    class ClosedStream(io.StringIO):
        def write(self, _text):
            raise OSError("stream is closed")

    logger = JsonlTraceLogger(blocked_path, stream=ClosedStream())
    logger.append(TraceRecord(step=1, assistant_text="still answered"))  # must not raise


# --------------------------------------------------------------------------- #
# Redaction (Req 11.5)
# --------------------------------------------------------------------------- #


def test_the_api_key_never_appears_in_a_written_line(tmp_path):
    path = tmp_path / "run.jsonl"
    key = "sk-live-0123456789"
    logger = JsonlTraceLogger(path, api_key=key)

    logger.append(
        TraceRecord(
            step=1,
            calls=(dispatched(0, "save_note", {"text": f"my key is {key}"},
                              f"authentication failed for {key}", status="error"),),
            assistant_text=f"The key {key} did not work.",
        )
    )

    raw = path.read_text(encoding="utf-8")
    assert key not in raw

    (record,) = records_in(path)
    # Every place the key reached: the tool input, the result, the assistant
    # text, and the mirrored copies of the first two.
    assert REDACTION_MARKER in record["assistant_text"]
    assert REDACTION_MARKER in record["tool_input"]["text"]
    assert REDACTION_MARKER in record["tool_result"]
    assert REDACTION_MARKER in record["tool_calls"][0]["input"]["text"]
    assert REDACTION_MARKER in record["tool_calls"][0]["result"]


def test_the_api_key_is_redacted_from_a_warning(blocked_path):
    key = "sk-live-9876543210"
    logger, warnings = logger_with_warnings(blocked_path, api_key=key)

    logger.append(TraceRecord(step=1, assistant_text=f"contains {key}"))

    assert key not in warnings.getvalue()
