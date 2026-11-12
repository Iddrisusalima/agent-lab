"""Unit tests for the tool dispatcher (Req 16.6, Req 5.4 - Req 5.9, Req 5.12).

Every test here goes through `ToolDispatcher.dispatch` and asserts on the
returned `ToolResult`, because the one promise the dispatcher makes is that a
result comes back for any call at all (Req 5.9). Stand-in tools keep the suite
independent of the four shipped tools and of any configuration: a spec and an
`execute` is the whole contract.
"""

import threading
import time

import pytest

from agent_lab.tools.base import ToolResult, ToolSpec
from agent_lab.tools.dispatcher import TRUNCATION_MARKER, ToolDispatcher
from agent_lab.tools.registry import ToolRegistry

# --------------------------------------------------------------------------- #
# Stand-in tools
# --------------------------------------------------------------------------- #


class EchoTool:
    """Declares one required string and one optional integer, and succeeds."""

    def __init__(self, name: str = "echo") -> None:
        self.spec = ToolSpec(
            name=name,
            description="Echo the arguments it was given back to the caller.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["text"],
            },
            timeout_seconds=10.0,
        )
        self.calls: list[dict] = []

    def execute(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(status="success", content=f"echo: {kwargs.get('text', '')}")


class RaisingTool:
    """Raises whatever it was constructed with, as a tool would on a bug."""

    def __init__(self, error: BaseException) -> None:
        self.spec = ToolSpec(
            name="raiser",
            description="Raise an exception instead of returning a result.",
            timeout_seconds=10.0,
        )
        self._error = error

    def execute(self, **kwargs) -> ToolResult:
        raise self._error


class BlockingTool:
    """Blocks until the test releases it, so a timeout can be observed."""

    def __init__(self, release: threading.Event, timeout_seconds: float | None = 1.0) -> None:
        self.spec = ToolSpec(
            name="blocker",
            description="Block until the caller releases it.",
            timeout_seconds=timeout_seconds,
        )
        self._release = release
        self.finished = threading.Event()

    def execute(self, **kwargs) -> ToolResult:
        # A bounded wait, so the worker thread cannot outlive the test session
        # even if the release is never set.
        self._release.wait(30)
        self.finished.set()
        return ToolResult(status="success", content="finished")


class LongResultTool:
    """Returns more content than any configured limit will allow."""

    def __init__(self, length: int) -> None:
        self.spec = ToolSpec(
            name="long",
            description="Return a long block of content.",
            timeout_seconds=10.0,
        )
        self._length = length

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(status="success", content="x" * self._length)


@pytest.fixture
def echo() -> EchoTool:
    return EchoTool()


@pytest.fixture
def dispatcher(echo: EchoTool) -> ToolDispatcher:
    # Several tools registered, so the unknown-name message has a real list to
    # quote back (Req 5.4).
    registry = ToolRegistry([echo, EchoTool("calculator"), EchoTool("read_notes")])
    return ToolDispatcher(registry)


# --------------------------------------------------------------------------- #
# Req 5.4: an unknown tool name
# --------------------------------------------------------------------------- #


def test_unknown_name_is_an_error_listing_the_registered_names(dispatcher):
    result = dispatcher.dispatch({"name": "calculater", "arguments": {}})

    assert result.is_error
    assert "calculater" in result.content
    assert "unknown" in result.content.lower()
    for registered in ("echo", "calculator", "read_notes"):
        assert registered in result.content


def test_a_case_variant_of_a_registered_name_is_unknown(dispatcher, echo):
    # Req 5.3 resolves case-sensitively and exactly, so `Echo` is not `echo`.
    result = dispatcher.dispatch({"name": "Echo", "arguments": {"text": "hi"}})

    assert result.is_error
    assert "Echo" in result.content
    assert echo.calls == []


# --------------------------------------------------------------------------- #
# Req 5.5: a missing required parameter, and a wrong-typed value
# --------------------------------------------------------------------------- #


def test_missing_required_parameter_names_it_with_type_and_required_status(dispatcher, echo):
    result = dispatcher.dispatch({"name": "echo", "arguments": {"count": 2}})

    assert result.is_error
    assert "text" in result.content
    assert "missing" in result.content
    assert "required" in result.content
    assert "'string'" in result.content
    # Req 5.3: no tool code runs when validation fails.
    assert echo.calls == []


def test_a_call_with_no_arguments_at_all_reports_the_missing_parameter(dispatcher, echo):
    # A call object without an `arguments` field is read as an empty mapping,
    # so the required parameter is reported rather than the payload shape.
    result = dispatcher.dispatch({"name": "echo"})

    assert result.is_error
    assert "text" in result.content
    assert "missing" in result.content
    assert echo.calls == []


def test_wrong_typed_parameter_names_it_with_what_arrived_and_what_was_declared(dispatcher, echo):
    result = dispatcher.dispatch({"name": "echo", "arguments": {"text": 7}})

    assert result.is_error
    assert "text" in result.content
    assert "integer" in result.content  # what arrived
    assert "'string'" in result.content  # what the schema declares
    assert echo.calls == []


def test_a_boolean_is_not_accepted_for_an_integer_parameter(dispatcher, echo):
    # `True` is an `int` in Python; the schema said integer, so it is a fault.
    result = dispatcher.dispatch({"name": "echo", "arguments": {"text": "hi", "count": True}})

    assert result.is_error
    assert "count" in result.content
    assert "boolean" in result.content
    assert echo.calls == []


# --------------------------------------------------------------------------- #
# Req 5.6: an undeclared parameter name
# --------------------------------------------------------------------------- #


def test_undeclared_parameter_names_it_and_lists_the_declared_names(dispatcher, echo):
    result = dispatcher.dispatch(
        {"name": "echo", "arguments": {"text": "hi", "path": "/tmp/notes.md"}}
    )

    assert result.is_error
    assert "'path'" in result.content
    assert "Undeclared" in result.content
    assert "text" in result.content
    assert "count" in result.content
    assert echo.calls == []


def test_all_three_argument_faults_are_reported_together(dispatcher, echo):
    # One correction round trip rather than three: a missing required
    # parameter, a wrong-typed value and an undeclared name in one message.
    result = dispatcher.dispatch({"name": "echo", "arguments": {"count": "two", "extra": 1}})

    assert result.is_error
    assert "text" in result.content and "missing" in result.content
    assert "count" in result.content
    assert "'extra'" in result.content
    assert echo.calls == []


# --------------------------------------------------------------------------- #
# Req 5.7: a tool that raises
# --------------------------------------------------------------------------- #


def test_a_raising_tool_becomes_an_error_naming_the_type_and_the_message():
    registry = ToolRegistry([RaisingTool(ValueError("the corpus index is missing"))])
    result = ToolDispatcher(registry).dispatch({"name": "raiser"})

    assert result.is_error
    assert "ValueError" in result.content
    assert "the corpus index is missing" in result.content


def test_no_exception_escapes_dispatch_when_a_tool_raises():
    # Req 5.9 states this directly: whatever the tool does, the loop gets a
    # result. `KeyboardInterrupt` is a `BaseException`, which is the case a
    # plain `except Exception` inside the dispatcher would have missed.
    for error in (RuntimeError("boom"), KeyboardInterrupt(), MemoryError()):
        registry = ToolRegistry([RaisingTool(error)])
        result = ToolDispatcher(registry).dispatch({"name": "raiser"})
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert type(error).__name__ in result.content


def test_a_tool_returning_the_wrong_type_becomes_an_error():
    class WrongReturn:
        spec = ToolSpec(name="wrong", description="Return the wrong type.", timeout_seconds=10.0)

        def execute(self, **kwargs):
            return "not a ToolResult"

    result = ToolDispatcher(ToolRegistry([WrongReturn()])).dispatch({"name": "wrong"})

    assert result.is_error
    assert "ToolResult" in result.content


# --------------------------------------------------------------------------- #
# Req 5.8: a tool that blocks past its configured timeout
# --------------------------------------------------------------------------- #


def test_a_tool_blocking_past_its_timeout_is_abandoned_with_the_timeout_named():
    release = threading.Event()
    tool = BlockingTool(release, timeout_seconds=1.0)
    dispatcher = ToolDispatcher(ToolRegistry([tool]))

    started = time.perf_counter()
    try:
        result = dispatcher.dispatch({"name": "blocker"})
    finally:
        # Let the abandoned worker thread finish, so it does not outlive the test.
        release.set()
    elapsed = time.perf_counter() - started

    assert result.is_error
    assert "timeout of 1 seconds" in result.content
    assert "abandoned" in result.content
    # It waited for the configured second, and it did not wait for the tool.
    assert 0.9 <= elapsed < 10


def test_a_tool_declaring_no_timeout_runs_inline_with_no_limit():
    # Req 9.2 in the making: interactive `save_note` waits on the operator, so
    # `timeout_seconds=None` must mean "no limit", not "default limit".
    release = threading.Event()
    release.set()
    tool = BlockingTool(release, timeout_seconds=None)
    main_thread = threading.current_thread().name
    observed: list[str] = []

    original_execute = tool.execute

    def execute(**kwargs):
        observed.append(threading.current_thread().name)
        return original_execute(**kwargs)

    tool.execute = execute  # type: ignore[method-assign]

    result = ToolDispatcher(ToolRegistry([tool])).dispatch({"name": "blocker"})

    assert result.status == "success"
    assert observed == [main_thread]


# --------------------------------------------------------------------------- #
# Req 5.9: nothing escapes `dispatch`, whatever the call looks like
# --------------------------------------------------------------------------- #


class HostileCall:
    """A call object whose fields raise when read."""

    @property
    def name(self):
        raise RuntimeError("reading the name failed")

    @property
    def arguments(self):
        raise RuntimeError("reading the arguments failed")


@pytest.mark.parametrize(
    "call",
    [
        None,
        "echo",
        42,
        [],
        {},
        {"name": None},
        {"name": 17, "arguments": {"text": "hi"}},
        {"name": "echo", "arguments": "text=hi"},
        {"name": "echo", "arguments": ["text", "hi"]},
        {"name": "echo", "arguments": {1: "hi"}},
        {"name": "echo", "arguments": {"text": "hi"}, "arguments_error": "invalid JSON"},
        HostileCall(),
    ],
    ids=[
        "none",
        "bare-string",
        "integer",
        "empty-list",
        "empty-dict",
        "null-name",
        "non-string-name",
        "string-arguments",
        "list-arguments",
        "non-string-argument-key",
        "arguments-parse-failure",
        "fields-that-raise",
    ],
)
def test_dispatch_returns_a_result_and_raises_nothing(dispatcher, call):
    result = dispatcher.dispatch(call)

    assert isinstance(result, ToolResult)
    assert result.status in ("success", "error")
    assert result.content


def test_a_well_formed_call_still_reaches_the_tool(dispatcher, echo):
    # The counterpart to the refusals above: validation is not refusing
    # everything.
    result = dispatcher.dispatch({"name": "echo", "arguments": {"text": "hi", "count": 2}})

    assert result.status == "success"
    assert "hi" in result.content
    assert echo.calls == [{"text": "hi", "count": 2}]


def test_dispatch_is_also_callable_directly(dispatcher):
    assert dispatcher({"name": "echo", "arguments": {"text": "hi"}}).status == "success"


# --------------------------------------------------------------------------- #
# Req 5.10, Req 5.12: the content the loop receives
# --------------------------------------------------------------------------- #


def test_the_configured_secret_is_absent_from_an_error_result():
    secret = "sk-test-0123456789"
    registry = ToolRegistry([RaisingTool(ValueError(f"request rejected for key {secret}"))])
    result = ToolDispatcher(registry, secret=secret).dispatch({"name": "raiser"})

    assert result.is_error
    assert secret not in result.content


def test_over_long_content_is_truncated_with_a_marker_stating_the_original_length():
    limit = 500
    registry = ToolRegistry([LongResultTool(1_500)])
    result = ToolDispatcher(registry, max_result_chars=limit).dispatch({"name": "long"})

    assert result.status == "success"
    assert result.truncated
    assert result.original_length == 1_500
    assert result.content == "x" * limit + TRUNCATION_MARKER.format(original=1_500, limit=limit)
