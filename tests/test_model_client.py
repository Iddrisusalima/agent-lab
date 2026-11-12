"""Unit tests for the provider adapter (Req 2.3, 2.5, 2.6, 2.7, 2.11, 2.13).

`OpenAIClient` takes its SDK client and its `sleep` callable as constructor
arguments, and these tests use both seams: every test below injects a scripted
fake SDK and a recording clock, so the suite makes no network call, needs no API
key, and waits no real seconds while asserting a seven-second retry schedule.

The autouse `no_real_sdk` fixture turns the "no network" claim into something
the suite enforces rather than something this docstring promises: it replaces
the one function that builds the real SDK client with a failing stand-in, so a
test that forgot to inject a fake fails loudly instead of reaching out.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from agent_lab.config import ConfigError, load_config
from agent_lab.model_client import (
    Message,
    ModelClientError,
    ModelReply,
    UnmatchedToolCallError,
    estimate_usage,
)
from agent_lab.providers import build_model_client
from agent_lab.providers import openai_client as adapter
from agent_lab.providers.openai_client import OpenAIClient

MODEL_ID = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a test agent."
FAKE_KEY = "sk-test-not-a-real-key"


# --------------------------------------------------------------------------- #
# Offline guard
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def no_real_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make building the real SDK client impossible for the duration of a test."""

    def refuse(api_key: str | None) -> Any:
        raise AssertionError(
            "the adapter tried to build a real SDK client; inject a fake instead"
        )

    monkeypatch.setattr(adapter, "_build_sdk_client", refuse)


# --------------------------------------------------------------------------- #
# The fake SDK: only `.chat.completions.create` is ever touched
# --------------------------------------------------------------------------- #


class FakeCompletions:
    def __init__(self, owner: "FakeSDK") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._next(**kwargs)


class FakeChat:
    def __init__(self, owner: "FakeSDK") -> None:
        self.completions = FakeCompletions(owner)


class FakeSDK:
    """A scripted stand-in for ``openai.OpenAI``.

    Each positional outcome is consumed by one attempt, in order: an exception
    instance is raised, anything else is returned as the response. With
    ``repeat_last=True`` the final outcome serves every further attempt, which is
    how "fails on every attempt" is scripted without writing the failure four
    times.
    """

    def __init__(self, *outcomes: Any, repeat_last: bool = False) -> None:
        self.chat = FakeChat(self)
        self._outcomes = list(outcomes)
        self._repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def _next(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index < len(self._outcomes):
            outcome = self._outcomes[index]
        elif self._repeat_last and self._outcomes:
            outcome = self._outcomes[-1]
        else:
            raise AssertionError(
                f"the adapter made attempt {index + 1}, beyond the "
                f"{len(self._outcomes)} scripted outcomes"
            )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def attempts(self) -> int:
        return len(self.calls)


class RecordingClock:
    """A ``sleep`` that records what it was asked to wait instead of waiting."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class FakeProviderError(Exception):
    """A provider error carrying an HTTP status, as the SDK's errors do.

    Deriving from ``Exception`` rather than ``OSError`` matters: the adapter
    classifies a transport error by builtin type, and an accidental ``OSError``
    base would make a 401 look retryable.
    """

    def __init__(self, status_code: int, message: str = "provider failure") -> None:
        super().__init__(f"{message} (HTTP {status_code})")
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# Response builders
# --------------------------------------------------------------------------- #


def sdk_response(
    *,
    text: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One provider response in the shape the adapter reads."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {"choices": [{"message": message}]}
    if usage is not None:
        body["usage"] = usage
    return body


def tool_call_payload(
    identifier: str, name: str, arguments: str
) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def user(content: str) -> Message:
    return Message(role="user", content=content)


def make_client(sdk: FakeSDK, clock: RecordingClock) -> OpenAIClient:
    return OpenAIClient(MODEL_ID, client=sdk, sleep=clock)


# --------------------------------------------------------------------------- #
# Req 2.5: a retryable failure is retried three times, sleeping 1 s, 2 s, 4 s
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retryable_failure_makes_four_attempts_and_sleeps_one_two_four(status: int) -> None:
    sdk = FakeSDK(FakeProviderError(status), repeat_last=True)
    clock = RecordingClock()
    client = make_client(sdk, clock)

    with pytest.raises(ModelClientError):
        client.complete(SYSTEM_PROMPT, [user("hello")])

    # One first attempt plus the three retries Req 2.5 allows, and nothing more.
    assert sdk.attempts == 4
    # The schedule exactly: 1 s before the second attempt, 2 s before the third,
    # 4 s before the fourth, and no sleep after the last failure.
    assert clock.sleeps == [1.0, 2.0, 4.0]


def test_transport_error_is_retryable_and_a_later_attempt_succeeds() -> None:
    sdk = FakeSDK(
        ConnectionError("connection reset"),
        TimeoutError("no reply in time"),
        sdk_response(text="recovered", usage={"prompt_tokens": 5, "completion_tokens": 2}),
    )
    clock = RecordingClock()
    client = make_client(sdk, clock)

    reply = client.complete(SYSTEM_PROMPT, [user("hello")])

    assert reply.text == "recovered"
    assert sdk.attempts == 3
    # Two sleeps because the third attempt succeeded; the schedule still starts
    # at 1 s and doubles.
    assert clock.sleeps == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# Req 2.6: a terminal failure raises at once, with no retry and no sleep
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "status, label",
    [
        (401, "authentication"),
        (403, "permission denied"),
        (404, "unknown model"),
        (400, "invalid request"),
        (422, "unprocessable request"),
    ],
)
def test_terminal_failure_makes_one_attempt_and_does_not_sleep(
    status: int, label: str
) -> None:
    sdk = FakeSDK(FakeProviderError(status), repeat_last=True)
    clock = RecordingClock()
    client = make_client(sdk, clock)

    with pytest.raises(ModelClientError) as raised:
        client.complete(SYSTEM_PROMPT, [user("hello")])

    assert sdk.attempts == 1, f"a {label} failure ({status}) must not be retried"
    assert clock.sleeps == []
    assert "not retried" in str(raised.value)


# --------------------------------------------------------------------------- #
# Req 2.7: exhausted retries raise one typed error naming provider and model,
# return no partial reply, and leave the caller's message list unmodified
# --------------------------------------------------------------------------- #


def test_exhausted_retries_raise_model_client_error_naming_provider_and_model() -> None:
    sdk = FakeSDK(FakeProviderError(503, "upstream unavailable"), repeat_last=True)
    clock = RecordingClock()
    client = make_client(sdk, clock)
    messages = [user("hello")]
    before = list(messages)

    with pytest.raises(ModelClientError) as raised:
        client.complete(SYSTEM_PROMPT, messages)

    error = raised.value
    assert error.provider == "openai"
    assert error.model_id == MODEL_ID
    text = str(error)
    assert "openai" in text
    assert MODEL_ID in text
    # The failure reason is described alongside the two identifiers.
    assert error.reason
    assert error.reason in text
    # No partial reply, and the conversation the caller passed in is untouched.
    assert messages == before


# --------------------------------------------------------------------------- #
# Req 2.11: an unmatched tool_call_id raises before any provider call
# --------------------------------------------------------------------------- #


def test_unmatched_tool_call_id_raises_before_issuing_a_call() -> None:
    sdk = FakeSDK(sdk_response(text="never reached"))
    clock = RecordingClock()
    client = make_client(sdk, clock)

    messages = [
        user("hello"),
        Message(role="tool", content="42", tool_call_id="call-nobody-issued"),
    ]

    with pytest.raises(UnmatchedToolCallError) as raised:
        client.complete(SYSTEM_PROMPT, messages)

    assert raised.value.tool_call_id == "call-nobody-issued"
    assert "call-nobody-issued" in str(raised.value)
    # The point of the requirement: the provider was never contacted.
    assert sdk.attempts == 0
    assert clock.sleeps == []


def test_a_tool_result_matching_an_issued_call_is_accepted() -> None:
    """The positive control for the check above (Req 2.10, Req 2.11)."""
    sdk = FakeSDK(
        sdk_response(
            tool_calls=[tool_call_payload("call_abc", "calculator", '{"a": 1, "b": 2}')]
        ),
        sdk_response(text="the answer is 3"),
    )
    clock = RecordingClock()
    client = make_client(sdk, clock)

    first = client.complete(SYSTEM_PROMPT, [user("add 1 and 2")])
    issued = first.tool_calls[0].id
    assert issued == "call_abc"

    messages = [
        user("add 1 and 2"),
        first.as_assistant_message(),
        Message(role="tool", content="3", tool_call_id=issued),
    ]
    second = client.complete(SYSTEM_PROMPT, messages)

    assert second.text == "the answer is 3"
    assert sdk.attempts == 2
    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# Req 2.2 / Req 2.3: usage is passed through when reported, estimated when not
# --------------------------------------------------------------------------- #


def test_provider_reported_usage_is_passed_through_unestimated() -> None:
    sdk = FakeSDK(
        sdk_response(
            text="hi",
            usage={"prompt_tokens": 123, "completion_tokens": 45},
        )
    )
    client = make_client(sdk, RecordingClock())

    reply = client.complete(SYSTEM_PROMPT, [user("hello")])

    assert reply.usage.input_tokens == 123
    assert reply.usage.output_tokens == 45
    assert reply.usage.estimated is False


def test_absent_usage_is_estimated_and_flagged() -> None:
    reply_text = "an answer long enough to be worth several tokens"
    sdk = FakeSDK(sdk_response(text=reply_text))
    client = make_client(sdk, RecordingClock())
    messages = [user("a question of some length")]

    reply = client.complete(SYSTEM_PROMPT, messages)

    expected = estimate_usage(SYSTEM_PROMPT, messages, reply_text, ())
    assert reply.usage.estimated is True
    assert reply.usage.input_tokens == expected.input_tokens
    assert reply.usage.output_tokens == expected.output_tokens
    assert reply.usage.input_tokens > 0
    assert reply.usage.output_tokens > 0


def test_a_half_filled_usage_block_is_estimated_rather_than_half_trusted() -> None:
    sdk = FakeSDK(sdk_response(text="hi", usage={"prompt_tokens": 10}))
    client = make_client(sdk, RecordingClock())

    reply = client.complete(SYSTEM_PROMPT, [user("hello")])

    assert reply.usage.estimated is True


# --------------------------------------------------------------------------- #
# Req 2.13: a reply carrying text and tool calls keeps both
# --------------------------------------------------------------------------- #


def test_reply_with_text_and_tool_calls_keeps_both() -> None:
    sdk = FakeSDK(
        sdk_response(
            text="Let me work that out.",
            tool_calls=[
                tool_call_payload(
                    "call_1", "calculator", '{"a": 1843, "b": 27, "operation": "multiply"}'
                ),
                tool_call_payload("call_2", "read_notes", "{}"),
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 8},
        )
    )
    client = make_client(sdk, RecordingClock())

    reply: ModelReply = client.complete(SYSTEM_PROMPT, [user("multiply 1843 by 27")])

    assert reply.text == "Let me work that out."
    assert reply.has_text and reply.has_tool_calls
    assert [call.name for call in reply.tool_calls] == ["calculator", "read_notes"]
    assert reply.tool_calls[0].arguments == {"a": 1843, "b": 27, "operation": "multiply"}
    assert reply.tool_calls[1].arguments == {}
    assert all(call.arguments_error is None for call in reply.tool_calls)
    # Both identifiers are recorded, so either result can be submitted later.
    assert reply.tool_calls[0].id in client.issued_ids
    assert reply.tool_calls[1].id in client.issued_ids


# --------------------------------------------------------------------------- #
# The factory seam used by every test above (Req 2.8)
# --------------------------------------------------------------------------- #


def test_build_model_client_forwards_the_injected_sdk_and_clock() -> None:
    config = load_config(
        {"PROVIDER": "openai", "OPENAI_API_KEY": FAKE_KEY, "MODEL_ID": MODEL_ID},
        probe=False,
    )
    sdk = FakeSDK(sdk_response(text="from the factory"))
    clock = RecordingClock()

    client = build_model_client(config, client=sdk, sleep=clock)
    reply = client.complete(SYSTEM_PROMPT, [user("hello")])

    assert client.model_id == MODEL_ID
    assert reply.text == "from the factory"
    assert sdk.attempts == 1


def test_build_model_client_rejects_an_unknown_provider_before_any_call() -> None:
    config = load_config(
        {"PROVIDER": "openai", "OPENAI_API_KEY": FAKE_KEY, "MODEL_ID": MODEL_ID},
        probe=False,
    )
    unknown = replace(config, provider="no-such-provider")

    with pytest.raises(ConfigError) as raised:
        build_model_client(unknown)

    assert "no-such-provider" in str(raised.value)
