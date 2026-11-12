"""Unit tests for the optional streaming output (task 14.1; Req 18.2, Req 18.9).

Streaming is a stretch feature with three moving parts, and these tests drive
all three together rather than separately, because the requirement is about what
the *operator* sees:

* `config.streaming_enabled`, derived from `STREAM_OUTPUT` (default `off`);
* `cli.run_command`, which builds an `AnswerStreamer`, hands it to the adapter
  as a token sink, and decides whether its own single `final_answer` write is
  still needed;
* `OpenAIClient._consume_stream`, which forwards each content fragment on
  receipt and reassembles the chunks into the shape of an ordinary response;
* `PlainRenderer.stream_fragment`, the one place a fragment reaches stdout.

So the tests below call `cli.run_command` with an environment dictionary and
three `StringIO` buffers, exactly as the CLI docstring says a test may. Three
seams are injected and nothing else: the SDK client (a scripted fake), the
`sleep` callable (a recorder), and -- so that the central assertion of Req 18.2
can name the two sides it compares -- the loop and the streamer the CLI builds,
wrapped to keep a reference to the objects the product constructed.

Offline by construction, like `tests/test_model_client.py`: the autouse
`no_real_sdk` fixture replaces the one function that builds a real SDK client
with a failing stand-in, so a test that forgot to inject a fake fails loudly
instead of reaching the network, and `tests/conftest.py` additionally refuses
every socket connect and clears every provider credential. The only
key-shaped string here is `conftest.FAKE_API_KEY`.

Req 18.2 also bounds the delay on a token at 1 second. No test asserts a
duration -- a wall-clock bound is not what a fake provider can demonstrate.
What is asserted instead is stronger and deterministic: the stream records what
stdout already held each time it was asked for the next chunk, so every token
is shown to be on stdout before the following one was even received.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field
from typing import IO, Any, Mapping, Sequence

import pytest

from agent_lab import cli
from agent_lab.agent_loop import AgentLoop, RunResult
from agent_lab.model_client import Message, estimate_usage
from agent_lab.providers import openai_client as adapter
from agent_lab.providers.openai_client import OpenAIClient
from agent_lab.redaction import REDACTION_MARKER
from agent_lab.render import FINAL_ANSWER_HEADING, SUMMARY_PREFIX, AnswerStreamer
from conftest import FAKE_API_KEY, FAKE_MODEL_ID

TASK = "say hello"
SYSTEM_PROMPT = "You are a test agent."


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

    One positional outcome per attempt, in order: an exception instance is
    raised, a :class:`DeltaStream` or any other iterable is returned as a chunk
    stream, and a mapping is returned as a complete response -- which is the
    Req 18.9 case, a provider that was asked to stream and did not.
    """

    def __init__(self, *outcomes: Any) -> None:
        self.chat = FakeChat(self)
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def _next(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index >= len(self._outcomes):
            raise AssertionError(
                f"the adapter made attempt {index + 1}, beyond the "
                f"{len(self._outcomes)} scripted outcomes"
            )
        outcome = self._outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def attempts(self) -> int:
        return len(self.calls)

    @property
    def first_request(self) -> dict[str, Any]:
        assert self.calls, "the adapter made no request"
        return self.calls[0]


class RecordingClock:
    """A ``sleep`` that records what it was asked to wait instead of waiting."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


# --------------------------------------------------------------------------- #
# Chunk builders, in the shape `_consume_stream` reads
# --------------------------------------------------------------------------- #


def content_chunk(text: str) -> dict[str, Any]:
    """One streamed content fragment: ``choices[0].delta.content``."""
    return {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    """The final chunk a provider honouring ``stream_options`` sends (Req 2.2)."""
    return {
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def complete_response(text: str) -> dict[str, Any]:
    """A whole response, as a provider that ignores ``stream=True`` returns it."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class DeltaStream:
    """An iterable of content chunks that watches stdout as it is consumed.

    Before yielding each chunk it records what the watched buffer already holds,
    so a test can show that fragment *n* was on stdout before fragment *n + 1*
    was received -- the incremental half of Req 18.2, asserted without a clock.
    """

    def __init__(self, *texts: str, trailing: Sequence[Mapping[str, Any]] = ()) -> None:
        self.texts = tuple(texts)
        self.trailing = tuple(trailing)
        self.watch: IO[str] | None = None
        self.snapshots: list[str] = []

    def __iter__(self) -> Any:
        for text in self.texts:
            self.snapshots.append(self.watch.getvalue() if self.watch is not None else "")
            yield content_chunk(text)
        for chunk in self.trailing:
            yield chunk

    @property
    def answer(self) -> str:
        """What the provider streamed, concatenated."""
        return "".join(self.texts)


# --------------------------------------------------------------------------- #
# Reading the captured stdout
# --------------------------------------------------------------------------- #


def answer_region(stdout: str) -> str:
    """Whatever stands under the final-answer heading, before the run summary.

    Written against the two constants the renderer uses rather than against a
    character count, and returns ``""`` before the heading has been written --
    which is the state of a mid-stream snapshot taken before the first token.
    """
    opening = FINAL_ANSWER_HEADING + "\n"
    if opening not in stdout:
        return ""
    after = stdout.split(opening, 1)[1]
    return after.split("\n" + SUMMARY_PREFIX, 1)[0]


# --------------------------------------------------------------------------- #
# Driving `agentlab run` with the fake provider
# --------------------------------------------------------------------------- #


@dataclass
class CliRun:
    """Everything one driven run exposes to an assertion."""

    exit_code: int
    stdout: str
    stderr: str
    sdk: FakeSDK
    clock: RecordingClock
    overrides: dict[str, Any] = field(default_factory=dict)
    client: OpenAIClient | None = None
    streamer: AnswerStreamer | None = None
    result: RunResult | None = None

    @property
    def final_answer(self) -> str:
        """The answer the loop returned -- the right-hand side of Req 18.2."""
        assert self.result is not None, "the loop was never run"
        assert self.result.final_answer is not None, "the run produced no final answer"
        return self.result.final_answer


@pytest.fixture
def cli_env(
    corpus_dir: Any, notes_file: Any, trace_file: Any, chroma_dir: Any
) -> dict[str, str]:
    """A whole environment for one run, every path under ``tmp_path`` (Req 16.9)."""
    return {
        "OPENAI_API_KEY": FAKE_API_KEY,
        "MODEL_ID": FAKE_MODEL_ID,
        "DOCS_DIR": str(corpus_dir),
        "NOTES_FILE": str(notes_file),
        "TRACE_FILE": str(trace_file),
        "CHROMA_DIR": str(chroma_dir),
    }


def drive_cli(
    monkeypatch: pytest.MonkeyPatch,
    env: Mapping[str, str],
    *outcomes: Any,
    stream_output: str | None = None,
    task: str = TASK,
) -> CliRun:
    """Run ``agentlab run`` against the scripted provider and capture both streams.

    ``stream_output`` is written into the environment only when given, so a test
    for the default state can leave ``STREAM_OUTPUT`` genuinely unset rather
    than set it to the value it defaults to.

    Three substitutions, and the streaming wiring is not among them: the adapter
    is the real :class:`OpenAIClient` built with the overrides ``run_command``
    chose, and the streamer and the loop are the real classes. The wrappers only
    keep a reference to what the product constructed.
    """
    run_env = dict(env)
    if stream_output is not None:
        run_env["STREAM_OUTPUT"] = stream_output

    stdout, stderr = io.StringIO(), io.StringIO()
    sdk = FakeSDK(*outcomes)
    clock = RecordingClock()
    for outcome in outcomes:
        if isinstance(outcome, DeltaStream):
            outcome.watch = stdout

    captured = CliRun(exit_code=-1, stdout="", stderr="", sdk=sdk, clock=clock)

    def build_client(config: Any, **overrides: Any) -> OpenAIClient:
        captured.overrides = dict(overrides)
        captured.client = OpenAIClient(
            config.model_id, client=sdk, sleep=clock, **overrides
        )
        return captured.client

    def build_streamer(renderer: Any) -> AnswerStreamer:
        captured.streamer = AnswerStreamer(renderer)
        return captured.streamer

    def build_loop(*args: Any, **kwargs: Any) -> AgentLoop:
        loop = AgentLoop(*args, **kwargs)
        inner = loop.run

        def run(task_text: str) -> RunResult:
            result = inner(task_text)
            captured.result = result
            return result

        loop.run = run  # type: ignore[method-assign]
        return loop

    monkeypatch.setattr(cli, "build_model_client", build_client)
    monkeypatch.setattr(cli, "AnswerStreamer", build_streamer)
    monkeypatch.setattr(cli, "AgentLoop", build_loop)

    captured.exit_code = cli.run_command(
        argparse.Namespace(task=task, non_interactive=True),
        cli.CliContext(
            env=run_env, stdin=io.StringIO(""), stdout=stdout, stderr=stderr
        ),
    )
    captured.stdout = stdout.getvalue()
    captured.stderr = stderr.getvalue()
    return captured


# --------------------------------------------------------------------------- #
# Req 18.6, Req 18.7: unset means off, and off means unchanged
# --------------------------------------------------------------------------- #


def test_streaming_is_off_when_the_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch, cli_env: dict[str, str]
) -> None:
    """With ``STREAM_OUTPUT`` absent, the run is the run that shipped.

    The disabled state is checked at all three places it has to hold: the CLI
    passes the adapter no streaming arguments, the request carries no ``stream``
    key, and stdout holds one final-answer write with no unavailability message
    -- there is nothing to be unavailable when nothing was asked for.
    """
    answer = "Hello, world!"
    assert "STREAM_OUTPUT" not in cli_env
    run = drive_cli(monkeypatch, cli_env, complete_response(answer))

    assert run.exit_code == 0
    # The whole of the feature's wiring, absent (Req 18.6).
    assert run.overrides == {}
    assert run.streamer is None
    assert run.client is not None and run.client.streaming_requested is False
    # The request is the request it always was: no streaming keys at all.
    assert "stream" not in run.sdk.first_request
    assert "stream_options" not in run.sdk.first_request

    # One write of the answer, under one heading (Req 13.4, Req 18.7).
    assert run.stdout.count(FINAL_ANSWER_HEADING) == 1
    assert run.stdout.count(answer) == 1
    assert answer_region(run.stdout) == answer == run.final_answer
    assert cli.STREAMING_UNAVAILABLE_MESSAGE not in run.stderr
    assert cli.STREAM_MISMATCH_MESSAGE not in run.stderr


# --------------------------------------------------------------------------- #
# Req 18.2: the concatenation guarantee
# --------------------------------------------------------------------------- #


def test_streamed_tokens_concatenate_to_the_answer_the_loop_returns(
    monkeypatch: pytest.MonkeyPatch, cli_env: dict[str, str]
) -> None:
    """The sharp clause of Req 18.2, asserted three ways.

    1. What reached stdout under the final-answer heading, concatenated, equals
       the ``final_answer`` the loop returned -- character for character, with
       nothing dropped, nothing doubled and nothing reordered.
    2. The streamer's own record of what it wrote equals the same string, so the
       equality in (1) is not an artefact of the CLI reprinting the answer: the
       second write is suppressed, and the mismatch warning is absent.
    3. Each fragment was on stdout before the next one was received, so the
       tokens were written as they arrived rather than assembled and printed at
       the end.

    And then the whole of stdout is compared with the same run performed with
    streaming off, which is Req 18.7's claim in its strongest form: enabling the
    feature changed not one byte of the required output.
    """
    fragments = ("The ", "harbor ", "station ", "reports ", "fair ", "weather.")
    stream = DeltaStream(*fragments)
    run = drive_cli(monkeypatch, cli_env, stream, stream_output="on")

    assert run.exit_code == 0
    assert run.overrides["stream"] is True
    assert run.client is not None and run.client.streamed_incrementally is True

    # (1) what reached stdout == what the loop returned.
    assert run.final_answer == stream.answer
    assert answer_region(run.stdout) == run.final_answer
    assert run.stdout.count(FINAL_ANSWER_HEADING) == 1
    assert run.stdout.count(stream.answer) == 1

    # (2) the sink's record agrees, and the CLI therefore wrote nothing twice.
    assert run.streamer is not None
    assert run.streamer.streamed == run.final_answer
    assert run.streamer.retractions == 0
    assert cli.STREAM_MISMATCH_MESSAGE not in run.stderr
    assert cli.STREAMING_UNAVAILABLE_MESSAGE not in run.stderr

    # (3) each fragment was visible before the next was asked for.
    assert len(stream.snapshots) == len(fragments)
    for index, snapshot in enumerate(stream.snapshots):
        assert answer_region(snapshot) == "".join(fragments[:index]), (
            f"before fragment {index} arrived, stdout held "
            f"{answer_region(snapshot)!r}"
        )

    # Req 18.7: byte-identical to the same answer printed with streaming off.
    plain = drive_cli(monkeypatch, cli_env, complete_response(stream.answer))
    assert run.stdout == plain.stdout


# --------------------------------------------------------------------------- #
# Req 18.9: asked to stream, the provider did not
# --------------------------------------------------------------------------- #


def test_a_provider_that_does_not_stream_prints_one_write_and_says_so(
    monkeypatch: pytest.MonkeyPatch, cli_env: dict[str, str]
) -> None:
    """Req 18.9: the complete answer in a single write, plus the message.

    The fake returns a whole response although ``stream=True`` was sent, which
    is what an OpenAI-compatible gateway without streaming support does. That is
    not a failure: the adapter normalises the response as usual, reports
    ``streaming_unavailable``, and the CLI prints the answer through its ordinary
    ``final_answer`` write and names the reason on stderr.
    """
    answer = "Streaming was not available, but the answer still arrived."
    run = drive_cli(
        monkeypatch, cli_env, complete_response(answer), stream_output="on"
    )

    assert run.exit_code == 0
    # Streaming was asked for, and the provider sent no deltas.
    assert run.overrides["stream"] is True
    assert run.client is not None
    assert run.client.streamed_incrementally is False
    assert run.client.streaming_unavailable is True

    # The message Req 18.9 requires, on stderr with every other warning (13.8).
    assert cli.STREAMING_UNAVAILABLE_MESSAGE in run.stderr
    assert "streaming output was unavailable" in run.stderr

    # One write, not a stream and not a duplicate: nothing was streamed, so the
    # CLI's own single write is the only thing that printed the answer.
    assert run.streamer is not None and run.streamer.wrote_answer is False
    assert run.streamer.streamed == ""
    assert run.stdout.count(FINAL_ANSWER_HEADING) == 1
    assert run.stdout.count(answer) == 1
    assert answer_region(run.stdout) == answer == run.final_answer
    assert cli.STREAM_MISMATCH_MESSAGE not in run.stderr

    # And stdout is what the same reply prints with streaming off: Req 18.9 adds
    # a line on stderr and changes nothing on stdout (Req 18.7).
    plain = drive_cli(monkeypatch, cli_env, complete_response(answer))
    assert run.stdout == plain.stdout


# --------------------------------------------------------------------------- #
# Req 13.9, Req 1.11: redaction survives fragmentation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fragments",
    [
        pytest.param(("the key is " + FAKE_API_KEY + " ok",), id="one-fragment"),
        pytest.param(("the key is sk-test-", "not-a-real-key ok"), id="split-in-two"),
        pytest.param(
            ("the key is ", "sk-test-", "not-a-real", "-key", " ok"), id="split-in-four"
        ),
        pytest.param(
            tuple("the key is " + FAKE_API_KEY + " ok"), id="one-character-each"
        ),
    ],
)
def test_the_api_key_never_reaches_a_stream_however_it_is_chunked(
    monkeypatch: pytest.MonkeyPatch, cli_env: dict[str, str], fragments: tuple[str, ...]
) -> None:
    """The configured key is redacted even when no single fragment contains it.

    ``redact`` replaces occurrences in the string it is given, so a key split
    across two provider chunks is an occurrence in neither. The renderer closes
    that by holding back, at the end of each fragment, the longest suffix that
    could begin the key, and redacting the joined buffer on the next fragment
    (``render.split_on_partial_secret``). These four chunkings put the key in
    one fragment, across two, across four, and one character at a time; in all
    of them the value is absent from both streams and the marker stands where it
    was.

    What this pins is exactly what the implementation documents and no more. The
    guarantee is about *the configured key*: the renderer is not a secret
    scanner and cannot recognise a credential it was never given. The hold-back
    is also bounded -- only an ambiguous tail waits, and ``end_stream`` releases
    whatever is still held -- so the answer still arrives in full.
    """
    stream = DeltaStream(*fragments)
    run = drive_cli(monkeypatch, cli_env, stream, stream_output="on")

    assert run.exit_code == 0
    # The test would pass vacuously if the model had not said the key at all.
    assert FAKE_API_KEY in stream.answer
    assert FAKE_API_KEY in run.final_answer

    assert FAKE_API_KEY not in run.stdout
    assert FAKE_API_KEY not in run.stderr
    assert REDACTION_MARKER in run.stdout
    # Nothing but the key was withheld: the surrounding prose arrived whole.
    assert answer_region(run.stdout) == stream.answer.replace(
        FAKE_API_KEY, REDACTION_MARKER
    )


# --------------------------------------------------------------------------- #
# Req 2.2, Req 2.3: usage accounting for a streamed reply
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider_reports_usage", [False, True])
def test_streamed_usage_is_reported_when_sent_and_estimated_when_not(
    provider_reports_usage: bool,
) -> None:
    """A streamed reply is priced from the provider's block, or from an estimate.

    The request asks for the block (``stream_options={"include_usage": True}``),
    because a streamed response otherwise carries none. Both answers to that
    request are legitimate and the adapter handles them in one place:

    * the provider sends a usage chunk -- the counts are reported as they are,
      ``estimated=False`` (Req 2.2);
    * the provider ignores the option -- the counts come from ``estimate_usage``
      over what was submitted and what came back, ``estimated=True`` (Req 2.3),
      which is what makes the cost tracker call the run's cost an estimate.

    Asserted at the adapter rather than through the CLI because the ``Usage``
    object is what the requirement is about, and this is where it is built.
    """
    fragments = ("Fair ", "weather ", "at ", "the ", "harbor.")
    trailing = (usage_chunk(120, 7),) if provider_reports_usage else ()
    sdk = FakeSDK(DeltaStream(*fragments, trailing=trailing))
    clock = RecordingClock()
    received: list[str] = []

    client = OpenAIClient(
        FAKE_MODEL_ID,
        client=sdk,
        sleep=clock,
        stream=True,
        token_sink=received.append,
    )
    messages = [Message(role="user", content=TASK)]
    reply = client.complete(SYSTEM_PROMPT, messages)

    # The two keys streaming adds to the request, and nothing waited.
    assert sdk.first_request["stream"] is True
    assert sdk.first_request["stream_options"] == {"include_usage": True}
    assert clock.sleeps == []

    assert reply.text == "".join(fragments)
    assert "".join(received) == reply.text

    if provider_reports_usage:
        assert reply.usage.estimated is False
        assert reply.usage.input_tokens == 120
        assert reply.usage.output_tokens == 7
    else:
        expected = estimate_usage(SYSTEM_PROMPT, messages, reply.text, ())
        assert reply.usage.estimated is True
        assert reply.usage == expected
        assert reply.usage.input_tokens > 0
        assert reply.usage.output_tokens > 0
