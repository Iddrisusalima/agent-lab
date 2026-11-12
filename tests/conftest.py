"""Shared test fixtures: the scripted model client and the offline guards (Req 16.8, Req 16.9).

Three things live here, and nothing else.

**``FakeModelClient``** -- a scripted implementation of the ``ModelClient``
protocol. It returns prepared :class:`~agent_lab.model_client.ModelReply` values
in order and records every payload it was handed, so a loop test can both drive
the model's side of a run and afterwards assert on what was submitted: the
system prompt, the messages in order, and exactly which tool schemas were
offered (Req 4.5, Req 5.2). A script entry may also be an *exception instance*,
which is raised when the call reaches it -- that is how the model-error path of
Req 4.13 is exercised without a provider.

**The offline guards** -- two autouse fixtures. One clears every provider
credential variable from the environment, so no test can accidentally read a
real key and no code path can pick one up from the developer's shell (Req 16.8).
The other replaces the socket connect operations with ones that raise, so an
accidental outbound call fails loudly in the test that caused it instead of
quietly costing money.

Both guards are *function-scoped fixtures*, deliberately. They take effect after
collection and are undone after each test, so nothing here runs while pytest is
importing test modules -- a module that imports a heavy dependency at collection
time (``chromadb``, in the retrieval tests) is untouched by the socket guard,
and a test that genuinely needs a local connection can opt out with
``@pytest.mark.allow_network``.

**The ``tmp_path`` path fixtures** -- the notes file, the trace file, the corpus
directory and the Chroma directory, all under the per-test temporary directory,
plus a ``Config`` built from them. Req 16.9 requires the suite to write only
inside pytest's temporary directories and to pass on a clean clone where none of
those four paths exists yet, so none of them is pre-created except the corpus
directory, which tests fill with files of their own.

Tests reach the helper types either through the factory fixtures below or with
``from conftest import FakeModelClient`` -- ``tests/`` is not a package, so
pytest puts this directory on ``sys.path`` and the plain import works.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import socket
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pytest

from agent_lab.config import Config, load_config
from agent_lab.model_client import (
    Message,
    ModelReply,
    ToolCall,
    Usage,
    validate_request,
)

# --------------------------------------------------------------------------- #
# Constants shared by the fixtures
# --------------------------------------------------------------------------- #

#: The placeholder credential the fixtures hand to ``load_config``. Obviously
#: not a key: Req 17.3 keeps every committed file free of credential values, and
#: a string shaped like a real key invites someone to wonder.
FAKE_API_KEY = "sk-test-not-a-real-key"

#: The model identifier the fake reports. Priced in ``pricing/prices.json``, so
#: a cost assertion over a fake run has real prices to work from.
FAKE_MODEL_ID = "gpt-4o-mini"

#: Credential variables cleared before every test. The suffix rule catches the
#: ``<PROVIDER>_API_KEY`` shape that ``config.api_key_var_for`` builds for any
#: provider, present or future; the explicit names catch the neighbouring
#: variables an SDK reads on its own (a base URL pointed at a paid proxy is as
#: expensive as a key).
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_API_TOKEN", "_SECRET_KEY")
_CREDENTIAL_NAMES = (
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "AZURE_OPENAI_ENDPOINT",
)


# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #


def pytest_configure(config: pytest.Config) -> None:
    """Register the one marker that switches the socket guard off.

    Nothing in the suite uses it today. It exists because the guard is blunt by
    design -- it refuses *every* connect, loopback included -- and a test that
    legitimately needs a local socket should be able to say so in one line
    rather than be the reason the guard gets weakened for everyone.
    """
    config.addinivalue_line(
        "markers",
        "allow_network: let this test open socket connections; the default is to refuse them.",
    )


# --------------------------------------------------------------------------- #
# Offline guard 1: no credential is readable (Req 16.8)
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def no_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every provider credential variable from the environment.

    Req 16.8 requires the suite to pass with no provider key set, which is only
    meaningful if it also passes *when one is set* -- otherwise a developer with
    a key in their shell runs a different suite from CI. Clearing the variables
    makes the two identical.

    ``load_config`` reads its environment from a parameter rather than from
    ``os.environ`` (see ``config.load_config``), so tests inject a dict and this
    fixture is a second line of defence rather than the only one.
    """
    for name in list(os.environ):
        if name.endswith(_CREDENTIAL_SUFFIXES) or name in _CREDENTIAL_NAMES:
            monkeypatch.delenv(name, raising=False)

    # Chroma phones home for usage statistics unless told not to. Set here
    # rather than in the retrieval tests so that the setting cannot be forgotten
    # in one file, and so the socket guard below never has to fire for it.
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")


# --------------------------------------------------------------------------- #
# Offline guard 2: no socket connects (Req 16.8)
# --------------------------------------------------------------------------- #


class NetworkAccessBlocked(AssertionError):
    """An attempt to open a network connection during a test.

    An ``AssertionError`` on purpose, and specifically *not* a
    :class:`~agent_lab.model_client.ModelClientError`: the agent loop catches
    that type and ends the run with the model-error outcome (Req 4.13), so a
    blocked connection raised as one would be absorbed into a passing assertion
    about error handling. As an ``AssertionError`` it surfaces as the failure of
    whichever test reached for the network.
    """

    def __init__(self, target: Any, how: str) -> None:
        super().__init__(
            f"a test tried to open a network connection to {target!r} via {how}. "
            f"The suite runs offline (Req 16.8): inject FakeModelClient, or a fake "
            f"retriever, instead of calling a provider. If this connection is local "
            f"and intended, mark the test with @pytest.mark.allow_network."
        )


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every outbound connect raise :class:`NetworkAccessBlocked`.

    Only the *connect* operations are patched. Name resolution, socket creation,
    reads and writes on already-open file descriptors are all left alone, which
    keeps the guard clear of the filesystem, of pytest's own machinery, and of
    anything that constructs a socket object without using it.

    ``ssl.SSLSocket.connect`` is patched as well, because it overrides the base
    method rather than delegating to it, and an HTTPS client that connects
    through a wrapped socket would otherwise slip past.
    """
    if request.node.get_closest_marker("allow_network") is not None:
        return

    def refuse_connect(self: Any, address: Any = None, *args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessBlocked(address, "socket.connect")

    def refuse_connect_ex(self: Any, address: Any = None, *args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessBlocked(address, "socket.connect_ex")

    def refuse_create_connection(address: Any = None, *args: Any, **kwargs: Any) -> Any:
        raise NetworkAccessBlocked(address, "socket.create_connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect, raising=False)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_connect_ex, raising=False)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection, raising=False)

    try:  # pragma: no cover - a Python built without ssl
        import ssl
    except ImportError:
        return
    monkeypatch.setattr(ssl.SSLSocket, "connect", refuse_connect, raising=False)


# --------------------------------------------------------------------------- #
# The scripted model client
# --------------------------------------------------------------------------- #


class ScriptExhausted(AssertionError):
    """The fake was called more times than its script has entries.

    Raised instead of returning an improvised reply, because an improvised reply
    would let a test that mis-scripted a run still pass, with the loop taking a
    path nobody wrote down. The message names the call that went past the end,
    so the fix -- another reply, or ``repeat_last=True`` -- is obvious.

    An ``AssertionError`` for the same reason as
    :class:`NetworkAccessBlocked`: the loop must not be able to catch it and
    turn a mis-scripted test into a green one.
    """


@dataclasses.dataclass(frozen=True)
class RecordedCall:
    """One ``complete`` call, exactly as the loop submitted it.

    ``index``
        Zero-based position in the sequence of calls, so an assertion can speak
        about "the second call" without counting.
    ``system_prompt``
        The prompt text submitted (Req 12.1).
    ``messages``
        The conversation as submitted, in order. ``Message`` is frozen, so the
        tuple is a faithful record without copying.
    ``tool_schemas``
        A deep copy of the schemas offered. Copied, not referenced, because
        Property 15 asserts that exactly the registered schemas were offered on
        every call, and a later edit to the registry's dicts would otherwise
        rewrite history.
    """

    index: int
    system_prompt: str
    messages: tuple[Message, ...]
    tool_schemas: tuple[Mapping[str, Any], ...]

    @property
    def schema_names(self) -> tuple[str, ...]:
        """The offered tool names, in the order they were offered (Req 5.2)."""
        return tuple(str(schema.get("name", "")) for schema in self.tool_schemas)

    @property
    def roles(self) -> tuple[str, ...]:
        """The role of each submitted message, in order (Req 2.12)."""
        return tuple(message.role for message in self.messages)

    @property
    def last_message(self) -> Message | None:
        """The most recent submitted message, or ``None`` if there were none."""
        return self.messages[-1] if self.messages else None


#: A script entry: a prepared reply, an exception instance to raise when the
#: call reaches it, or a callable that is given the recorded call and returns
#: either of those. The callable form is what lets a test decide a reply from
#: what the loop actually submitted -- "answer with a tool call until a tool
#: result mentions X" -- without writing a second fake.
ScriptEntry = Any


class FakeModelClient:
    """A scripted ``ModelClient``: prepared replies out, submitted payloads kept.

    Satisfies the :class:`~agent_lab.model_client.ModelClient` protocol --
    ``model_id`` plus ``complete(system_prompt, messages, tool_schemas)`` -- and
    imports no provider SDK, which is the whole point: the agent loop is tested
    against the protocol before the real adapter is ever called (Req 16.7,
    Req 16.8).

    ``script``
        The entries to serve, one per call, in order. Each is a ``ModelReply``,
        an exception instance to raise, or a callable taking the
        :class:`RecordedCall` and returning one of those.
    ``model_id``
        What the cost tracker will price the run at (Req 3.2).
    ``repeat_last``
        When true, the final entry serves every call past the end of the script.
        This is how "the model keeps asking for tools forever" is scripted for
        the step-limit and cost-cap tests without writing the same reply eight
        times.
    ``validate``
        When true (the default), each call is checked against the bounds
        Req 2.1 states, using the same ``validate_request`` the real adapter
        uses. A fake that accepts submissions the real client would refuse is a
        fake that hides bugs.

    Every call is appended to :attr:`calls` *before* the entry is served, so the
    payload of a call that raised is recorded too -- a test asserting on the
    model-error path can still see what was submitted.
    """

    def __init__(
        self,
        script: Iterable[ScriptEntry] = (),
        *,
        model_id: str = FAKE_MODEL_ID,
        repeat_last: bool = False,
        validate: bool = True,
    ) -> None:
        self.model_id = model_id
        self.script: list[ScriptEntry] = list(script)
        self.repeat_last = repeat_last
        self.validate = validate
        self.calls: list[RecordedCall] = []

    # -- the protocol operation ------------------------------------------- #

    def complete(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]] = (),
    ) -> ModelReply:
        """Record the payload, then serve the next script entry (Req 2.1)."""
        if self.validate:
            validate_request(system_prompt, messages, tool_schemas)

        record = RecordedCall(
            index=len(self.calls),
            system_prompt=system_prompt,
            messages=tuple(messages),
            tool_schemas=tuple(copy.deepcopy(dict(schema)) for schema in tool_schemas),
        )
        self.calls.append(record)

        entry = self._entry_for(record.index)
        if callable(entry) and not isinstance(entry, BaseException):
            entry = entry(record)
        if isinstance(entry, BaseException):
            raise entry
        if not isinstance(entry, ModelReply):
            raise AssertionError(
                f"script entry {record.index} must be a ModelReply, an exception "
                f"instance, or a callable returning one of those; received "
                f"{type(entry).__name__}"
            )
        return entry

    def _entry_for(self, index: int) -> ScriptEntry:
        if index < len(self.script):
            return self.script[index]
        if self.repeat_last and self.script:
            return self.script[-1]
        raise ScriptExhausted(
            f"FakeModelClient was called {index + 1} times but its script holds "
            f"{len(self.script)} entr{'y' if len(self.script) == 1 else 'ies'}. "
            f"Add another entry, or construct it with repeat_last=True to let the "
            f"final entry serve every further call."
        )

    # -- what the test asserts on ----------------------------------------- #

    @property
    def call_count(self) -> int:
        """How many times ``complete`` was called, failures included."""
        return len(self.calls)

    @property
    def last_call(self) -> RecordedCall | None:
        """The most recent recorded call, or ``None`` before the first."""
        return self.calls[-1] if self.calls else None

    def schemas_offered(self) -> list[tuple[str, ...]]:
        """The tool names offered on each call, one tuple per call (Req 4.5)."""
        return [call.schema_names for call in self.calls]

    def submitted_messages(self) -> list[tuple[Message, ...]]:
        """The conversation as submitted on each call, one tuple per call."""
        return [call.messages for call in self.calls]

    def tool_results_submitted(self) -> list[str]:
        """The content of every tool message submitted, in submission order.

        The one assertion the grounding and self-correction requirements need
        most often: that a tool result -- delimited (Req 12.8) or an error
        (Req 5.11) -- reached the model.
        """
        seen: list[str] = []
        for call in self.calls:
            for message in call.messages:
                if message.role == "tool":
                    seen.append(message.content)
        # The loop resubmits the whole conversation each step, so the same result
        # appears in every later call; de-duplicate by position, keeping order.
        unique: list[str] = []
        for content in seen:
            if content not in unique:
                unique.append(content)
        return unique


# --------------------------------------------------------------------------- #
# Builders for scripted replies
# --------------------------------------------------------------------------- #


def make_usage(input_tokens: int = 10, output_tokens: int = 5, *, estimated: bool = False) -> Usage:
    """A ``Usage`` with small non-zero counts, so cost arithmetic is visible."""
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, estimated=estimated)


def make_tool_call(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    id: str | None = None,
    arguments_error: str | None = None,
) -> ToolCall:
    """One tool request, with an identifier generated when none is given.

    The generated shape matches what the real adapter synthesises for a provider
    that sends no identifier (``call-{step}-{index}``), so nothing downstream
    behaves differently for a fake call than for a real one.
    """
    return ToolCall(
        id=id or f"call-fake-{name}-{abs(hash((name, repr(arguments)))) % 100000}",
        name=name,
        arguments=dict(arguments or {}),
        arguments_error=arguments_error,
    )


def make_reply(
    text: str = "",
    tool_calls: Sequence[ToolCall] = (),
    *,
    usage: Usage | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> ModelReply:
    """A ``ModelReply``, which is mostly a shorter way of writing the usage.

    ``make_reply("done")`` is a final answer (Req 4.3);
    ``make_reply(tool_calls=[make_tool_call("calculator", {...})])`` is a
    tool-only step; passing both keeps both, which Req 2.13 requires.
    """
    return ModelReply(
        text=text,
        tool_calls=tuple(tool_calls),
        usage=usage if usage is not None else make_usage(input_tokens, output_tokens),
    )


@pytest.fixture
def make_model_client() -> Callable[..., FakeModelClient]:
    """Factory fixture: ``make_model_client(reply_a, reply_b, repeat_last=True)``.

    A factory rather than a prepared client, because the script is the thing
    each test is actually choosing, and there is no sensible default script.
    """

    def factory(*script: ScriptEntry, **kwargs: Any) -> FakeModelClient:
        # A single iterable argument is accepted as well as varargs, so a test
        # that built its script in a list does not have to unpack it.
        if len(script) == 1 and isinstance(script[0], (list, tuple)):
            entries: Iterable[ScriptEntry] = script[0]
        else:
            entries = script
        return FakeModelClient(entries, **kwargs)

    return factory


# --------------------------------------------------------------------------- #
# Temporary paths (Req 16.9)
# --------------------------------------------------------------------------- #


@pytest.fixture
def notes_file(tmp_path: Path) -> Path:
    """The notes path for one test. The file does not exist; its parent does.

    Absent rather than empty, because Req 10.2 and Req 10.3 are different
    results and a pre-created file would quietly make the absent-file case
    unreachable.
    """
    return tmp_path / "notes.md"


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    """The trace path for one test. Neither the file nor its parent exists.

    The missing parent is deliberate and mirrors the documented default
    (``traces/run.jsonl``): Req 11.10 requires the logger to create the file and
    any missing parent directory at run start, so every test using this fixture
    exercises that.
    """
    return tmp_path / "traces" / "run.jsonl"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """An existing, empty corpus directory for one test.

    Created, unlike the other three, because a test that wants documents writes
    them here and a test that wants the *absent* directory case of Req 8.9
    should name a path it knows does not exist rather than delete this one.
    """
    path = tmp_path / "corpus"
    path.mkdir()
    return path


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    """The Chroma persistence path for one test. Does not exist.

    Req 16.9 again: the suite must pass on a clean clone with no persistence
    directory present, and must never write to the repository's own ``.chroma``.
    """
    return tmp_path / ".chroma"


@pytest.fixture
def tmp_config(
    notes_file: Path,
    trace_file: Path,
    corpus_dir: Path,
    chroma_dir: Path,
) -> Config:
    """A real validated ``Config`` whose every path is under ``tmp_path``.

    Built by ``load_config`` from an injected dict rather than assembled by hand,
    so the defaults, the parsing and the range checks of Req 1.6 - Req 1.9 are
    the ones the product uses. ``probe=False`` skips the access check of
    Req 1.10, which has its own tests and which some callers here deliberately
    point at paths that do not exist yet.

    ``interactive=True`` matches Req 9.11's default; a test that wants the
    non-interactive gate uses ``dataclasses.replace(tmp_config,
    interactive=False)``, which is also the only way the product allows it.
    """
    return load_config(
        {
            "OPENAI_API_KEY": FAKE_API_KEY,
            "MODEL_ID": FAKE_MODEL_ID,
            "DOCS_DIR": str(corpus_dir),
            "NOTES_FILE": str(notes_file),
            "TRACE_FILE": str(trace_file),
            "CHROMA_DIR": str(chroma_dir),
        },
        interactive=True,
        probe=False,
    )
