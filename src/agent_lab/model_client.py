"""The model-calling seam: one protocol, four value types, one estimator (Req 2).

This module is the boundary between the agent loop and whichever large language
model provider is configured. It contains the ``ModelClient`` protocol with its
single ``complete`` operation (Req 2.1), the normalised value types that cross
that boundary (``Message``, ``ToolCall``, ``Usage``, ``ModelReply``), the two
typed errors the caller can expect, and the pure token estimator used when a
provider response omits its usage block (Req 2.3).

**It imports no provider SDK, and it never will.** Everything vendor-shaped --
the HTTP client, the retry table, the tool-schema envelope, the parsing of a
provider's tool-call payload -- lives in ``providers/openai_client.py`` behind
this protocol. That split is what lets the loop be written once against
``complete`` and tested entirely offline against a scripted fake (Req 2.8,
Req 16.8), and what makes a second provider a new file plus one factory entry
rather than an edit to the loop.

**Why the value types are frozen.** A reply travels to four consumers in the
same step: the conversation state, the cost tracker, the trace logger and the
renderer. If any one of them could edit what it received, the others would see
a different reply, and the trace would stop being evidence of what happened.

**Where validation belongs.** The validation in ``__post_init__`` here is
developer-facing: these objects are constructed by the adapter and the loop, not
by the model, so a malformed one is a bug worth failing loudly at construction
rather than a condition to report at the end of a run. Model-supplied content is
the opposite case and is never validated by raising -- an unparseable argument
payload becomes a ``ToolCall`` carrying ``arguments_error``, which the
dispatcher turns into a schema error the model can correct (Req 5.5).

**The structural contract with the dispatcher.** ``ToolDispatcher.dispatch``
reads a call by looking for ``name``, ``arguments`` and ``arguments_error``
through attribute access, deliberately not importing this module. ``ToolCall``
exposes exactly those three names as attributes, so the two sides meet without
either depending on the other.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------- #
# Bounds stated by Req 2.1
# --------------------------------------------------------------------------- #

# Req 2.1: the operation accepts an ordered message list of 1 to 1000 messages.
# One is the floor because a call with nothing to answer is meaningless; the
# ceiling is a guard, not a target -- MAX_STEPS of 8 with at most 8 tool calls
# per step cannot approach it, so reaching it means something is appending in a
# loop.
MIN_MESSAGES = 1
MAX_MESSAGES = 1_000

# Req 2.1: 0 to 64 tool schemas. Zero is permitted so the same client can be
# used for a plain completion with no tools offered; the registry ships four.
MIN_TOOL_SCHEMAS = 0
MAX_TOOL_SCHEMAS = 64

# Req 2.3: the estimator's ratio. Four characters per token is the rule of thumb
# for English text in byte-pair vocabularies. It is an estimate and is reported
# as one -- every usage value derived from it carries ``estimated=True``, so no
# cost figure computed from it can be mistaken for a measured one.
CHARS_PER_TOKEN = 4

Role = Literal["system", "user", "assistant", "tool"]

# The four roles, as a set, for the one place that checks membership.
ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


# --------------------------------------------------------------------------- #
# The typed errors (Req 2.6, Req 2.7, Req 2.11)
# --------------------------------------------------------------------------- #


class ModelClientError(Exception):
    """The single error type a failed model call raises (Req 2.6, Req 2.7).

    One type for every reason a call can fail -- authentication, an invalid
    request, an unknown model, or exhausted retries -- because the loop's
    response is the same in every case: end the run with the model error
    outcome, keep the conversation and the trace intact, and report (Req 4.13).
    Distinguishing the reasons is the adapter's job and is carried in the
    message, not in the type.

    ``provider``, ``model_id`` and ``reason`` are optional and, when supplied,
    are composed into the message, because Req 2.7 requires the error to
    describe all three. They are also kept as attributes so the renderer can
    report them without parsing prose.

    No partial reply is ever attached: Req 2.7 requires the caller's message
    list to be left unmodified, and an error carrying half a reply would invite
    a caller to append it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.reason = reason
        super().__init__(_compose_error_message(message, provider, model_id, reason))


class UnmatchedToolCallError(ModelClientError):
    """A tool result refers to a tool call that was never issued (Req 2.11).

    Raised before any provider request is made, so the submission is refused
    rather than sent and rejected. The unmatched identifier is named in the
    message and kept on ``tool_call_id``.

    A ``ModelClientError`` subclass on purpose. This condition is a bookkeeping
    fault in the caller rather than a provider failure, but the loop already
    catches ``ModelClientError`` and ends the run with an intact trace, which is
    a better answer than an exception escaping to the CLI. Code that wants to
    tell the two apart can still catch this type first.
    """

    def __init__(self, tool_call_id: Any, issued: Iterable[str] = ()) -> None:
        self.tool_call_id = tool_call_id
        known = sorted(str(item) for item in issued)
        known_text = ", ".join(repr(item) for item in known) if known else "none"
        super().__init__(
            f"Tool result references tool_call_id {tool_call_id!r}, which no preceding "
            f"reply in this run issued; no provider call was made. "
            f"Identifiers issued so far: {known_text}."
        )


def _compose_error_message(
    message: str,
    provider: str | None,
    model_id: str | None,
    reason: str | None,
) -> str:
    """Append whichever of provider, model id and reason were supplied (Req 2.7)."""
    parts = [message] if message else []
    details = []
    if provider:
        details.append(f"provider={provider}")
    if model_id:
        details.append(f"model={model_id}")
    if reason:
        details.append(f"reason={reason}")
    if details:
        parts.append("(" + ", ".join(details) + ")")
    return " ".join(parts) if parts else "the model call failed"


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolCall:
    """One normalised tool request (Req 2.4).

    ``id``
        The tool-call identifier, unique within the run, by which a tool result
        is matched to exactly one request (Req 2.4, Req 2.10). The adapter
        copies the provider's identifier when there is one and synthesises
        ``call-{step}-{index}`` otherwise, so the loop's bookkeeping does not
        depend on provider behaviour.
    ``name``
        The requested tool name, exactly as it arrived. Not normalised: the
        registry matches case-sensitively and exactly (Req 5.3), so folding case
        here would route a call the model got wrong to a tool it did not ask
        for.
    ``arguments``
        The decoded argument mapping. Empty when the model sent none, and also
        empty when the payload could not be decoded -- in which case
        ``arguments_error`` says so.
    ``arguments_error``
        Set when the provider's argument payload failed to parse. The adapter
        records the failure here instead of raising, and the dispatcher reports
        it as a schema error the model can correct rather than ending the run.

    ``name``, ``arguments`` and ``arguments_error`` are the three attributes
    ``ToolDispatcher`` reads, and they are attributes rather than a nested
    payload for that reason.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"tool call id must be a non-empty string, received {self.id!r}")
        if not isinstance(self.name, str):
            # The adapter is responsible for presenting the provider's name as a
            # string. An empty name is allowed and resolves to the unknown-tool
            # error result; a non-string one is a defect in the adapter.
            raise ValueError(
                f"tool call {self.id!r} name must be a string, "
                f"received {type(self.name).__name__}"
            )
        if self.arguments_error is not None and not isinstance(self.arguments_error, str):
            raise ValueError(
                f"tool call {self.id!r} arguments_error must be a string or None, "
                f"received {type(self.arguments_error).__name__}"
            )
        if not isinstance(self.arguments, Mapping):
            raise ValueError(
                f"tool call {self.id!r} arguments must be a mapping, "
                f"received {type(self.arguments).__name__}; set arguments_error and pass an "
                f"empty mapping when the payload could not be decoded"
            )
        # Store an independent dict, so a caller that keeps and later mutates the
        # mapping it passed in cannot change what the dispatcher validates or
        # what the trace recorded. One level is enough: the values come from
        # decoded JSON and are handed to the tool as they are.
        object.__setattr__(self, "arguments", dict(self.arguments))

    @property
    def failed_to_parse(self) -> bool:
        """Whether the argument payload could not be decoded."""
        return bool(self.arguments_error)


@dataclass(frozen=True)
class Message:
    """One entry of conversation state, in the order the loop appended it.

    ``role``
        One of ``system``, ``user``, ``assistant``, ``tool``.
    ``content``
        The text. Empty is allowed and is normal on an assistant message that
        carries only tool calls -- a model asking for a tool without commenting
        on it.
    ``tool_calls``
        The calls an assistant message asked for. Empty on every other role.
    ``tool_call_id``
        Which call a tool message answers (Req 2.12). Required on a tool
        message, because a result that cannot be matched to a request is not
        submittable (Req 2.11), and absent on every other role.

    The role-specific fields are checked rather than merely documented: a tool
    message with no identifier would be refused by the adapter at submission
    time, which is a long way from the line that built it.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"message role must be one of {sorted(ROLES)}, received {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError(
                f"message content must be a string, received {type(self.content).__name__}"
            )

        calls = _as_tool_call_tuple(self.tool_calls, "message")
        if calls and self.role != "assistant":
            raise ValueError(f"only an assistant message may carry tool calls, role is {self.role!r}")
        object.__setattr__(self, "tool_calls", calls)

        if self.role == "tool":
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
                raise ValueError(
                    "a tool message must carry the non-empty tool_call_id of the call it answers"
                )
        elif self.tool_call_id is not None:
            raise ValueError(
                f"only a tool message may carry a tool_call_id, role is {self.role!r}"
            )


@dataclass(frozen=True)
class Usage:
    """The token counts for one model call (Req 2.1, Req 2.2, Req 2.3).

    ``input_tokens`` / ``output_tokens``
        Non-negative integers. Provider-reported when the response includes a
        usage block, estimated when it does not.
    ``estimated``
        ``False`` when the counts came from the provider (Req 2.2), ``True``
        when they came from :func:`estimate_tokens` (Req 2.3). The flag travels
        with the numbers all the way into the trace record and the run summary,
        because a cost computed from an estimate is a different kind of claim
        from one computed from a measurement.

    Non-negativity is enforced here rather than in the cost tracker. The tracker
    must be monotonically increasing across successive reads (Req 3.6), and a
    negative count is the one input that could break that, so it is refused at
    the point where the number enters the system.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False

    def __post_init__(self) -> None:
        _require_non_negative_int(self.input_tokens, "input_tokens")
        _require_non_negative_int(self.output_tokens, "output_tokens")
        if not isinstance(self.estimated, bool):
            raise ValueError(f"estimated must be a bool, received {self.estimated!r}")

    @property
    def total_tokens(self) -> int:
        """Input plus output, which the run summary reports (Req 3.9)."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelReply:
    """What one ``complete`` call returns (Req 2.1, Req 2.13).

    ``text``
        The assistant text, possibly empty. Empty is not an error: a model that
        asks for a tool and says nothing else is the common case.
    ``tool_calls``
        The requested calls, possibly empty. Empty with non-whitespace text is
        the final answer (Req 4.3).
    ``usage``
        The token counts for the call that produced this reply.

    Req 2.13 is the reason both fields are always present and independent. A
    reply carrying text *and* tool calls keeps both: the adapter does not choose
    one, the loop does not drop one. Dropping the text would lose the model's
    stated reasoning from the trace; dropping the calls would strand the run.

    Both fields empty is a legitimate value here and a model error at the loop's
    level (Req 4.13) -- this type records what came back, and the loop decides
    what it means.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError(f"reply text must be a string, received {type(self.text).__name__}")
        object.__setattr__(self, "tool_calls", _as_tool_call_tuple(self.tool_calls, "reply"))
        if not isinstance(self.usage, Usage):
            raise ValueError(f"reply usage must be a Usage, received {type(self.usage).__name__}")

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model asked for at least one tool (Req 4.2)."""
        return bool(self.tool_calls)

    @property
    def has_text(self) -> bool:
        """Whether the text holds at least one non-whitespace character.

        The exact test Req 4.3 and Req 4.13 turn on: whitespace-only text is not
        a final answer, so the question is asked once here and not re-spelled at
        each site that asks it.
        """
        return bool(self.text.strip())

    def as_assistant_message(self) -> Message:
        """This reply as the assistant message to append to the conversation.

        Req 2.12 wants each tool result placed after the assistant message
        carrying its matching tool call, which means the reply has to enter the
        message list before the results do. Keeping the conversion here is what
        guarantees both halves of Req 2.13 survive it.
        """
        return Message(role="assistant", content=self.text, tool_calls=self.tool_calls)


# --------------------------------------------------------------------------- #
# The protocol (Req 2.1)
# --------------------------------------------------------------------------- #


@runtime_checkable
class ModelClient(Protocol):
    """One operation, one return type (Req 2.1).

    ``model_id``
        The model identifier this client calls. Read by the cost tracker to
        price a call (Req 3.2) and named in a failure message (Req 2.7), so it
        is part of the protocol rather than a detail of one adapter.

    ``complete``
        Sends the system prompt, the messages and the tool schemas, and returns
        a normalised :class:`ModelReply`. Everything else -- retries with their
        1 s, 2 s and 4 s delays, the 120 s per-attempt timeout, the terminal
        versus retryable split, tool-call identifier synthesis, usage
        estimation -- happens inside an implementation and is invisible here
        (Req 2.5, Req 2.6, Req 2.7).

    A protocol rather than a base class: the only other implementation is a
    scripted fake in the tests, and requiring it to inherit would buy nothing.

    ``tool_schemas`` is typed as a sequence of mappings because that is what
    ``ToolRegistry.schemas()`` returns -- ``{name, description, parameters}``
    dicts. Each adapter translates that neutral shape into its provider's
    envelope, so the registry does not learn about providers.
    """

    model_id: str

    def complete(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]] = (),
    ) -> ModelReply:
        """Call the model once and return its reply (Req 2.1).

        Raises :class:`ModelClientError` if the call cannot be completed, and
        :class:`UnmatchedToolCallError` if a tool result in ``messages`` refers
        to a call no preceding reply issued, in which case no provider request
        is made (Req 2.11).
        """
        ...


def validate_request(
    system_prompt: Any,
    messages: Sequence[Message],
    tool_schemas: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Check the bounds Req 2.1 states, before any provider request is made.

    Lives here, next to the protocol, so every adapter enforces the same bounds
    with the same message instead of each restating them. Raises
    ``ModelClientError``: an out-of-range request is a caller fault, but it is
    a fault discovered at call time, and the loop's answer to a call that could
    not be made is the same either way -- end the run with the model error
    outcome and an intact trace (Req 4.13).
    """
    if not isinstance(system_prompt, str):
        raise ModelClientError(
            f"system_prompt must be a string, received {type(system_prompt).__name__}."
        )

    count = len(messages)
    if count < MIN_MESSAGES or count > MAX_MESSAGES:
        raise ModelClientError(
            f"a model call accepts between {MIN_MESSAGES} and {MAX_MESSAGES} messages, "
            f"received {count}."
        )
    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            raise ModelClientError(
                f"message at index {index} must be a Message, "
                f"received {type(message).__name__}."
            )

    schema_count = len(tool_schemas)
    if schema_count < MIN_TOOL_SCHEMAS or schema_count > MAX_TOOL_SCHEMAS:
        raise ModelClientError(
            f"a model call accepts between {MIN_TOOL_SCHEMAS} and {MAX_TOOL_SCHEMAS} tool "
            f"schemas, received {schema_count}."
        )


# --------------------------------------------------------------------------- #
# The token estimator (Req 2.3)
# --------------------------------------------------------------------------- #


def estimate_tokens(text: Any) -> int:
    """Estimate the tokens in ``text`` at roughly four characters per token.

    A module-level pure function, which is the point: Req 2.3's fallback path is
    reached only when a provider omits its usage block, so testing it through an
    adapter would mean faking a provider response to reach three lines of
    arithmetic. Here it can be checked directly.

    Rounds up, so any non-empty text costs at least one token, and clamps at
    zero, so the result can never be negative no matter what it is handed
    (Req 2.3 requires a non-negative integer). ``None`` and non-string values
    are coerced rather than refused: this function runs on a fallback path, and
    raising here would turn a usable reply into a failed call.
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:  # pragma: no cover - a hostile __str__
            return 0
    return tokens_from_chars(len(text))


def tokens_from_chars(characters: int) -> int:
    """The same estimate from a character count instead of the text (Req 2.3).

    :func:`estimate_tokens` is this function plus ``len``. It is separate
    because the whole-call estimate sums character counts across many messages
    and would otherwise have to concatenate them into one string to ask the same
    question. Clamped at zero, so a negative count -- which only a caller bug
    could produce -- returns zero rather than a negative token count.
    """
    if not isinstance(characters, int) or isinstance(characters, bool):
        return 0
    if characters <= 0:
        return 0
    return max(0, math.ceil(characters / CHARS_PER_TOKEN))


def estimate_usage(
    system_prompt: Any,
    messages: Sequence[Message] | Iterable[Message] = (),
    reply_text: Any = "",
    reply_tool_calls: Iterable[ToolCall] = (),
) -> Usage:
    """Estimate a whole call's usage from what was submitted and returned.

    Req 2.3: the input estimate counts the system prompt and every submitted
    message, the output estimate counts the returned text and tool calls, and
    the result is flagged ``estimated=True`` so no downstream figure can pass it
    off as measured.

    Tool calls are counted, not ignored. A reply whose only content is a call
    with a long argument payload consumed output tokens, and charging it zero
    would understate the run's cost precisely when the model is working hardest.

    Pure and side-effect free, like :func:`estimate_tokens`, and written not to
    raise for the same reason.
    """
    input_chars = len(_coerce_text(system_prompt))
    for message in messages or ():
        input_chars += _message_chars(message)

    output_chars = len(_coerce_text(reply_text))
    for call in reply_tool_calls or ():
        output_chars += _tool_call_chars(call)

    return Usage(
        input_tokens=tokens_from_chars(input_chars),
        output_tokens=tokens_from_chars(output_chars),
        estimated=True,
    )


def _coerce_text(value: Any) -> str:
    """Render ``value`` as text for counting, never raising."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:  # pragma: no cover - a hostile __str__
        return ""


def _message_chars(message: Any) -> int:
    """The characters one submitted message contributes to the input estimate."""
    content = _coerce_text(getattr(message, "content", message))
    total = len(content)
    for call in getattr(message, "tool_calls", ()) or ():
        total += _tool_call_chars(call)
    identifier = getattr(message, "tool_call_id", None)
    if identifier:
        total += len(_coerce_text(identifier))
    return total


def _tool_call_chars(call: Any) -> int:
    """The characters one tool call contributes, name plus serialised arguments."""
    total = len(_coerce_text(getattr(call, "name", "")))
    arguments = getattr(call, "arguments", None)
    if arguments:
        try:
            # ``sort_keys`` makes the estimate deterministic for the same
            # arguments regardless of insertion order; ``default=str`` keeps a
            # value the decoder produced but ``json`` cannot re-encode from
            # breaking the estimate.
            total += len(json.dumps(arguments, sort_keys=True, default=str))
        except Exception:  # pragma: no cover - an unserialisable payload
            total += len(_coerce_text(arguments))
    error = getattr(call, "arguments_error", None)
    if error:
        total += len(_coerce_text(error))
    return total


# --------------------------------------------------------------------------- #
# Shared validation helpers
# --------------------------------------------------------------------------- #


def _require_non_negative_int(value: Any, name: str) -> None:
    """Req 2.1: token counts are non-negative integers.

    ``bool`` is refused explicitly. ``True`` is an ``int`` in Python, and a
    usage field holding ``True`` would silently become one token.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, received {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, received {value}")


def _as_tool_call_tuple(calls: Any, owner: str) -> tuple[ToolCall, ...]:
    """Normalise ``calls`` to a tuple of ``ToolCall``, or raise.

    A tuple, not a list, because the containing value is frozen and a mutable
    field inside a frozen object is only half a guarantee. A list, tuple or any
    other iterable is accepted at construction and converted, so a caller
    building a reply does not have to remember which one this is.
    """
    if calls is None:
        return ()
    if isinstance(calls, ToolCall):
        raise ValueError(f"{owner} tool_calls must be a sequence of ToolCall, not a single ToolCall")
    if isinstance(calls, (str, bytes, Mapping)):
        raise ValueError(
            f"{owner} tool_calls must be a sequence of ToolCall, received {type(calls).__name__}"
        )
    try:
        items = tuple(calls)
    except TypeError as exc:
        raise ValueError(
            f"{owner} tool_calls must be a sequence of ToolCall, received {type(calls).__name__}"
        ) from exc
    for item in items:
        if not isinstance(item, ToolCall):
            raise ValueError(
                f"{owner} tool_calls must contain only ToolCall values, "
                f"received {type(item).__name__}"
            )
    return items


__all__ = [
    "CHARS_PER_TOKEN",
    "MAX_MESSAGES",
    "MAX_TOOL_SCHEMAS",
    "MIN_MESSAGES",
    "MIN_TOOL_SCHEMAS",
    "ROLES",
    "Message",
    "ModelClient",
    "ModelClientError",
    "ModelReply",
    "Role",
    "ToolCall",
    "UnmatchedToolCallError",
    "Usage",
    "estimate_tokens",
    "estimate_usage",
    "tokens_from_chars",
    "validate_request",
]
