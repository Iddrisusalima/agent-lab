"""The one shipped provider adapter: OpenAI-compatible chat completions (Req 2).

Everything vendor-shaped lives here, behind the ``ModelClient`` protocol defined
in :mod:`agent_lab.model_client`: the SDK construction, the tool envelope, the
message translation, the parsing of a provider tool-call payload, the token
usage block, and the one retry table. The agent loop never imports this module;
it receives whatever :func:`agent_lab.providers.build_model_client` returns and
talks to it through ``complete`` alone (Req 2.8).

Four decisions shape the file.

**The SDK and the clock are parameters.** ``OpenAIClient`` accepts an already
built SDK client and a ``sleep`` callable. In production neither is supplied and
the adapter builds ``openai.OpenAI(api_key=..., max_retries=0)`` and uses
``time.sleep``. In a test both are supplied, so the retry schedule is asserted
against a scripted fake and a recording clock -- no network, no API key, and no
real waiting (task 5.6).

**The SDK does not retry; this module does.** ``max_retries=0`` is passed to the
constructor so there is exactly one retry policy in the process and the observed
attempt count is the policy's attempt count (design: Model client). A per-attempt
timeout of 120 seconds is passed on every request, so a hung attempt becomes a
retryable failure rather than a hung run (Req 2.5).

**One classification table.** :func:`classify_failure` is the single place that
decides whether a failure is retryable or terminal, and it is a module-level
function so the table can be tested directly instead of through four faked
request cycles (Req 2.5, Req 2.6). It classifies by exception type where the SDK
provides one and by HTTP status otherwise, which also lets a fake raise a
plausible error object without importing SDK internals.

**Model-supplied content never raises.** A provider tool call whose argument
payload is not decodable JSON becomes a ``ToolCall`` carrying
``arguments_error``, which the dispatcher turns into a schema error the model can
correct (Req 5.5). The adapter raises only when the *call itself* could not be
made or completed (Req 2.6, Req 2.7, Req 2.11).

**Streaming is an opt-in that rejoins the normal path immediately.** Req 18.2 is
off unless ``STREAM_OUTPUT=on``, and when it is on, ``stream=True`` adds two
keys to the request and :meth:`OpenAIClient._consume_stream` forwards each
content fragment to a sink as it arrives. That method then reassembles the
chunks into the shape of an ordinary response and hands it to the same
``_as_reply``, so the tool-call parsing, the identifier bookkeeping and the
usage rules are one implementation rather than two (Req 18.7). With streaming
off, nothing in this module reads the three streaming constructor arguments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from agent_lab.model_client import (
    Message,
    ModelClientError,
    ModelReply,
    ToolCall,
    UnmatchedToolCallError,
    Usage,
    estimate_usage,
    validate_request,
)

# --------------------------------------------------------------------------- #
# The policy constants Req 2.5 states
# --------------------------------------------------------------------------- #

#: The provider identifier this adapter serves, as it appears in ``PROVIDER``
#: and in every failure message (Req 2.7, Req 2.8).
PROVIDER_NAME = "openai"

# Req 2.5: "does not return a reply within 120 seconds" is a per-attempt bound,
# not a bound on the whole retry sequence. It is passed to the SDK on every
# request, so the SDK itself raises a timeout error the table below classifies
# as retryable.
REQUEST_TIMEOUT_SECONDS = 120.0

# Req 2.5: up to 3 additional attempts after the first, with delays of 1, 2 and
# 4 seconds *before* the respective retries. Four attempts in total, three
# sleeps. The two constants are kept in step by an assertion below rather than
# by a comment, because a mismatch would silently change the policy.
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)
assert len(RETRY_DELAYS_SECONDS) == MAX_RETRIES

# HTTP statuses treated as retryable when an exception carries one: rate
# limiting, request timeout, and every provider-side 5xx (Req 2.5). Statuses in
# the 4xx range other than 408 and 429 describe a request the provider
# understood and refused, so repeating it unchanged cannot help.
_RETRYABLE_STATUSES = frozenset({408, 429})
_TERMINAL_STATUSES = frozenset({400, 401, 403, 404, 405, 409, 410, 413, 414, 415, 422})

# Req 18.2 (optional, off unless STREAM_OUTPUT=on): what is added to the request
# when the adapter is asked to stream. ``include_usage`` makes this API family
# send a final chunk carrying the usage block, which a streamed response
# otherwise omits -- see `OpenAIClient._usage_of` and Req 2.2/2.3 for what
# happens when a gateway ignores the option.
STREAM_OPTIONS: Mapping[str, Any] = {"include_usage": True}


# --------------------------------------------------------------------------- #
# The exception-classification table (Req 2.5, Req 2.6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Classification:
    """How one failed attempt is to be treated.

    ``retryable``
        ``True`` to sleep and try again while attempts remain (Req 2.5),
        ``False`` to raise immediately with no retry (Req 2.6).
    ``label``
        The short phrase naming the class of failure. It becomes the ``reason``
        of the :class:`ModelClientError`, which Req 2.7 requires to describe the
        failure reason alongside the provider and the model identifier.
    """

    retryable: bool
    label: str


def _sdk_exception_types() -> dict[str, type[BaseException]]:
    """The SDK exception classes, by name, or an empty mapping if unavailable.

    Imported lazily and defensively. The adapter is constructible with an
    injected SDK client, and a test that does so should not need the real
    package to be importable; nor should a newer SDK that renamed one class turn
    the whole table into an ``ImportError`` at module import time. Anything the
    table cannot find by name simply falls through to the status-based and
    builtin branches of :func:`classify_failure`.
    """
    try:
        import openai
    except Exception:  # pragma: no cover - the SDK is a declared dependency
        return {}
    found: dict[str, type[BaseException]] = {}
    for name in (
        "APIConnectionError",
        "APITimeoutError",
        "APIResponseValidationError",
        "APIStatusError",
        "AuthenticationError",
        "BadRequestError",
        "InternalServerError",
        "NotFoundError",
        "PermissionDeniedError",
        "RateLimitError",
        "UnprocessableEntityError",
        "OpenAIError",
    ):
        candidate = getattr(openai, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found[name] = candidate
    return found


def classify_failure(exc: BaseException) -> Classification:
    """Decide whether a failed attempt is retried or raised (Req 2.5, Req 2.6).

    This is the single table the design calls for, and it is ordered: the most
    specific match wins.

    ======================================================  ===========
    Class of failure                                        Treatment
    ======================================================  ===========
    timeout (the 120 s per-attempt bound)                   retry
    connection / transport error                            retry
    rate limit (429)                                        retry
    provider-side 5xx                                       retry
    authentication (401)                                    raise now
    permission denied (403)                                 raise now
    unknown model (404)                                     raise now
    invalid request (400, 422)                              raise now
    anything unrecognised                                   raise now
    ======================================================  ===========

    Unrecognised failures are terminal on purpose. A retry costs real seconds
    and, for a call the provider already accepted once, real money; repeating a
    failure nobody has classified is as likely to be repeating a defect in this
    program as it is to be riding out a blip. Terminal is the recoverable
    choice, because the loop ends the run with an intact trace that says what
    happened (Req 4.13).

    Classification is by exception type where the SDK offers one and by HTTP
    status where it does not, so a scripted fake can raise an object carrying
    ``status_code`` and be classified exactly as the real error would be.
    """
    sdk = _sdk_exception_types()

    def is_a(name: str) -> bool:
        kind = sdk.get(name)
        return kind is not None and isinstance(exc, kind)

    # -- retryable by type ---------------------------------------------------
    if is_a("APITimeoutError"):
        return Classification(True, f"the attempt exceeded the {REQUEST_TIMEOUT_SECONDS:.0f}s timeout")
    if is_a("APIConnectionError"):
        return Classification(True, "a transport error reaching the provider")
    if is_a("RateLimitError"):
        return Classification(True, "the provider rate-limited the request")
    if is_a("InternalServerError"):
        return Classification(True, "a provider-side server error")

    # -- terminal by type ----------------------------------------------------
    if is_a("AuthenticationError"):
        return Classification(False, "the provider rejected the credential")
    if is_a("PermissionDeniedError"):
        return Classification(False, "the credential is not permitted to use this model")
    if is_a("NotFoundError"):
        return Classification(False, "the provider does not know this model")
    if is_a("BadRequestError") or is_a("UnprocessableEntityError"):
        return Classification(False, "the provider rejected the request as invalid")
    if is_a("APIResponseValidationError"):
        return Classification(False, "the provider returned a response the SDK could not validate")

    # -- by HTTP status, which is how a fake or an unmapped SDK class lands ---
    status = _status_code_of(exc)
    if status is not None:
        if status in _RETRYABLE_STATUSES:
            return Classification(True, f"the provider returned HTTP {status}")
        if 500 <= status <= 599:
            return Classification(True, f"a provider-side server error (HTTP {status})")
        if status in _TERMINAL_STATUSES:
            return Classification(False, f"the provider refused the request (HTTP {status})")
        return Classification(False, f"the provider returned HTTP {status}")

    # -- builtin transport and timeout failures ------------------------------
    # Reached when something below the SDK raises, or when a fake raises a plain
    # builtin to stand in for a dropped connection.
    if isinstance(exc, TimeoutError):
        return Classification(True, "the attempt timed out")
    if isinstance(exc, (ConnectionError, OSError)):
        return Classification(True, "a transport error reaching the provider")

    return Classification(False, f"an unclassified {type(exc).__name__}: {exc}")


def _status_code_of(exc: BaseException) -> int | None:
    """The HTTP status an exception carries, directly or on its response."""
    for holder in (exc, getattr(exc, "response", None)):
        if holder is None:
            continue
        value = getattr(holder, "status_code", None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


# --------------------------------------------------------------------------- #
# Reading provider payloads without assuming how they are shaped
# --------------------------------------------------------------------------- #


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an object attribute or a mapping key.

    The SDK returns pydantic models; a fake is far easier to write with nested
    dicts; and an OpenAI-compatible gateway may return either once it has been
    through a serialisation round trip. One reader handles all three, so the
    parsing below is written once.
    """
    if source is None:
        return default
    if isinstance(source, Mapping):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    return default if value is None else value


def _text_of(value: Any) -> str:
    """Render provider content as the ``str`` ``ModelReply.text`` requires.

    ``None`` becomes the empty string, which is the normal shape of a reply that
    only asks for a tool. A content list -- the multipart form some
    OpenAI-compatible endpoints return -- is flattened by concatenating the text
    parts, because discarding it would lose the model's stated reasoning from
    the trace (Req 2.13).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            else:
                text = _field(item, "text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class OpenAIClient:
    """``ModelClient`` over an OpenAI-compatible chat completions endpoint.

    One instance serves one run, which is what makes "unique within the run"
    (Req 2.4) and "a preceding reply in the same run" (Req 2.11) checkable here:
    the instance holds the set of tool-call identifiers it has issued and the
    step counter used to synthesise one when the provider sends none.

    :param model_id: the model identifier to call, from ``MODEL_ID``. Reported
        by the cost tracker (Req 3.2) and named in every failure (Req 2.7).
    :param api_key: the credential, used only to construct the SDK client. Never
        logged, never placed in an error message.
    :param client: an already built SDK client. Supplied by tests; when omitted
        the adapter builds ``openai.OpenAI`` with ``max_retries=0``.
    :param sleep: the delay function used between retries, defaulting to
        ``time.sleep``. Injected so the retry schedule can be asserted without
        a test taking seven seconds.
    :param timeout: the per-attempt timeout in seconds (Req 2.5).
    :param stream: ``True`` to ask the provider for incremental output tokens
        and forward each one to ``token_sink`` as it arrives (Req 18.2).
        ``False``, the default, leaves every line of the request and the
        response handling exactly as it was (Req 18.6, Req 18.7).
    :param token_sink: called with each received content fragment while a reply
        is still a candidate final answer. Ignored when ``stream`` is ``False``.
    :param on_retract: called at most once per reply, if content had already
        been forwarded to ``token_sink`` and the same reply then turned out to
        request a tool -- so the caller can disown what it wrote. See
        :meth:`_consume_stream`.
    """

    provider = PROVIDER_NAME

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        client: Any = None,
        sleep: Callable[[float], Any] = time.sleep,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
        stream: bool = False,
        token_sink: Callable[[str], Any] | None = None,
        on_retract: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, received {model_id!r}")
        self.model_id = model_id
        self._sleep = sleep
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._retry_delays = tuple(float(delay) for delay in retry_delays)
        # Req 2.4, Req 2.10: every identifier this client has handed out, for the
        # life of the run.
        self._issued_ids: set[str] = set()
        # Req 4.4 numbers steps from 1, and the synthesised identifier
        # `call-{step}-{index}` reads the same way, so this counts calls issued
        # by this client and is incremented before the request.
        self._step = 0
        # Req 18.2, Req 18.9: all three default to the disabled state, and
        # nothing below `complete` consults them unless `stream` is True.
        self._stream = bool(stream)
        self._token_sink = token_sink
        self._on_retract = on_retract
        self._saw_incremental = False
        self._saw_complete_reply = False
        self._client = client if client is not None else _build_sdk_client(api_key)

    # -- the protocol operation ---------------------------------------------

    def complete(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]] = (),
    ) -> ModelReply:
        """Call the model once and return a normalised reply (Req 2.1).

        The order of work is the order the requirements demand, and every check
        that can refuse the call happens before the request is issued:

        1. the Req 2.1 bounds, via the shared ``validate_request``;
        2. the Req 2.11 tool-result match, which raises
           :class:`UnmatchedToolCallError` and makes no provider call;
        3. translation into the provider's message and tool envelopes, in
           received order with the system prompt first (Req 2.12);
        4. the attempt loop with its classification table (Req 2.5, Req 2.6);
        5. normalisation of the reply, its tool calls and its usage (Req 2.2,
           Req 2.3, Req 2.4, Req 2.13).

        ``messages`` is read and never written: the payload is a fresh list of
        fresh dicts, so a failed call leaves the caller's conversation exactly as
        it was (Req 2.7).
        """
        validate_request(system_prompt, messages, tool_schemas)
        self._reject_unmatched_tool_results(messages)

        payload = self._as_provider_messages(system_prompt, messages)
        tools = self._as_provider_tools(tool_schemas)

        self._step += 1
        response = self._request(payload, tools)
        return self._as_reply(response, system_prompt, messages)

    # -- Req 2.11 / Req 2.12: validate before issuing -----------------------

    def _reject_unmatched_tool_results(self, messages: Sequence[Message]) -> None:
        """Refuse a submission whose tool results do not match issued calls.

        Two conditions are checked, both before any request is made:

        *Unmatched* (Req 2.11): a tool message whose ``tool_call_id`` is not in
        the set this client has issued during the run raises
        :class:`UnmatchedToolCallError` naming the identifier.

        *Mis-ordered* (Req 2.12): a tool message that appears **before** the
        assistant message carrying its matching call raises
        :class:`ModelClientError`. The adapter does not reorder the caller's
        list -- reordering would hide a bookkeeping fault in the loop and send a
        conversation the model was never shown -- so it validates instead, and
        the provider's own stricter rejection of the same payload never happens.
        """
        assistant_positions: dict[str, int] = {}
        for position, message in enumerate(messages):
            if message.role == "assistant":
                for call in message.tool_calls:
                    assistant_positions.setdefault(call.id, position)

        for position, message in enumerate(messages):
            if message.role != "tool":
                continue
            identifier = message.tool_call_id
            if identifier not in self._issued_ids:
                raise UnmatchedToolCallError(identifier, self._issued_ids)
            declared_at = assistant_positions.get(identifier)
            if declared_at is not None and declared_at > position:
                raise ModelClientError(
                    f"the tool result for tool_call_id {identifier!r} is at position "
                    f"{position} but the assistant message requesting it is at position "
                    f"{declared_at}; a tool result must follow the reply that asked for "
                    f"it. No provider call was made.",
                    provider=self.provider,
                    model_id=self.model_id,
                    reason="tool results submitted out of order",
                )

    # -- translation into the provider's shapes (Req 2.12) ------------------

    def _as_provider_messages(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> list[dict[str, Any]]:
        """The wire payload: system prompt first, then the caller's order.

        Req 2.12 in one function. The system prompt leads every call, the
        caller's messages follow in the order received, and each tool result
        keeps its position after the assistant message carrying its matching
        call -- which is where the loop put it and where
        ``_reject_unmatched_tool_results`` has just confirmed it is.
        """
        payload: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if message.role == "system":
                # A caller-supplied system message stays where it was placed; the
                # prompt above is still first, which is all Req 2.12 asks.
                payload.append({"role": "system", "content": message.content})
            elif message.role == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
            elif message.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, default=str),
                            },
                        }
                        for call in message.tool_calls
                    ]
                    # This provider family rejects an assistant message that has
                    # tool calls and an empty string as content; null is the
                    # accepted form for "asked for a tool, said nothing".
                    if not message.content:
                        entry["content"] = None
                payload.append(entry)
            else:
                payload.append({"role": "user", "content": message.content})
        return payload

    def _as_provider_tools(
        self, tool_schemas: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Wrap the registry's neutral schemas in this provider's envelope.

        ``ToolRegistry.schemas()`` yields ``{name, description, parameters}``
        (Req 5.2); this provider wants each of those inside a ``function`` key.
        The translation lives here so the registry never learns about providers.

        Returns ``None`` rather than an empty list when no tools are offered,
        because this API rejects ``tools: []``.
        """
        if not tool_schemas:
            return None
        envelopes: list[dict[str, Any]] = []
        for index, schema in enumerate(tool_schemas):
            if not isinstance(schema, Mapping):
                raise ModelClientError(
                    f"tool schema at index {index} must be a mapping, "
                    f"received {type(schema).__name__}.",
                    provider=self.provider,
                    model_id=self.model_id,
                    reason="malformed tool schema",
                )
            function: dict[str, Any] = {
                "name": schema.get("name", ""),
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
            }
            envelopes.append({"type": "function", "function": function})
        return envelopes

    # -- the attempt loop (Req 2.5, Req 2.6, Req 2.7) -----------------------

    def _request(self, payload: list[dict[str, Any]], tools: Any) -> Any:
        """Issue the call, retrying per the classification table.

        One first attempt plus ``MAX_RETRIES`` more, sleeping 1 s, 2 s and 4 s
        before the second, third and fourth (Req 2.5). A terminal
        classification raises on the spot without sleeping (Req 2.6). A failure
        on the final attempt raises a :class:`ModelClientError` naming the
        provider, the model identifier and the reason, with no partial reply
        (Req 2.7). The original exception is chained, so a traceback still shows
        what the SDK actually raised.
        """
        attempts = self._max_retries + 1
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": payload,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = tools
            # "auto" is this family's default when tools are present; stating it
            # keeps behaviour the same across OpenAI-compatible gateways that
            # default differently.
            kwargs["tool_choice"] = "auto"

        if self._stream:
            # Req 18.2: the only two lines of the request that streaming
            # changes. With `stream` False -- the default and the whole of the
            # required product -- the payload is byte-for-byte what it was.
            kwargs["stream"] = True
            kwargs["stream_options"] = dict(STREAM_OPTIONS)

        last: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                if self._stream:
                    return self._consume_stream(self._create(**kwargs))
                return self._create(**kwargs)
            except (ModelClientError, KeyboardInterrupt, SystemExit):
                # Not provider failures: a misconfigured adapter and an operator
                # interrupt both pass straight through unretried.
                raise
            except BaseException as exc:  # noqa: BLE001 - classified immediately below
                last = exc
                verdict = classify_failure(exc)
                if not verdict.retryable:
                    raise ModelClientError(
                        "the model call failed and was not retried.",
                        provider=self.provider,
                        model_id=self.model_id,
                        reason=verdict.label,
                    ) from exc
                if attempt == attempts:
                    break
                self._sleep(self._retry_delays[min(attempt, len(self._retry_delays)) - 1])

        raise ModelClientError(
            f"the model call failed after {attempts} attempts "
            f"({self._max_retries} retries).",
            provider=self.provider,
            model_id=self.model_id,
            reason=classify_failure(last).label if last is not None else "unknown",
        ) from last

    def _create(self, **kwargs: Any) -> Any:
        """The one line that touches the SDK, isolated so a fake can stand in."""
        return self._client.chat.completions.create(**kwargs)

    # -- streaming (Req 18.2, Req 18.9 -- optional, off by default) ----------

    def _consume_stream(self, stream: Any) -> Any:
        """Forward output tokens as they arrive, and reassemble the response.

        The return value is a plain dict in the shape of a *non-streaming*
        response, which :meth:`_as_reply` then normalises. That is the point of
        doing it this way: the text flattening, the tool-call identifier
        claiming of Req 2.4 and the usage handling of Req 2.2 and Req 2.3 are
        the same code for a streamed reply as for an ordinary one, so streaming
        cannot drift from the required behaviour (Req 18.7).

        **Which tokens are forwarded.** Content fragments are forwarded to the
        sink the instant they arrive (Req 18.2) *while the reply is still a
        candidate final answer*. A tool-call delta ends that: Req 18.2 ties the
        written tokens to the answer the loop returns, and the loop's answer is
        the text of a reply that requests no tool, so a reply that requests one
        must not have its prose presented as an answer. Tool-call deltas are
        examined before content in the same chunk, which is how a reply that
        speaks and calls in one chunk forwards nothing. For the one ordering
        this cannot anticipate -- prose first, a tool request several chunks
        later -- ``on_retract`` is called, because by then the tokens have been
        written and the honest move is to say so rather than to pretend
        (:class:`~agent_lab.render.AnswerStreamer` carries that through).

        **Req 18.9.** A provider that ignores ``stream=True`` hands back a
        complete response object rather than an iterable of chunks. That is not
        an error: the object is returned unchanged for :meth:`_as_reply` to
        normalise, and :attr:`streaming_unavailable` becomes ``True`` so the CLI
        can print the answer in one write and say that streaming was
        unavailable.

        **A failure mid-stream is terminal once a token has been written.**
        Retrying would re-emit tokens the operator has already read and break
        the concatenation of Req 18.2, so it is raised as a
        :class:`ModelClientError`, which the attempt loop above passes through
        unretried. A failure before the first forwarded token is left to that
        loop to classify and retry as usual (Req 2.5).
        """
        if not _is_chunk_stream(stream):
            self._saw_complete_reply = True
            return stream

        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage: Any = None
        chunks = 0
        forwarded = 0
        suppressed = False

        try:
            for chunk in stream:
                chunks += 1
                chunk_usage = _field(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = _field(chunk, "choices", ()) or ()
                if not choices:
                    continue
                choice = choices[0]
                delta = _field(choice, "delta", None)
                if delta is None:
                    # A gateway that answered a streaming request with one
                    # complete response, wrapped in an iterable. Same case as a
                    # non-iterable reply: Req 18.9, not a failure.
                    if _field(choice, "message", None) is not None and forwarded == 0:
                        self._saw_complete_reply = True
                        return chunk
                    continue
                self._saw_incremental = True

                raw_calls = _field(delta, "tool_calls", ()) or ()
                if raw_calls:
                    if forwarded and not suppressed and self._on_retract is not None:
                        self._on_retract()
                    suppressed = True
                    _merge_tool_call_deltas(calls, raw_calls)

                content = _text_of(_field(delta, "content", ""))
                if content:
                    text_parts.append(content)
                    if not suppressed and self._token_sink is not None:
                        # Req 18.2: on receipt, with no buffering here. The
                        # renderer flushes, so the token is on screen when this
                        # returns.
                        self._token_sink(content)
                        forwarded += 1
        except (ModelClientError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - classified right here
            if forwarded:
                raise ModelClientError(
                    "the response stream failed after output tokens had already "
                    "been written, so the call was not retried.",
                    provider=self.provider,
                    model_id=self.model_id,
                    reason=classify_failure(exc).label,
                ) from exc
            raise

        if chunks == 0:
            # The call succeeded and delivered nothing. No delay will change
            # that, which is the same judgement `_as_reply` makes about a
            # response carrying no choices.
            raise ModelClientError(
                "the provider returned an empty response stream.",
                provider=self.provider,
                model_id=self.model_id,
                reason="empty response",
            )

        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"] or None,
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for _, slot in sorted(calls.items())
            ]
        return {"choices": [{"message": message}], "usage": usage}

    # -- normalising the response (Req 2.2, Req 2.3, Req 2.4, Req 2.13) -----

    def _as_reply(
        self,
        response: Any,
        system_prompt: str,
        messages: Sequence[Message],
    ) -> ModelReply:
        """Turn the provider's response into a :class:`ModelReply`.

        Req 2.13 is the shape of this function: ``text`` and ``tool_calls`` are
        read independently and both are returned, so a reply that comments on
        what it is about to do keeps the comment and the call.
        """
        choices = _field(response, "choices", ()) or ()
        if not choices:
            # Nothing to normalise and nothing to retry: the call succeeded and
            # returned no candidate reply, which no delay will change.
            raise ModelClientError(
                "the provider returned a response with no choices.",
                provider=self.provider,
                model_id=self.model_id,
                reason="empty response",
            )
        message = _field(choices[0], "message", None)
        text = _text_of(_field(message, "content", ""))
        tool_calls = self._as_tool_calls(_field(message, "tool_calls", ()) or ())
        usage = self._usage_of(response, system_prompt, messages, text, tool_calls)
        return ModelReply(text=text, tool_calls=tool_calls, usage=usage)

    def _as_tool_calls(self, raw_calls: Any) -> tuple[ToolCall, ...]:
        """Convert provider tool requests into internal ones (Req 2.4, Req 2.10).

        For each request: take the provider's identifier when it sent a usable
        one, otherwise synthesise ``call-{step}-{index}`` -- ``step`` being this
        client's 1-based call number and ``index`` the 0-based position of the
        request within this reply -- then record the identifier in the per-run
        set so a later tool result can be matched against it.

        The argument payload arrives from this provider family as a JSON string.
        It is decoded into a dict; a payload that is not decodable, or decodes to
        something other than an object, produces a ``ToolCall`` with empty
        arguments and ``arguments_error`` set. That is deliberate: the model
        getting its own arguments wrong is a correctable mistake the dispatcher
        reports back to it (Req 5.5), not a reason to end the run.
        """
        calls: list[ToolCall] = []
        for index, raw in enumerate(raw_calls):
            function = _field(raw, "function", None)
            name = _field(function, "name", "")
            if not isinstance(name, str):
                name = str(name)
            arguments, error = _decode_arguments(_field(function, "arguments", None))
            identifier = self._claim_id(_field(raw, "id", None), index)
            calls.append(
                ToolCall(
                    id=identifier,
                    name=name,
                    arguments=arguments,
                    arguments_error=error,
                )
            )
        return tuple(calls)

    def _claim_id(self, provider_id: Any, index: int) -> str:
        """Settle on an identifier unique within the run, and record it.

        The provider's own identifier is preferred, because echoing it back on
        the tool result is what lets the provider match the two. It is refused
        only when it is unusable (absent or empty) or when it would collide with
        an identifier already issued in this run, since Req 2.4 requires
        uniqueness within the run and the matching in Req 2.11 would otherwise
        be ambiguous. In either case the synthesised form is used, and a
        synthesised form that also collides gains a numeric suffix rather than
        silently reusing an identifier.
        """
        candidate = provider_id if isinstance(provider_id, str) else None
        if candidate is not None:
            candidate = candidate.strip() or None
        if candidate is None or candidate in self._issued_ids:
            candidate = f"call-{self._step}-{index}"
        if candidate in self._issued_ids:
            suffix = 2
            while f"{candidate}-{suffix}" in self._issued_ids:
                suffix += 1
            candidate = f"{candidate}-{suffix}"
        self._issued_ids.add(candidate)
        return candidate

    def _usage_of(
        self,
        response: Any,
        system_prompt: str,
        messages: Sequence[Message],
        text: str,
        tool_calls: tuple[ToolCall, ...],
    ) -> Usage:
        """Provider-reported usage, or the estimate when it is absent.

        Req 2.2: when the response carries a usage block, its ``prompt_tokens``
        and ``completion_tokens`` are reported as they are, with
        ``estimated=False``.

        Req 2.3: when the block is absent -- or present but not holding two
        usable non-negative integers, which is the same thing for this purpose
        -- the counts come from :func:`estimate_usage` over what was submitted
        and what came back, with ``estimated=True``. Partially trusting a
        half-filled block would report a measured input count beside a zero
        output count and flag the pair as measured, which is worse than an
        estimate that says it is one.

        **Streaming changes nothing here, by design.** A streamed response
        carries no usage block of its own, which is why the request asks for one
        with :data:`STREAM_OPTIONS` and why :meth:`_consume_stream` carries the
        block from whichever chunk states it into the reassembled response. A
        provider that honours the option reports measured counts exactly as it
        would unstreamed (Req 2.2); one that ignores it lands on the
        ``estimate_usage`` branch below with ``estimated=True``, which is what
        Req 2.3 asks for and what the cost tracker's unavailable-token-count
        warning (Req 3.4) then reports to the operator.
        """
        usage = _field(response, "usage", None)
        if usage is not None:
            prompt = _as_token_count(_field(usage, "prompt_tokens", None))
            completion = _as_token_count(_field(usage, "completion_tokens", None))
            if prompt is not None and completion is not None:
                return Usage(input_tokens=prompt, output_tokens=completion, estimated=False)
        return estimate_usage(system_prompt, messages, text, tool_calls)

    # -- introspection -------------------------------------------------------

    @property
    def issued_ids(self) -> frozenset[str]:
        """Every tool-call identifier issued in this run (Req 2.4, Req 2.10)."""
        return frozenset(self._issued_ids)

    @property
    def calls_issued(self) -> int:
        """How many provider calls this client has started, for the identifiers."""
        return self._step

    @property
    def streaming_requested(self) -> bool:
        """Whether this client was built to stream output tokens (Req 18.2)."""
        return self._stream

    @property
    def streamed_incrementally(self) -> bool:
        """Whether the provider has actually sent an incremental delta."""
        return self._saw_incremental

    @property
    def streaming_unavailable(self) -> bool:
        """Req 18.9: streaming was asked for and the provider does not do it.

        ``True`` only once a reply has come back complete rather than in
        deltas, so the CLI reports the fact after the run rather than guessing
        at it beforehand.
        """
        return self._stream and self._saw_complete_reply

    def __repr__(self) -> str:
        return f"OpenAIClient(provider={self.provider!r}, model_id={self.model_id!r})"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _decode_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Decode a provider argument payload into a dict, or report why not.

    Returns ``(arguments, arguments_error)``. The error text is written for the
    model, because that is who reads it: the dispatcher puts it in the tool
    result so the next reply can correct the call (Req 5.5).
    """
    if raw is None:
        return {}, None
    if isinstance(raw, Mapping):
        # Already decoded, which is how some OpenAI-compatible gateways and most
        # hand-written fakes present arguments.
        return dict(raw), None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return {}, f"the arguments were not valid UTF-8 ({exc})"
    if not isinstance(raw, str):
        return {}, (
            f"the arguments must be a JSON object, received {type(raw).__name__}"
        )
    if not raw.strip():
        # An empty payload means "no arguments", which is a legitimate call to a
        # tool whose parameters are all optional. Any missing required parameter
        # is the dispatcher's error to report (Req 5.5), not this function's.
        return {}, None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, f"the arguments were not valid JSON ({exc})"
    if not isinstance(decoded, Mapping):
        return {}, (
            f"the arguments must be a JSON object, received "
            f"{type(decoded).__name__}: {raw[:200]}"
        )
    return dict(decoded), None


def _is_chunk_stream(value: Any) -> bool:
    """Whether a streaming request came back as an iterable of chunks (Req 18.9).

    A complete response -- the SDK's ``ChatCompletion``, or the dict a fake or a
    gateway returns -- is not a chunk stream, and Req 18.9 is the answer to it.
    Strings, bytes and mappings are excluded explicitly: all three are iterable
    and none is a stream of chunks.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    return hasattr(value, "__iter__") or hasattr(value, "__next__")


def _merge_tool_call_deltas(
    accumulated: dict[int, dict[str, str]], raw_calls: Any
) -> None:
    """Fold one chunk's tool-call deltas into the calls assembled so far.

    This API family streams a tool call across chunks: the first carries the
    index, the identifier and the function name, and later ones carry further
    slices of the argument JSON for the same index. Arguments are therefore
    concatenated and the identifier and the name are taken from the first chunk
    that states them -- a name is not concatenated, because a provider that
    repeats it in full would otherwise produce ``calculatorcalculator``.

    The ``index`` field is trusted when it is an integer and the position within
    the chunk is used otherwise, so a reply requesting two tools keeps them
    apart even from a gateway that omits the field.
    """
    for position, raw in enumerate(raw_calls):
        index = _field(raw, "index", None)
        if isinstance(index, bool) or not isinstance(index, int):
            index = position
        slot = accumulated.setdefault(index, {"id": "", "name": "", "arguments": ""})
        identifier = _field(raw, "id", None)
        if isinstance(identifier, str) and identifier.strip() and not slot["id"]:
            slot["id"] = identifier
        function = _field(raw, "function", None)
        name = _field(function, "name", None)
        if isinstance(name, str) and name and not slot["name"]:
            slot["name"] = name
        arguments = _field(function, "arguments", None)
        if isinstance(arguments, str):
            slot["arguments"] += arguments


def _as_token_count(value: Any) -> int | None:
    """A non-negative integer token count, or ``None`` if the value is unusable."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _build_sdk_client(api_key: str | None) -> Any:
    """Construct the real SDK client with its own retrying disabled.

    ``max_retries=0`` is the whole point of this function being separate: the
    retry policy is the table in this module, and an SDK that quietly retried
    underneath it would make the attempt count untestable and the 1/2/4-second
    schedule a fiction (design: Model client).

    The timeout is passed per request rather than here, because Req 2.5's 120
    seconds is a per-attempt bound and the request-level argument is what the
    attempt loop controls.
    """
    if not api_key or not str(api_key).strip():
        # Reached only if something bypassed the factory, which refuses an empty
        # credential before step 1 (Req 2.9). Raised as a ModelClientError here
        # because at this point no configuration error can be reported.
        raise ModelClientError(
            "no API key was supplied to the provider adapter.",
            provider=PROVIDER_NAME,
            reason="missing credential",
        )
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - the SDK is a declared dependency
        raise ModelClientError(
            "the openai package is not installed; install the project "
            "dependencies from pyproject.toml.",
            provider=PROVIDER_NAME,
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    return OpenAI(api_key=api_key, max_retries=0)


__all__ = [
    "MAX_RETRIES",
    "PROVIDER_NAME",
    "REQUEST_TIMEOUT_SECONDS",
    "RETRY_DELAYS_SECONDS",
    "STREAM_OPTIONS",
    "Classification",
    "OpenAIClient",
    "classify_failure",
]
