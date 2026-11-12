"""The tool contract: what a tool declares, what it returns, and nothing else.

Three types live here and they are the only things the registry (Req 5.1), the
dispatcher (Req 5.3 - 5.12), the model client (Req 4.5) and the four shipped
tools agree on:

* ``ToolSpec``  — the declaration the registry publishes for a tool (Req 5.2).
* ``ToolResult`` — the single value a dispatch produces, success or error
  (Req 5.9).
* ``Tool``      — the protocol a tool satisfies: a spec and an ``execute``.

This module imports nothing from the rest of the package. Keeping it free of
configuration, redaction and filesystem concerns is what lets a test construct a
spec and a result without standing up a run, and it keeps the dependency arrow
pointing one way: tools depend on the contract, the contract depends on nobody.

The validation in ``__post_init__`` is deliberate. These invariants are
developer-facing — a tool author, not the model, supplies a ``ToolSpec`` — so
failing loudly at construction is right: a malformed schema registered at
startup would otherwise surface as a confusing validation error during a run,
long after the mistake was made. Model-supplied arguments are a different
matter entirely and are validated by the dispatcher against this schema, which
is why nothing here ever raises in response to model input.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Req 5.9: the two status values a dispatch can produce. Named aliases rather
# than bare literals so the dispatcher, the renderer and the trace logger all
# spell the same two strings.
ToolStatus = Literal["success", "error"]

# Req 9.3, Req 9.4, Req 9.6: the confirmation decision carried out of a write so
# the trace record can state it. ``None`` is every tool that asks for nothing.
Approval = Literal["granted", "declined", "auto"]

# Req 5.2: a description is one sentence of at most 200 characters.
MAX_DESCRIPTION_CHARS = 200

# The characters that end a sentence. A description may contain at most one of
# these and only as its final character, which is the strictest reading of "one
# sentence" that can be checked mechanically. The cost is that an abbreviation
# such as "e.g." cannot appear in a description; the benefit is that "one
# sentence" means something enforceable rather than aspirational.
_SENTENCE_TERMINATORS = ".!?"

# The JSON Schema keywords a parameter schema may use at its top level. A
# parameter schema here is always an object schema: tool arguments arrive as a
# name/value mapping, so there is no other shape to support, and restricting the
# keyword set keeps every provider's tool-schema translation trivial.
_ALLOWED_SCHEMA_KEYS = frozenset({"type", "properties", "required", "additionalProperties"})

# The JSON Schema primitive type names a parameter may declare. The dispatcher
# maps each of these onto a Python type check (Req 5.5), so a declaration
# outside this set would be unenforceable.
_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})


class ToolSpecError(ValueError):
    """A tool declared an invalid spec.

    Raised at construction, which in practice means at import or registry build
    time — before a run starts, before a model call, and therefore never in the
    middle of a step.
    """


@dataclass(frozen=True)
class ToolSpec:
    """What the registry publishes about one tool (Req 5.2).

    ``name``
        The identifier the model requests and the dispatcher resolves by
        case-sensitive exact match (Req 5.3).
    ``description``
        One sentence of at most 200 characters stating the tool's purpose.
    ``parameters``
        A JSON Schema object declaring every parameter's name, its type, and
        whether it is required. This is the single source for all three of the
        dispatcher's validation rules: missing required parameter and wrong type
        (Req 5.5), and undeclared parameter name (Req 5.6).
    ``timeout_seconds``
        The per-tool execution limit the dispatcher applies (Req 5.8), or
        ``None`` to run the tool inline on the calling thread with no limit.
        ``None`` exists for exactly one case: interactive ``save_note`` must
        wait on the operator without any time limit (Req 9.2), and terminal
        input belongs on the main thread.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        self._validate_name()
        self._validate_description()
        self._validate_timeout()
        # Store a deep copy, so a caller that keeps and later mutates the dict it
        # passed in cannot change what the registry publishes or what the
        # dispatcher validates against. Req 12.9 wants the registry's contents
        # fixed at construction; a shared mutable schema would be a way around
        # that. A plain dict, not a MappingProxyType: these schemas are handed
        # to ``json.dumps`` on every model call and a mapping proxy is not
        # JSON-serialisable.
        object.__setattr__(self, "parameters", _validated_parameters(self.parameters, self.name))

    # -- validation -----------------------------------------------------------

    def _validate_name(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ToolSpecError("tool name must be a non-empty string")
        if self.name != self.name.strip():
            # Surrounding whitespace would make the case-sensitive exact match
            # of Req 5.3 turn on invisible characters.
            raise ToolSpecError(f"tool name {self.name!r} must not have leading or trailing whitespace")

    def _validate_description(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolSpecError(f"tool {self.name!r} must declare a non-empty description")
        if len(self.description) > MAX_DESCRIPTION_CHARS:
            raise ToolSpecError(
                f"tool {self.name!r} description is {len(self.description)} characters, "
                f"at most {MAX_DESCRIPTION_CHARS} allowed"
            )
        if "\n" in self.description or "\r" in self.description:
            raise ToolSpecError(f"tool {self.name!r} description must be one sentence on one line")
        body = self.description.rstrip()
        interior = body[:-1] if body[-1] in _SENTENCE_TERMINATORS else body
        if any(char in _SENTENCE_TERMINATORS for char in interior):
            raise ToolSpecError(
                f"tool {self.name!r} description must be one sentence: sentence-ending "
                f"punctuation may appear only as the final character"
            )

    def _validate_timeout(self) -> None:
        if self.timeout_seconds is None:
            return
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise ToolSpecError(
                f"tool {self.name!r} timeout_seconds must be a number or None, "
                f"received {self.timeout_seconds!r}"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ToolSpecError(
                f"tool {self.name!r} timeout_seconds must be a finite positive number or None, "
                f"received {self.timeout_seconds!r}"
            )

    # -- accessors the registry and dispatcher read ---------------------------
    #
    # The knowledge of how a parameter schema is laid out stays in this module.
    # The dispatcher asks these questions instead of reaching into the schema
    # dict itself, so the shape is described in exactly one place.

    @property
    def declared_parameters(self) -> tuple[str, ...]:
        """Every declared parameter name, in declaration order (Req 5.6)."""
        return tuple(self.parameters.get("properties", {}))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """The declared parameter names marked required (Req 5.5)."""
        return tuple(self.parameters.get("required", ()))

    def declared_type(self, parameter: str) -> str | list[str] | None:
        """The JSON Schema type declared for ``parameter``, or ``None``.

        A list is returned for a parameter that accepts more than one type — the
        calculator's ``a`` and ``b`` take a number or a decimal literal string
        (Req 6.1). The dispatcher quotes this value back when it reports a type
        mismatch (Req 5.5).
        """
        schema = self.parameters.get("properties", {}).get(parameter)
        if schema is None:
            return None
        return schema.get("type")

    def is_required(self, parameter: str) -> bool:
        """Whether ``parameter`` is declared required (Req 5.5)."""
        return parameter in self.required_parameters


@dataclass(frozen=True)
class ToolResult:
    """The single value one dispatch produces (Req 5.9).

    Every path through the dispatcher returns one of these — a successful
    execution, a validation refusal, a raised exception, a timeout — so the
    agent loop has one value shape to handle and no exception to catch.

    ``status``
        ``"success"`` or ``"error"``.
    ``content``
        The text the loop appends to the conversation and the renderer shows.
        Already scrubbed of secrets and of absolute filesystem paths by the time
        it leaves the dispatcher (Req 5.10).
    ``truncated`` / ``original_length``
        Set together when the dispatcher shortened ``content`` to the configured
        maximum; ``original_length`` is the pre-truncation character count the
        marker states (Req 5.12).
    ``approval``
        The confirmation decision for a tool that asked for one, so the step's
        trace record can carry it (Req 9.3, Req 9.4, Req 9.6). ``None`` for
        every tool that asks for nothing.

    Frozen, because a result travels to three consumers — the conversation, the
    renderer and the trace logger — and none of them should be able to edit what
    the others see.
    """

    status: ToolStatus
    content: str
    truncated: bool = False
    original_length: int | None = None
    approval: Approval | None = None

    def __post_init__(self) -> None:
        if self.status not in ("success", "error"):
            raise ToolSpecError(f"status must be 'success' or 'error', received {self.status!r}")
        if not isinstance(self.content, str):
            raise ToolSpecError(f"content must be a string, received {type(self.content).__name__}")
        if not isinstance(self.truncated, bool):
            raise ToolSpecError(f"truncated must be a bool, received {self.truncated!r}")
        if self.approval is not None and self.approval not in ("granted", "declined", "auto"):
            raise ToolSpecError(
                f"approval must be 'granted', 'declined', 'auto' or None, received {self.approval!r}"
            )
        if self.original_length is not None:
            if isinstance(self.original_length, bool) or not isinstance(self.original_length, int):
                raise ToolSpecError(
                    f"original_length must be an int or None, received {self.original_length!r}"
                )
            if self.original_length < 0:
                raise ToolSpecError(
                    f"original_length must not be negative, received {self.original_length}"
                )
        if self.truncated and self.original_length is None:
            # Req 5.12: the truncation marker has to state the pre-truncation
            # character count, so a truncated result without that count could
            # not be rendered correctly.
            raise ToolSpecError("a truncated result must carry its original_length")

    @property
    def is_error(self) -> bool:
        """Whether this result is an error, for the loop and the failure marker.

        Req 5.11 appends an error result inside the same step; Req 13.7 precedes
        it with a textually distinct failure marker. Both ask the same question,
        so both ask it the same way.
        """
        return self.status == "error"


@runtime_checkable
class Tool(Protocol):
    """What the registry accepts and the dispatcher executes.

    A protocol rather than a base class: a tool is a spec plus a callable, and
    inheritance would buy nothing while making a test double more work to write.

    ``execute`` takes keyword arguments only, because tool arguments arrive from
    the model as a name/value mapping and position has no meaning there. It is
    called only after the dispatcher has resolved the name and validated the
    arguments against ``spec.parameters`` (Req 5.3), so an implementation can
    rely on receiving declared parameter names of the declared types.

    Returning a ``ToolResult`` is the normal way to report a refusal — an
    implementation should prefer an error result over raising. The dispatcher
    does catch exceptions and convert them (Req 5.7), but a result carries a
    message written for the model, and an exception carries one written for a
    developer.
    """

    spec: ToolSpec

    def execute(self, **kwargs: Any) -> ToolResult: ...


def _validated_parameters(parameters: Any, tool_name: str) -> dict[str, Any]:
    """Check a parameter schema's shape and return an independent deep copy.

    Only the fragment of JSON Schema the dispatcher can actually enforce is
    accepted. Anything else is rejected here rather than silently ignored during
    a run, because a declaration the dispatcher does not read is a validation
    rule that does not exist.
    """
    if not isinstance(parameters, dict):
        raise ToolSpecError(
            f"tool {tool_name!r} parameters must be a JSON Schema object, "
            f"received {type(parameters).__name__}"
        )

    unknown = sorted(set(parameters) - _ALLOWED_SCHEMA_KEYS)
    if unknown:
        raise ToolSpecError(
            f"tool {tool_name!r} parameter schema has unsupported keys {unknown}; "
            f"supported keys are {sorted(_ALLOWED_SCHEMA_KEYS)}"
        )

    declared_type = parameters.get("type", "object")
    if declared_type != "object":
        raise ToolSpecError(
            f"tool {tool_name!r} parameter schema must have type 'object', received {declared_type!r}"
        )

    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ToolSpecError(
            f"tool {tool_name!r} parameter schema 'properties' must be an object, "
            f"received {type(properties).__name__}"
        )
    for parameter, schema in properties.items():
        if not isinstance(parameter, str) or not parameter:
            raise ToolSpecError(f"tool {tool_name!r} declares a parameter with a non-string name")
        if not isinstance(schema, dict):
            raise ToolSpecError(
                f"tool {tool_name!r} parameter {parameter!r} must declare a schema object"
            )
        _validate_parameter_type(schema.get("type"), parameter, tool_name)

    required = parameters.get("required", [])
    if not isinstance(required, (list, tuple)) or any(not isinstance(item, str) for item in required):
        raise ToolSpecError(
            f"tool {tool_name!r} parameter schema 'required' must be a list of parameter names"
        )
    undeclared = sorted(set(required) - set(properties))
    if undeclared:
        raise ToolSpecError(
            f"tool {tool_name!r} marks undeclared parameters {undeclared} as required; "
            f"declared parameters are {sorted(properties)}"
        )

    normalised = copy.deepcopy(dict(parameters))
    # Normalise the three keys the dispatcher reads so it never has to ask
    # whether they are present. ``additionalProperties: False`` restates
    # Req 5.6 inside the schema itself, which is what a provider shows the model.
    normalised["type"] = "object"
    normalised["properties"] = normalised.get("properties", {})
    normalised["required"] = list(required)
    normalised.setdefault("additionalProperties", False)
    return normalised


def _validate_parameter_type(declared: Any, parameter: str, tool_name: str) -> None:
    """Require every parameter to declare an enforceable JSON Schema type (Req 5.2)."""
    if declared is None:
        raise ToolSpecError(f"tool {tool_name!r} parameter {parameter!r} must declare a type")
    names = declared if isinstance(declared, list) else [declared]
    if not names:
        raise ToolSpecError(f"tool {tool_name!r} parameter {parameter!r} must declare a type")
    unsupported = sorted({name for name in names if name not in _JSON_TYPES})
    if unsupported:
        raise ToolSpecError(
            f"tool {tool_name!r} parameter {parameter!r} declares unsupported types {unsupported}; "
            f"supported types are {sorted(_JSON_TYPES)}"
        )


__all__ = [
    "MAX_DESCRIPTION_CHARS",
    "Approval",
    "Tool",
    "ToolResult",
    "ToolSpec",
    "ToolSpecError",
    "ToolStatus",
]
