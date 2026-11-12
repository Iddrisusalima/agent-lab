"""The tool registry: the one place that knows which tools exist (Req 5.1, 5.2).

Two consumers, two questions:

* the model client asks ``schemas()`` — "what may the model request?" — and gets
  exactly the registered declarations on every call (Req 4.5, Req 5.2);
* the dispatcher asks ``resolve(name)`` — "which tool is this?" — and gets a
  case-sensitive exact match or nothing at all (Req 5.3, Req 5.4).

Those two answers come from the same mapping, which is the whole point of the
type. If the schemas offered to the model and the tools the dispatcher can
resolve were built from separate sources they could drift, and the model would
be told about a tool that cannot be called, or a tool would be callable without
ever being declared.

The registry holds no configuration, performs no execution and reads no
filesystem. Validation of a tool's declaration already happened in
``ToolSpec.__post_init__``; what is left here is identity: one name, one tool.

**Mutation.** ``register`` is the only way to add a tool and it is called while
the registry is being built, by the factory in ``tools/__init__`` wiring —
never from inside a step. Nothing in a tool result or a model reply reaches it,
and the accessors hand out copies rather than internal state, so the registry's
contents cannot be changed by anything the model says (Req 12.9).
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Iterator

from agent_lab.tools.base import Tool, ToolSpec, ToolSpecError


class DuplicateToolError(ToolSpecError):
    """Two tools were registered under the same name (Req 5.1).

    A ``ToolSpecError`` subclass because it is the same kind of mistake as a
    malformed spec: a developer-facing wiring error, raised while the registry
    is built and therefore before a run starts.
    """


class ToolRegistry:
    """The registered tools, keyed by their exact names (Req 5.1).

    Registration order is preserved and is the order ``names()`` and
    ``schemas()`` report, so the tool list the model sees is stable from run to
    run. A stable order keeps provider prompt caching effective and makes a
    trace file comparable between runs.
    """

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        # A plain dict: insertion-ordered since 3.7, which is the ordering
        # guarantee the accessors rely on.
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    # -- building -------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add ``tool`` under its declared name (Req 5.1).

        Raises ``DuplicateToolError`` if that exact name is already registered.
        Comparison is case-sensitive and exact, matching how the dispatcher
        resolves a requested name (Req 5.3): ``Calculator`` and ``calculator``
        are two different names, so registering both is not a collision — the
        dispatcher would route each to its own tool with no ambiguity.

        Raises ``ToolSpecError`` if the object does not satisfy the ``Tool``
        protocol, which is checked here rather than left to fail at dispatch
        time, when the mistake would surface as a tool error in the middle of a
        run instead of as a startup failure.
        """
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise ToolSpecError(
                f"a registered tool must expose a ToolSpec as its 'spec' attribute, "
                f"received {type(spec).__name__}"
            )
        if not callable(getattr(tool, "execute", None)):
            raise ToolSpecError(f"tool {spec.name!r} must expose a callable 'execute'")

        if spec.name in self._tools:
            raise DuplicateToolError(
                f"a tool named {spec.name!r} is already registered; "
                f"registered names are {list(self._tools)}"
            )
        self._tools[spec.name] = tool

    # -- what the dispatcher asks ---------------------------------------------

    def resolve(self, name: Any) -> Tool | None:
        """Return the tool registered under exactly ``name``, or ``None``.

        Case-sensitive exact match, no normalisation, no fuzzy fallback
        (Req 5.3). ``None`` rather than an exception for an unknown name,
        because an unknown name is an expected model mistake with a defined
        answer: the dispatcher turns it into an error result listing the
        registered names (Req 5.4). Raising here would make the dispatcher
        catch an exception to produce a result it already knows how to build.

        A non-string ``name`` — a model can send anything — resolves to
        ``None`` by the same route as an unknown string, so the caller has one
        case to handle rather than two.
        """
        if not isinstance(name, str):
            return None
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        """Every registered name, in registration order (Req 5.1).

        Used by the dispatcher to list the valid names when it refuses an
        unknown one (Req 5.4), and as the oracle for the check that the model is
        offered exactly the registered tools (Req 4.5). A tuple, so a caller
        cannot edit the registry's view of itself.
        """
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Every registered ``ToolSpec``, in registration order.

        The typed counterpart to ``schemas()``, for callers that want the
        declaration rather than a wire payload — the renderer naming a tool, or
        a test reading ``timeout_seconds``. ``ToolSpec`` is frozen and stores a
        copy of its own schema, so handing these out shares nothing mutable.
        """
        return tuple(tool.spec for tool in self._tools.values())

    # -- what the model client asks -------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """The tool declarations to offer the model, in registration order.

        One entry per registered tool carrying its name, its one-sentence
        description and its parameter schema stating each parameter's type and
        required status (Req 5.2). Provider-neutral on purpose: this is the
        internal shape, and each provider adapter translates it into whatever
        envelope that API wants, so adding a provider does not touch the
        registry.

        ``timeout_seconds`` is deliberately absent. It governs how the
        dispatcher executes a call, which is not something the model decides or
        needs to know.

        Returns a fresh deep copy on every call, so a provider adapter that
        annotates or reorders the list it was given cannot alter what the next
        call offers (Req 12.9). Four tools makes the copy free.
        """
        return [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "parameters": copy.deepcopy(tool.spec.parameters),
            }
            for tool in self._tools.values()
        ]

    # -- conveniences ---------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(tuple(self._tools.values()))

    def __repr__(self) -> str:
        return f"ToolRegistry(names={list(self._tools)})"


__all__ = ["DuplicateToolError", "ToolRegistry"]
