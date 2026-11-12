"""Build the tool registry: the four required tools, wired once (Req 5.1).

The CLI, the eval runner, and any test that needs a real tool layer all call
:func:`build_tool_registry`. It is the only place that knows which tools exist
and what each of them is given, which is what makes "exactly `calculator`,
`search_docs`, `save_note`, `read_notes`" checkable in one place rather than
spread across three entry points that could drift (Req 5.1, Req 4.5).

**Why the collaborators are injected.** The confirmation gate and the retriever
arrive as arguments rather than being constructed here. Both have a cheap
substitute — an auto-approving gate for the eval runner, a scripted retriever for
a test — and both have an expensive real implementation that the CLI builds once
at startup. Keeping their construction out of this function is what lets the
whole tool layer be exercised offline, with no index and no terminal
(design: ports and adapters).

**Why timeouts are decided here.** ``ToolSpec.timeout_seconds`` governs
dispatch: the dispatcher applies no default of its own, and ``None`` means "run
inline on the calling thread with no limit" (Req 5.8). So the configured
``TOOL_TIMEOUT_SECONDS`` has to be handed to each tool that should be bounded,
and this function does it explicitly for all three of them rather than relying
on each constructor's default.

**Why the fifth name is allowed at all.** Req 5.1 fixes the registry at four
tools and Req 18.3 states the one documented exception: with ``WEB_SEARCH=on``
an optional ``web_search`` tool is registered beside them, with the same spec
shape and under the same dispatcher. :func:`_verify` therefore checks a *built*
expectation rather than a fixed one -- the four required names first and in
order, then exactly the optional names this configuration asks for -- which
keeps a wiring mistake a startup failure instead of a missing tool mid-run.
With no stretch variable set the expectation is the original four and nothing
changes about what the model is offered (Req 18.6, Req 18.7).

``save_note`` is the single exception and is given no timeout argument at all.
It derives its own: ``None`` in interactive mode, because Req 9.2 requires
waiting for the operator *without any time limit* and terminal input belongs on
the main thread, and the configured timeout in non-interactive mode, where
nothing waits for a human. Overriding it here would silently reintroduce the
10-second abandon that Req 9.2 rules out, so this function leaves it alone.
"""

from __future__ import annotations

from agent_lab.config import Config
from agent_lab.confirmation import ConfirmationGate
from agent_lab.retrieval.base import Retriever
from agent_lab.tools.base import ToolSpecError
from agent_lab.tools.calculator import CalculatorTool
from agent_lab.tools.read_notes import ReadNotesTool
from agent_lab.tools.registry import ToolRegistry
from agent_lab.tools.save_note import SaveNoteTool
from agent_lab.tools.search_docs import SearchDocsTool
from agent_lab.tools.web_search import WebSearchBackend, WebSearchTool

# Req 5.1: the four names, in the order they are registered and therefore the
# order their schemas are offered to the model. A named constant so the check
# below, the model-client test for Req 4.5, and a reader all consult one list.
REQUIRED_TOOL_NAMES: tuple[str, ...] = (
    "calculator",
    "search_docs",
    "save_note",
    "read_notes",
)

# Req 18.3: the optional names, in the order they would be registered, each
# after all four required ones. Only ``web_search`` exists, and only when
# ``WEB_SEARCH=on``; the tuple is named so that the check below reads as a rule
# rather than as a special case, and so that "which names may appear beyond the
# four" has one answer a reader can find.
OPTIONAL_TOOL_NAMES: tuple[str, ...] = ("web_search",)


def optional_tool_names(config: Config) -> tuple[str, ...]:
    """The optional tool names this configuration registers (Req 18.3, Req 18.6).

    Empty unless a stretch-feature variable is set, which is what makes
    Req 18.6 checkable from outside the factory: with nothing configured the
    registry holds exactly :data:`REQUIRED_TOOL_NAMES`, so the schemas offered
    to the model are byte-for-byte what they were before this feature existed.
    """
    return ("web_search",) if config.web_search_enabled else ()


def build_tool_registry(
    config: Config,
    *,
    gate: ConfirmationGate,
    retriever: Retriever,
    web_search_backend: WebSearchBackend | None = None,
) -> ToolRegistry:
    """Register the four tools and return the registry (Req 5.1).

    :param config: the validated configuration of the run; supplies the corpus
        path, the notes path, the per-tool timeout, and the result character
        limit.
    :param gate: the confirmation gate ``save_note`` asks before writing. The
        model has no channel that reaches it, which is the structural reason an
        injected "save HACKED" instruction cannot write without the operator's
        keystroke (Req 12.6, Req 12.9).
    :param retriever: what ``search_docs`` searches through. Any object with
        ``search(query, k) -> list[(chunk_text, source_name)]`` (Req 7.7).
    :param web_search_backend: the search service the optional ``web_search``
        tool calls, when that tool is registered at all. ``None`` -- the
        default, and what every entry point passes today -- leaves the tool
        answering that no backend is configured, so enabling ``WEB_SEARCH``
        cannot by itself cause a network call (Req 18.3).
    :raises ToolSpecError: if the registry does not end up holding exactly the
        four required names, in order, followed by exactly the optional names
        this configuration asks for.

    The registry is returned fully built and is not mutated afterwards by
    anything in a run: ``register`` is called only here, and nothing reachable
    from a model reply or a tool result can add, remove, or replace a tool
    (Req 12.9).
    """
    registry = ToolRegistry()

    # Req 6: no configuration beyond its timeout. Stateless, so one instance
    # serves every step.
    registry.register(CalculatorTool(timeout_seconds=config.tool_timeout_seconds))

    # Req 8: the corpus path and the result limit come from the config, the
    # search itself from the injected retriever.
    registry.register(
        SearchDocsTool(config, retriever, timeout_seconds=config.tool_timeout_seconds)
    )

    # Req 9: no timeout argument -- see the module docstring. The tool derives
    # ``None`` in interactive mode so the dispatcher runs it inline and the
    # operator has unlimited time to answer (Req 9.2).
    registry.register(SaveNoteTool(config, gate))

    # Req 10: reads one configured path, bounded by the configured timeout.
    registry.register(ReadNotesTool(config, timeout_seconds=config.tool_timeout_seconds))

    # Req 18.3, and the only configuration-dependent line in this function: the
    # fifth tool is registered *after* all four required ones, so the required
    # names keep both their membership and their position, and the schema list
    # the model is offered grows at the end rather than being reordered.
    optional = optional_tool_names(config)
    if config.web_search_enabled:
        registry.register(
            WebSearchTool(
                config,
                web_search_backend,
                timeout_seconds=config.tool_timeout_seconds,
            )
        )

    _verify(registry, optional)
    return registry


def _verify(registry: ToolRegistry, optional: tuple[str, ...] = ()) -> None:
    """Confirm the registry holds the required four, then exactly ``optional``.

    A check rather than an ``assert``: ``python -O`` strips assertions, and this
    is the one guarantee the model client's tool list and the dispatcher's
    resolution both rest on. It runs once at wiring time, so a mistake surfaces
    before the first step rather than as a missing tool mid-run.

    The expected list is built rather than fixed, which keeps the check as
    strict as it was while admitting the one documented exception. Three things
    still fail here: a missing or renamed required tool, a required tool that
    moved out of its position, and an optional tool present without its
    configuration (or absent with it). ``optional=()`` -- the unconfigured case
    of Req 18.6 -- reduces to the original equality against
    :data:`REQUIRED_TOOL_NAMES`, so nothing about the default build is relaxed.
    """
    unsupported = [name for name in optional if name not in OPTIONAL_TOOL_NAMES]
    if unsupported:
        raise ToolSpecError(
            f"the tool registry was asked for unsupported optional tools "
            f"{unsupported}; the supported optional names are "
            f"{list(OPTIONAL_TOOL_NAMES)}"
        )

    expected = REQUIRED_TOOL_NAMES + tuple(optional)
    names = registry.names()
    if names != expected:
        raise ToolSpecError(
            f"the tool registry must hold exactly {list(expected)} in that order, "
            f"but it holds {list(names)}"
        )


__all__ = [
    "OPTIONAL_TOOL_NAMES",
    "REQUIRED_TOOL_NAMES",
    "build_tool_registry",
    "optional_tool_names",
]
