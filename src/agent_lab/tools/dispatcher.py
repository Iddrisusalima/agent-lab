"""The one path from the agent loop into tool code (Req 5.3 - Req 5.12).

``ToolDispatcher.dispatch(call)`` takes whatever the model asked for and returns
exactly one :class:`~agent_lab.tools.base.ToolResult`. It returns one for a
successful execution, for an unknown tool name, for arguments that do not match
the declared schema, for a tool that raised, for a tool that ran past its
timeout, and for a fault inside the dispatcher itself. Nothing propagates to the
loop (Req 5.9), so a bad model reply becomes an observation the model can
correct rather than a crash that ends the run.

The order inside ``dispatch`` is fixed and is the reason this module exists:

1. **resolve** the name by case-sensitive exact match (Req 5.3, Req 5.4);
2. **validate** the arguments against the declared schema (Req 5.5, Req 5.6) --
   no tool code has run at this point, which is the whole of Req 5.3's second
   clause;
3. **execute** under the tool's timeout (Req 5.8);
4. **catch** anything raised and name the exception type and message (Req 5.7);
5. **scrub** secrets and absolute filesystem paths (Req 5.10), then **truncate**
   past the configured limit with a marker giving the pre-truncation character
   count (Req 5.12).

Steps 1 and 2 run before step 3 and both have to pass, which is why validation
lives here and not in each tool: four tools would otherwise each reimplement the
same three refusals, and a tool that forgot one would be executed with
arguments the schema never allowed.

**Why a thread and not an alarm.** The classic timeout recipe,
``signal.alarm`` with a ``SIGALRM`` handler, does not exist on Windows, and
``signal.signal`` cannot be called off the main thread anywhere. The host is
Windows, so the dispatcher submits the execution to a single-worker
``ThreadPoolExecutor`` and waits with ``future.result(timeout=...)``. On expiry
it stops waiting, shuts the executor down without waiting, and returns an error
naming the timeout in seconds. The requirement's word is "abandon", and
abandoning is exactly what this does -- a Python thread cannot be killed from
outside. Two things keep that honest: every shipped tool is written to finish in
milliseconds, and the only one that can block for long is the one that waits on
the operator, which is precisely the tool that opts out of the timeout.

**Why a tool may opt out.** ``spec.timeout_seconds is None`` means "run inline
on the calling thread, no future, no limit". Interactive ``save_note`` has to
wait for the operator with no time limit (Req 9.2), which a 10-second abandon
would contradict, and terminal input belongs on the main thread anyway.

**Totality.** ``dispatch`` accepts any object. A ``ToolCall`` has ``name`` and
``arguments``; so does a plain dict; a string, ``None``, or an integer has
neither and is answered with the unknown-name error. The body is guarded by a
single ``except BaseException`` with a return in every branch (Req 5.9), and the
scrubbing helpers are written not to raise, so the guard is the backstop rather
than the mechanism.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Mapping

from agent_lab.redaction import redact
from agent_lab.tools.base import Approval, Tool, ToolResult, ToolSpec
from agent_lab.tools.registry import ToolRegistry

# Req 5.12: the documented default maximum result length, and Req 5.8's
# documented default timeout. Both are normally supplied by ``Config``; these
# constants exist so a dispatcher can be built without one, and so the default
# is written down once in code as well as in `.env.example`.
DEFAULT_MAX_RESULT_CHARS = 4_000
DEFAULT_TIMEOUT_SECONDS = 10.0

# The repository root: ``src/agent_lab/tools/dispatcher.py`` -> parents[3].
# Req 5.10 wants filesystem locations expressed relative to the project
# directory, and this is the directory they are expressed relative to.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Req 5.12: appended after the truncated content. It states that truncation
# happened and the character count before it, which is the only way the model
# can tell "this is the whole answer" from "this is the first 4000 characters".
TRUNCATION_MARKER = (
    "\n[truncated: the result was {original} characters and was cut to the "
    "configured limit of {limit}]"
)


# --------------------------------------------------------------------------- #
# Type checking against the declared JSON Schema types (Req 5.5)
# --------------------------------------------------------------------------- #

# ``ToolSpec`` restricts a declared type to these seven names, so every
# declaration is enforceable and this table is total over what can be declared.
#
# ``bool`` is excluded from "number" and "integer" explicitly: ``True`` is an
# ``int`` in Python, and a model that sends ``true`` for a numeric parameter has
# made a mistake worth reporting rather than silently reading as 1.
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    # A JSON array arrives as a list; a tuple is the same thing in Python terms
    # and is accepted so a caller constructing a call by hand is not refused on
    # a technicality.
    "array": lambda value: isinstance(value, (list, tuple)),
    "object": lambda value: isinstance(value, Mapping),
    "null": lambda value: value is None,
}


def _json_type_name(value: Any) -> str:
    """Name ``value``'s type in the vocabulary the schema uses.

    The error message has to say what arrived as well as what was declared, and
    saying it in JSON Schema words lets the model compare the two directly
    instead of translating from Python type names.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _declared_type_text(declared: str | list[str] | None) -> str:
    """Render a declared type for a message: one name, or a list of alternatives."""
    if declared is None:
        return "no declared type"
    if isinstance(declared, list):
        if not declared:
            return "no declared type"
        if len(declared) == 1:
            return f"type {declared[0]!r}"
        return "type " + " or ".join(repr(name) for name in declared)
    return f"type {declared!r}"


def _matches_declared_type(value: Any, declared: str | list[str] | None) -> bool:
    """Whether ``value`` satisfies the declared type (Req 5.5).

    A list of names means the parameter accepts any one of them -- the
    calculator's ``a`` and ``b`` take a number or a decimal literal string
    (Req 6.1) -- so matching any alternative is a match. An unknown name cannot
    be enforced, and refusing a value for a rule that does not exist would be
    worse than accepting it, so an unrecognised declaration passes.
    """
    if declared is None:
        return True
    names = declared if isinstance(declared, list) else [declared]
    if not names:
        return True
    for name in names:
        check = _TYPE_CHECKS.get(name)
        if check is None or check(value):
            return True
    return False


def _parameter_text(spec: ToolSpec, parameter: str) -> str:
    """``"query (string, required)"`` -- the declaration quoted back (Req 5.5)."""
    declared = _declared_type_text(spec.declared_type(parameter)).removeprefix("type ")
    status = "required" if spec.is_required(parameter) else "optional"
    return f"{parameter} ({declared}, {status})"


def _declared_list_text(spec: ToolSpec) -> str:
    """Every declared parameter with its type and required status (Req 5.5, Req 5.6)."""
    declared = spec.declared_parameters
    if not declared:
        return "this tool declares no parameters"
    return ", ".join(_parameter_text(spec, name) for name in declared)


# --------------------------------------------------------------------------- #
# Reading the call, whatever shape it arrives in
# --------------------------------------------------------------------------- #


def _call_field(call: Any, *names: str) -> Any:
    """Read the first present field of ``call`` named by ``names``.

    The dispatcher is defined against a structural input -- a name and an
    argument mapping -- rather than against the ``ToolCall`` dataclass, so it
    does not depend on the model client module and can be tested with a plain
    dict or a small stand-in. Attribute access first, because that is the real
    shape; mapping access second, because a decoded provider payload is a dict.

    ``getattr`` is wrapped: an arbitrary object may define a property that
    raises, and a call object that misbehaves must not be the thing that breaks
    totality.
    """
    for name in names:
        try:
            if hasattr(call, name):
                return getattr(call, name)
        except Exception:  # pragma: no cover - a hostile descriptor
            continue
    if isinstance(call, Mapping):
        for name in names:
            try:
                if name in call:
                    return call[name]
            except Exception:  # pragma: no cover - a hostile mapping
                continue
    return None


def _requested_name(call: Any) -> Any:
    """The tool name the call asks for, in whatever type it arrived as.

    Returned unconverted: ``registry.resolve`` answers ``None`` for a non-string
    name, and the unknown-name error quotes the value as received so the model
    sees what it actually sent (Req 5.4).
    """
    return _call_field(call, "name", "tool_name", "tool")


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #


class ToolDispatcher:
    """Validates, times, executes, and scrubs one tool call (Req 5.3 - Req 5.12).

    ``registry``
        The tools that exist. The dispatcher resolves through it and quotes its
        names when it refuses an unknown one, so the model is never told about a
        tool the dispatcher cannot reach (Req 5.4).
    ``config``
        Optional :class:`~agent_lab.config.Config`, read for
        ``tool_result_max_chars`` (Req 5.12) and for ``api_key``, the secret
        scrubbed from every result (Req 5.10, Req 1.11). Passing the keyword
        arguments instead overrides whatever the config says, which is what the
        tests do.

    Stateless between calls by construction: nothing a tool returns is retained,
    so one call cannot influence the next, and the limits are fixed at
    construction where no model output can reach them (Req 12.9).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Any | None = None,
        *,
        max_result_chars: int | None = None,
        secret: str | None = None,
        project_root: Path | str | None = None,
    ) -> None:
        self._registry = registry

        resolved_max = max_result_chars
        if resolved_max is None:
            resolved_max = getattr(config, "tool_result_max_chars", None)
        if not isinstance(resolved_max, int) or isinstance(resolved_max, bool) or resolved_max < 1:
            resolved_max = DEFAULT_MAX_RESULT_CHARS
        self._max_result_chars: int = resolved_max

        resolved_secret = secret if secret is not None else getattr(config, "api_key", None)
        self._secret: str | None = resolved_secret if isinstance(resolved_secret, str) else None

        root = project_root if project_root is not None else PROJECT_ROOT
        try:
            self._project_root = Path(root).resolve()
        except Exception:  # pragma: no cover - an unresolvable root
            self._project_root = PROJECT_ROOT

    # -- the single entry point ----------------------------------------------

    def dispatch(self, call: Any) -> ToolResult:
        """Return exactly one ``ToolResult`` for ``call``, never raising (Req 5.9).

        One ``try`` around the whole body with a return in every branch. The
        ``except BaseException`` is deliberate: the requirement is that the loop
        receives a result for every call it makes, and an escaping
        ``KeyboardInterrupt`` from inside a tool would break that promise for
        the one call the operator interrupted. The loop's own bounds, not an
        exception from here, are what end a run.
        """
        try:
            name = _requested_name(call)

            # 1. Resolve (Req 5.3). Case-sensitive exact match, no fallback.
            tool = self._registry.resolve(name)
            if tool is None:
                return self._unknown_tool(name)
            spec = tool.spec

            # 2. Validate (Req 5.5, Req 5.6). No tool code has run yet.
            arguments_error = _call_field(call, "arguments_error")
            if arguments_error:
                return self._bad_arguments_payload(spec, arguments_error)

            raw_arguments = _call_field(call, "arguments", "args", "inputs", "parameters")
            if raw_arguments is None:
                # A call with no arguments at all is the same thing as a call
                # with an empty mapping: any required parameter is then missing
                # and is reported as such, which is a better message than
                # "arguments must be an object".
                raw_arguments = {}
            if not isinstance(raw_arguments, Mapping):
                return self._non_mapping_arguments(spec, raw_arguments)

            invalid = self._validate(spec, raw_arguments)
            if invalid is not None:
                return invalid

            # 3 and 4. Execute under the tool's timeout, catching whatever it
            # raises (Req 5.7, Req 5.8).
            result = self._execute(tool, dict(raw_arguments))

            # 5. Scrub and truncate (Req 5.10, Req 5.12).
            return self._finalise(result)
        except BaseException as exc:  # noqa: BLE001 - Req 5.9 is the reason
            return self._internal_error(exc)

    # Callable, so a test double or the loop can hold ``dispatcher`` and call it
    # directly; ``dispatch`` stays the documented name.
    __call__ = dispatch

    # -- step 1: resolution failed -------------------------------------------

    def _unknown_tool(self, name: Any) -> ToolResult:
        """Req 5.4: the name is unknown, and here are the names that are not."""
        registered = self._registry.names()
        known = ", ".join(registered) if registered else "none are registered"
        return self._finalise(
            ToolResult(
                status="error",
                content=(
                    f"Unknown tool {name!r}: no tool is registered under that exact name. "
                    f"Tool names are matched case-sensitively and exactly. "
                    f"Registered tools: {known}."
                ),
            )
        )

    # -- step 2: validation failed -------------------------------------------

    def _bad_arguments_payload(self, spec: ToolSpec, detail: Any) -> ToolResult:
        """The provider sent arguments that could not be decoded into a mapping.

        The adapter records a parse failure on the call rather than crashing
        (design: Model client), and it arrives here as a schema error: the model
        asked for a real tool, but what it sent cannot be checked against the
        schema, so no tool code runs.
        """
        return self._finalise(
            ToolResult(
                status="error",
                content=(
                    f"Tool {spec.name!r} was not executed: its arguments could not be read "
                    f"as a JSON object ({detail}). Send a JSON object whose keys are the "
                    f"declared parameters: {_declared_list_text(spec)}."
                ),
            )
        )

    def _non_mapping_arguments(self, spec: ToolSpec, received: Any) -> ToolResult:
        """Arguments arrived as something other than a name/value mapping."""
        return self._finalise(
            ToolResult(
                status="error",
                content=(
                    f"Tool {spec.name!r} was not executed: arguments must be a JSON object, "
                    f"but a value of type {_json_type_name(received)!r} was received. "
                    f"Declared parameters: {_declared_list_text(spec)}."
                ),
            )
        )

    def _validate(self, spec: ToolSpec, arguments: Mapping[Any, Any]) -> ToolResult | None:
        """Check ``arguments`` against ``spec``; return an error result or ``None``.

        Three faults, two requirements, one message. Req 5.5 covers a missing
        required parameter and a wrong-typed value, naming each offender with
        the type and required status the schema declares for it. Req 5.6 covers
        an undeclared parameter name, naming each offender and listing the
        declared names. A single call can commit all three, and reporting them
        together lets the model fix the call in one correction instead of
        discovering the faults one round trip at a time.

        A non-string key is counted as undeclared: no declared name is a
        non-string, so the key cannot match one, and reporting it is more useful
        than ignoring it.
        """
        declared = set(spec.declared_parameters)

        undeclared = [key for key in arguments if not (isinstance(key, str) and key in declared)]

        invalid: list[str] = []
        for parameter in spec.required_parameters:
            if parameter not in arguments:
                invalid.append(
                    f"{parameter}: missing, but it is required and must be "
                    f"{_declared_type_text(spec.declared_type(parameter))}"
                )
        for parameter in spec.declared_parameters:
            if parameter not in arguments:
                continue
            value = arguments[parameter]
            declared_type = spec.declared_type(parameter)
            if _matches_declared_type(value, declared_type):
                continue
            status = "required" if spec.is_required(parameter) else "optional"
            invalid.append(
                f"{parameter}: received {_json_type_name(value)}, but it is declared "
                f"{_declared_type_text(declared_type)} and is {status}"
            )

        if not undeclared and not invalid:
            return None

        lines = [f"Tool {spec.name!r} was not executed because its arguments are invalid."]
        if invalid:
            lines.append("Invalid parameters:")
            lines.extend(f"  - {item}" for item in invalid)
        if undeclared:
            names = ", ".join(repr(key) for key in undeclared)
            lines.append(f"Undeclared parameters: {names}.")
        lines.append(f"Declared parameters: {_declared_list_text(spec)}.")
        return self._finalise(ToolResult(status="error", content="\n".join(lines)))

    # -- steps 3 and 4: execution --------------------------------------------

    def _execute(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool under its declared timeout and return a result (Req 5.7, 5.8).

        ``spec.timeout_seconds is None`` runs the tool inline on the calling
        thread with no future and no limit, which is what interactive
        ``save_note`` needs (Req 9.2): the operator is given unlimited time, and
        ``input()`` stays on the main thread.

        Otherwise the call goes to a fresh single-worker executor. Fresh per
        dispatch on purpose: a tool abandoned at its timeout leaves its thread
        running, and a shared executor would make the next call queue behind it.
        Four tool calls per step makes the cost of a thread irrelevant next to
        that.
        """
        spec = tool.spec
        timeout = spec.timeout_seconds

        if timeout is None:
            return self._run_catching(tool, arguments)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{spec.name}")
        try:
            future = executor.submit(self._run_catching, tool, arguments)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                # Req 5.8: abandon the execution. ``cancel`` cannot stop a
                # thread that is already running, and ``shutdown(wait=False)``
                # is what makes this return now rather than when the tool
                # eventually finishes.
                future.cancel()
                return ToolResult(
                    status="error",
                    content=(
                        f"Tool {spec.name!r} exceeded its execution timeout of "
                        f"{self._format_seconds(timeout)} seconds and the execution was "
                        f"abandoned. No result is available for this call."
                    ),
                )
        finally:
            executor.shutdown(wait=False)

    def _run_catching(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        """Call ``execute`` and convert any exception into an error result (Req 5.7).

        This runs on the worker thread when a timeout applies, which is why the
        conversion happens here rather than in ``dispatch``: an exception raised
        on the worker would otherwise be re-raised by ``future.result`` and the
        error result would lose the distinction between "the tool raised" and
        "the dispatcher broke".
        """
        spec = tool.spec
        try:
            result = tool.execute(**arguments)
        except BaseException as exc:  # noqa: BLE001 - Req 5.7 converts, never propagates
            return ToolResult(
                status="error",
                content=(
                    f"Tool {spec.name!r} raised {type(exc).__name__}: {exc}. "
                    f"No result was produced."
                ),
            )
        if not isinstance(result, ToolResult):
            # A tool that returns the wrong type is a developer fault, but it
            # arrives mid-run, so it becomes an observation like any other
            # failure rather than an exception the loop has to handle.
            return ToolResult(
                status="error",
                content=(
                    f"Tool {spec.name!r} returned {type(result).__name__} instead of a "
                    f"ToolResult. No result was produced."
                ),
            )
        return result

    @staticmethod
    def _format_seconds(timeout: float) -> str:
        """Render a timeout the way an operator wrote it: ``10``, not ``10.0``."""
        try:
            return f"{float(timeout):g}"
        except Exception:  # pragma: no cover - ToolSpec already validated it
            return str(timeout)

    # -- step 5: scrub and truncate ------------------------------------------

    def _finalise(self, result: ToolResult) -> ToolResult:
        """Scrub secrets and absolute paths, then truncate (Req 5.10, Req 5.12).

        Scrubbing comes first so the character count the truncation marker
        reports is the count of the text that would actually have been shown,
        and so truncation can never cut a secret in half and leave a prefix of
        it in the content.
        """
        content = result.content if isinstance(result.content, str) else str(result.content)
        content = self.scrub(content, paths=result.status == "error")

        original_length = len(content)
        truncated = original_length > self._max_result_chars
        if truncated:
            content = content[: self._max_result_chars] + TRUNCATION_MARKER.format(
                original=original_length, limit=self._max_result_chars
            )

        approval: Approval | None = result.approval
        return ToolResult(
            status=result.status,
            content=content,
            truncated=truncated or result.truncated,
            # Req 5.12: a truncated result states its pre-truncation count. A
            # tool that already truncated its own content (``read_notes`` has a
            # read budget) keeps the count it reported.
            original_length=original_length if truncated else result.original_length,
            approval=approval,
        )

    def scrub(self, text: str, *, paths: bool = True) -> str:
        """Remove secrets, and optionally rewrite filesystem locations (Req 5.10).

        Secret removal is unconditional: Req 1.11 forbids the key value from
        every output sink, success or failure.

        Path rewriting is applied to error content only, which is the exact
        scope of Req 5.10. The reason is not economy, it is correctness: a
        successful ``read_notes`` or ``search_docs`` result is *document text*,
        and a line of a note that happens to look like a path must read back
        byte for byte (Req 10.4). Rewriting what the operator wrote would be a
        corruption, not a redaction. Error text, by contrast, is written by this
        application, and the only paths in it are the ones a tool was working
        on.

        Written not to raise: it is called from every error branch above,
        including the internal-error branch, so a fault here would be a fault in
        the code that reports faults.
        """
        try:
            scrubbed = redact(text, self._secret)
            return self._relativise_paths(scrubbed) if paths else scrubbed
        except Exception:  # pragma: no cover - defence for the reporting path
            try:
                return redact(text, self._secret)
            except Exception:  # pragma: no cover
                return text

    def _relativise_paths(self, text: str) -> str:
        """Rewrite absolute filesystem paths as project-relative ones (Req 5.10).

        An absolute path in a tool error leaks the directory layout of the
        machine, and on this host that includes the operator's user name. Only
        paths that look absolute are touched -- a drive-letter path, a UNC
        share, or a POSIX path -- and each is rewritten relative to the project
        root. A path on another drive has no relative form, so its file name is
        kept and the rest is dropped, which is still enough for the model to
        understand what failed.
        """
        if not text:
            return text

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            # The regex deliberately runs up to the first delimiter, which can
            # swallow sentence punctuation that belongs to the message, so
            # trailing punctuation is handed back unchanged.
            stripped = raw.rstrip(".,;:!?)\"'")
            trailer = raw[len(stripped) :]
            if not stripped:
                return raw
            try:
                relative = os.path.relpath(stripped, self._project_root)
            except Exception:
                relative = Path(stripped).name or stripped
            return f"{relative}{trailer}"

        return _ABSOLUTE_PATH.sub(replace, text)

    # -- the backstop --------------------------------------------------------

    def _internal_error(self, exc: BaseException) -> ToolResult:
        """The dispatcher itself faulted; still return a result (Req 5.9).

        Reached only by a bug in this module or by something raised where no
        other branch could catch it. Nested guards, because the one promise this
        function exists to keep is that a ``ToolResult`` comes back.
        """
        try:
            return ToolResult(
                status="error",
                content=self.scrub(
                    f"The tool call could not be completed: the dispatcher raised "
                    f"{type(exc).__name__}: {exc}."
                ),
            )
        except BaseException:  # pragma: no cover - the last resort
            return ToolResult(
                status="error",
                content="The tool call could not be completed because of an internal error.",
            )


# A drive-letter path, a UNC share, or a POSIX path with at least two segments.
#
# The segment characters exclude the Windows-illegal set plus whitespace and
# quotes, so a path stops at the end of its token rather than eating the rest of
# the sentence.
#
# The POSIX branch carries two guards. It requires at least two segments,
# because a lone ``/word`` is more often prose than a filesystem location, and
# it refuses a match preceded by ``:`` or ``/``, which is what keeps the path
# component of a URL (``https://host/a/b``) from being rewritten as a relative
# path.
_ABSOLUTE_PATH = re.compile(
    r"""
    (?:
        [A-Za-z]:[\\/](?:[^\\/:*?"<>|\s]+[\\/]?)*                 # C:\dir\file
      | \\\\[^\\/:*?"<>|\s]+[\\/](?:[^\\/:*?"<>|\s]+[\\/]?)*       # \\host\share\file
      | (?<![:/\w.-])/(?:[^/:*?"<>|\s]+/)+[^/:*?"<>|\s]*          # /dir/file
    )
    """,
    re.VERBOSE,
)


__all__ = [
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_TIMEOUT_SECONDS",
    "PROJECT_ROOT",
    "TRUNCATION_MARKER",
    "ToolDispatcher",
]
