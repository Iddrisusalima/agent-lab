"""The eval suite data model and its validation (Req 14.2, 14.9, 14.10, 14.13).

`evals/tasks.json` is the one place the 12 evaluation tasks are declared
(Req 14.1, Req 14.9). This module is the whole of the code side of that file: it
reads it, validates it strictly, and hands the runner frozen
:class:`EvalTask` / :class:`EvalCheck` values. It evaluates nothing --
``checks.py`` owns the per-type evaluation logic -- and it runs nothing.

Four decisions shape it.

**Validation happens once, before any task runs.** Req 14.13 asks that a
malformed suite produce a message naming the affected task identifier and the
specific defect, execute no task, leave an existing report untouched, and exit
non-zero. The only way to promise "no task ran" is to validate the entire file
before the first run starts, so :func:`load_suite` raises on the first defect it
finds and returns a fully validated :class:`EvalSuite` or nothing at all. A
loader that validated lazily, per task, would have already spent money by the
time it reached a defect in task 9.

**:class:`EvalSuiteError` is not a :class:`~agent_lab.config.ConfigError`.**
A broken price table or an absent API key is a configuration fault and exits 6;
a broken eval suite is an eval failure and exits 7
(:data:`~agent_lab.exit_codes.EVAL_FAILURE`, Req 14.13). Since the CLI already
catches `ConfigError` and exits 6, subclassing it here would quietly report the
wrong code, so this error stands on its own and carries its exit code as a
class attribute for the CLI to use.

**This module is the single source of truth for the supported check types.**
Req 14.3 fixes the eight types; :data:`CHECK_SPECS` states each one's required
parameters and their types, and nothing else in the project may re-list them.
``checks.py`` looks its evaluators up by these names and ``tasks.json`` is
validated against these parameter lists, so the data file, the validator and the
evaluators cannot drift apart. Adding a type means adding a spec here and an
evaluator there, and the loader will reject a suite that uses a type with no
spec.

**Bounds come from the modules that own them.** ``max_steps`` is bounded by
:data:`~agent_lab.agent_loop.MAX_STEPS` (Req 14.2) and ``tool_called`` names are
checked against :data:`~agent_lab.tools.factory.REQUIRED_TOOL_NAMES` (Req 5.1),
rather than against numbers and strings copied into this file. A check asserting
that a tool named ``calculater`` was called can never hold, so a typo is a defect
in the suite and is reported as one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from agent_lab import exit_codes
from agent_lab.agent_loop import MAX_STEPS
from agent_lab.tools.factory import REQUIRED_TOOL_NAMES

# --------------------------------------------------------------------------- #
# Where the suite lives
# --------------------------------------------------------------------------- #

# ``src/agent_lab/evals/schema.py`` -> parents[3] is the repository root, the
# same walk `pricing.py` and `prompts.py` use. Req 14.9 fixes the path.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIRNAME = "evals"
TASKS_FILENAME = "tasks.json"
DEFAULT_TASKS_PATH = PROJECT_ROOT / TASKS_DIRNAME / TASKS_FILENAME

# --------------------------------------------------------------------------- #
# The shape of the file
# --------------------------------------------------------------------------- #

VERSION_KEY = "version"
TASKS_KEY = "tasks"
SUITE_KEYS: tuple[str, ...] = (VERSION_KEY, TASKS_KEY)

# The schema revision the design's example carries. A future incompatible shape
# gets version 2 and this constant moves with it, so an old file is reported
# rather than half-read.
SUITE_VERSION = 1

TASK_ID_KEY = "id"
TASK_CATEGORY_KEY = "category"
TASK_PROMPT_KEY = "prompt"
TASK_MAX_STEPS_KEY = "max_steps"
TASK_CHECKS_KEY = "checks"

# Req 14.2: every one of these is required of every task.
TASK_KEYS: tuple[str, ...] = (
    TASK_ID_KEY,
    TASK_CATEGORY_KEY,
    TASK_PROMPT_KEY,
    TASK_MAX_STEPS_KEY,
    TASK_CHECKS_KEY,
)

CHECK_ID_KEY = "id"
CHECK_TYPE_KEY = "type"

# Req 14.2 bounds. The identifier bound is also a practical one: Req 15.3 has the
# runner embed the task identifier in a trace file name, so an identifier is
# restricted to characters that are safe in a path segment on every platform and
# can never be a directory traversal.
ID_MIN_CHARS = 1
ID_MAX_CHARS = 64
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PROMPT_MIN_CHARS = 1
PROMPT_MAX_CHARS = 2000

# An asserted text of zero length would hold against every answer, which is
# never what a check means; the upper bound is generous and only there to catch a
# whole document pasted into a `text` parameter.
TEXT_MIN_CHARS = 1
TEXT_MAX_CHARS = 2000

# A per-task dollar assertion. The loop's own cap is 0.25 per run
# (`agent_loop.COST_CAP`); the ceiling here is the suite ceiling's order of
# magnitude, so an obviously mistyped figure is still rejected.
USD_MIN = 0.0
USD_MAX = 100.0

# --------------------------------------------------------------------------- #
# Categories (design: suite composition, Req 14.4-14.8, 14.12)
# --------------------------------------------------------------------------- #

CATEGORY_SINGLE_TOOL = "single_tool"
CATEGORY_MULTI_TOOL = "multi_tool"
CATEGORY_ABSENT_ANSWER = "absent_answer"
CATEGORY_TOOL_ERROR = "tool_error"
CATEGORY_NO_TOOL = "no_tool"
CATEGORY_INJECTION = "injection"
CATEGORY_COST_GUARD = "cost_guard"

# Req 14.1 and the design's composition table: exactly 12 tasks in this split.
# `tasks.json` (task 11.5) is written against these counts and the resource lint
# asserts them, which is why the numbers live here rather than in either.
CATEGORY_COMPOSITION: Mapping[str, int] = MappingProxyType(
    {
        CATEGORY_SINGLE_TOOL: 3,
        CATEGORY_MULTI_TOOL: 4,
        CATEGORY_ABSENT_ANSWER: 1,
        CATEGORY_TOOL_ERROR: 1,
        CATEGORY_NO_TOOL: 1,
        CATEGORY_INJECTION: 1,
        CATEGORY_COST_GUARD: 1,
    }
)

CATEGORIES: tuple[str, ...] = tuple(CATEGORY_COMPOSITION)

# Req 14.1. Equal to the composition by construction, so the two can never
# disagree.
SUITE_SIZE = sum(CATEGORY_COMPOSITION.values())

# --------------------------------------------------------------------------- #
# The eight check types (Req 14.3) -- the single source of truth
# --------------------------------------------------------------------------- #

CHECK_ANSWER_CONTAINS = "answer_contains"
CHECK_ANSWER_NOT_CONTAINS = "answer_not_contains"
CHECK_TOOL_CALLED = "tool_called"
CHECK_NO_TOOL_CALLED = "no_tool_called"
CHECK_ENDED_WITH_FINAL_ANSWER = "ended_with_final_answer"
CHECK_NOTES_NOT_CONTAINS = "notes_not_contains"
CHECK_STEPS_AT_MOST = "steps_at_most"
CHECK_COST_AT_MOST = "cost_at_most"

# The parameter kinds the validator knows how to check. Kept as plain strings so
# a spec reads like the design's table rather than like a type system.
PARAM_TEXT = "text"  # a non-empty string, TEXT_MIN_CHARS..TEXT_MAX_CHARS
PARAM_TOOL_NAME = "tool_name"  # one of the registered tool names
PARAM_STEP_COUNT = "step_count"  # an int in 1..MAX_STEPS
PARAM_USD = "usd"  # a finite number in USD_MIN..USD_MAX


@dataclass(frozen=True)
class CheckSpec:
    """What one supported check type is called and what it requires.

    ``required_params`` maps a parameter name to its kind, drawn from the
    ``PARAM_*`` constants above. ``evaluated_against`` records, in one phrase,
    the :class:`~agent_lab.agent_loop.RunResult` field the evaluator reads; it is
    documentation for the reader and the error messages, not behaviour.
    """

    type: str
    required_params: Mapping[str, str]
    evaluated_against: str

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(self.required_params)


def _spec(type_: str, params: Mapping[str, str], against: str) -> CheckSpec:
    return CheckSpec(
        type=type_,
        required_params=MappingProxyType(dict(params)),
        evaluated_against=against,
    )


# Req 14.3's eight types, in the design's table order. ``checks.py`` (task 11.2)
# provides exactly one evaluator per key; ``tasks.json`` (task 11.5) may use
# exactly these types and must supply exactly these parameters.
CHECK_SPECS: Mapping[str, CheckSpec] = MappingProxyType(
    {
        CHECK_ANSWER_CONTAINS: _spec(
            CHECK_ANSWER_CONTAINS, {"text": PARAM_TEXT}, "normalised final_answer"
        ),
        CHECK_ANSWER_NOT_CONTAINS: _spec(
            CHECK_ANSWER_NOT_CONTAINS, {"text": PARAM_TEXT}, "normalised final_answer"
        ),
        CHECK_TOOL_CALLED: _spec(
            CHECK_TOOL_CALLED, {"tool": PARAM_TOOL_NAME}, "tools_called"
        ),
        CHECK_NO_TOOL_CALLED: _spec(
            CHECK_NO_TOOL_CALLED, {}, "tools_called is empty"
        ),
        CHECK_ENDED_WITH_FINAL_ANSWER: _spec(
            CHECK_ENDED_WITH_FINAL_ANSWER, {}, "outcome == FINAL_ANSWER"
        ),
        CHECK_NOTES_NOT_CONTAINS: _spec(
            CHECK_NOTES_NOT_CONTAINS,
            {"text": PARAM_TEXT},
            "normalised notes_text_at_end",
        ),
        CHECK_STEPS_AT_MOST: _spec(
            CHECK_STEPS_AT_MOST, {"count": PARAM_STEP_COUNT}, "steps"
        ),
        CHECK_COST_AT_MOST: _spec(CHECK_COST_AT_MOST, {"usd": PARAM_USD}, "cost_usd"),
    }
)

CHECK_TYPES: tuple[str, ...] = tuple(CHECK_SPECS)


# --------------------------------------------------------------------------- #
# The error
# --------------------------------------------------------------------------- #


class EvalSuiteError(Exception):
    """The eval suite is absent, unparseable, or defective (Req 14.13).

    Deliberately *not* a :class:`~agent_lab.config.ConfigError`: the CLI maps
    that to exit code 6, while Req 14.13 wants an eval failure, which is code 7.
    :attr:`exit_code` carries that code so the caller does not have to decide.

    Every message names the file and the specific defect, and names the affected
    task identifier whenever the defect is inside a task -- or the task's
    position in the list when the identifier itself is what is missing.
    """

    exit_code = exit_codes.EVAL_FAILURE


# --------------------------------------------------------------------------- #
# The data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalCheck:
    """One assertion over a finished run (Req 14.2, Req 14.3).

    ``params`` holds exactly the parameters the declared type requires, with
    ``id`` and ``type`` removed, which is the ``params`` argument that each
    ``checks.py`` evaluator takes. It is a read-only mapping, so an evaluator
    cannot mutate the suite it was handed.
    """

    id: str
    type: str
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def spec(self) -> CheckSpec:
        """The :class:`CheckSpec` for this check's type."""
        return CHECK_SPECS[self.type]

    def describe(self) -> str:
        """A one-line rendering for a report row, e.g. ``value (answer_contains)``."""
        return f"{self.id} ({self.type})"


@dataclass(frozen=True)
class EvalTask:
    """One evaluation task: a prompt, a category, a step budget, its checks."""

    id: str
    category: str
    prompt: str
    max_steps: int
    checks: tuple[EvalCheck, ...]

    def check_ids(self) -> tuple[str, ...]:
        return tuple(check.id for check in self.checks)


@dataclass(frozen=True)
class EvalSuite:
    """The validated contents of `evals/tasks.json` (Req 14.9).

    Tasks are held in declaration order, which is the order Req 15.1 has the
    runner execute them in.
    """

    version: int
    tasks: tuple[EvalTask, ...]
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[EvalTask]:
        return iter(self.tasks)

    def ids(self) -> tuple[str, ...]:
        """Every task identifier, in declaration order."""
        return tuple(task.id for task in self.tasks)

    def get(self, task_id: str) -> EvalTask | None:
        """The task with this identifier, or ``None``.

        Matching is case-sensitive and exact, like every other identifier in the
        project. Req 15.14 turns the ``None`` into "unknown identifier, nothing
        ran, non-zero exit".
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def counts_by_category(self) -> Mapping[str, int]:
        """How many tasks each category holds, including categories with none."""
        counts = {category: 0 for category in CATEGORIES}
        for task in self.tasks:
            counts[task.category] += 1
        return MappingProxyType(counts)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _fault(path: Path, detail: str) -> EvalSuiteError:
    """Build the one error shape: the file, then what is wrong with it."""
    return EvalSuiteError(f"The eval suite at {path} is not usable: {detail}")


def _task_fault(path: Path, where: str, detail: str) -> EvalSuiteError:
    """A defect inside one task, named by identifier or by position (Req 14.13)."""
    return _fault(path, f"{where}: {detail}")


def _quote_list(values: Sequence[str]) -> str:
    return ", ".join(repr(value) for value in values)


def _expected_keys(keys: Sequence[str]) -> str:
    return _quote_list(sorted(keys))


def _reject_unknown_keys(
    path: Path,
    where: str | None,
    mapping: Mapping[str, Any],
    allowed: Sequence[str],
    what: str,
) -> None:
    """Reject any key that is neither declared nor an underscore comment.

    Underscore-prefixed keys are comments at every level of the project's JSON
    resources, as in `pricing/prices.json`. Any other unexpected key is far more
    likely a misspelling of a declared one, and a misspelled ``prompt`` that
    loaded silently would mean a task running the wrong text.
    """
    unexpected = sorted(
        key for key in mapping if key not in allowed and not key.startswith("_")
    )
    if not unexpected:
        return
    detail = (
        f"{what} carries unrecognised "
        f"{'keys' if len(unexpected) > 1 else 'key'} {_quote_list(unexpected)}; "
        f"the declared {'keys are' if len(allowed) > 1 else 'key is'} "
        f"{_expected_keys(allowed)}."
    )
    if where:
        raise _task_fault(path, where, detail)
    raise _fault(path, detail)


def _is_int(value: Any) -> bool:
    # `isinstance(True, int)` is True in Python, so booleans are excluded by hand
    # wherever a number is wanted -- the same guard the calculator applies.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_identifier(path: Path, where: str, label: str, value: Any) -> str:
    """Validate a task or check identifier (Req 14.2)."""
    if not isinstance(value, str):
        raise _task_fault(
            path,
            where,
            f"its {label} is {type(value).__name__}, not a string ({value!r}).",
        )
    if len(value) < ID_MIN_CHARS or len(value) > ID_MAX_CHARS:
        raise _task_fault(
            path,
            where,
            f"its {label} is {len(value)} characters long ({value!r}); an "
            f"identifier must be {ID_MIN_CHARS} to {ID_MAX_CHARS} characters.",
        )
    if not _ID_PATTERN.match(value):
        raise _task_fault(
            path,
            where,
            f"its {label} {value!r} is not a usable identifier; use letters, "
            "digits, '.', '_' and '-' only, starting with a letter or a digit, "
            "because the runner puts a task identifier in a trace file name.",
        )
    return value


def _check_text_param(
    path: Path, where: str, check_id: str, name: str, value: Any
) -> str:
    if not isinstance(value, str):
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of type "
            f"{type(value).__name__}, not a string ({value!r}).",
        )
    if len(value) < TEXT_MIN_CHARS or len(value) > TEXT_MAX_CHARS:
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of {len(value)} characters; an "
            f"asserted text must be {TEXT_MIN_CHARS} to {TEXT_MAX_CHARS} "
            "characters.",
        )
    if not value.strip():
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a whitespace-only {name!r} ({value!r}), "
            "which would hold against any text.",
        )
    return value


def _check_tool_param(
    path: Path, where: str, check_id: str, name: str, value: Any
) -> str:
    if not isinstance(value, str):
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of type "
            f"{type(value).__name__}, not a string ({value!r}).",
        )
    if value not in REQUIRED_TOOL_NAMES:
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} names the tool {value!r}, which is not "
            f"registered; the registered tools are "
            f"{_quote_list(REQUIRED_TOOL_NAMES)}.",
        )
    return value


def _check_step_count_param(
    path: Path, where: str, check_id: str, name: str, value: Any
) -> int:
    if not _is_int(value):
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of {value!r}, which is not an "
            "integer.",
        )
    if value < 1 or value > MAX_STEPS:
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of {value}; a step count must "
            f"be between 1 and MAX_STEPS ({MAX_STEPS}), the bound every run is "
            "held to.",
        )
    return value


def _check_usd_param(
    path: Path, where: str, check_id: str, name: str, value: Any
) -> float:
    if not _is_number(value):
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of {value!r}, which is not a "
            "number.",
        )
    amount = float(value)
    if not math.isfinite(amount):
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a non-finite {name!r} ({value!r}).",
        )
    if amount <= USD_MIN or amount > USD_MAX:
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} has a {name!r} of {value!r}; a dollar bound "
            f"must be greater than {USD_MIN:.2f} and at most {USD_MAX:.2f}.",
        )
    return amount


_PARAM_CHECKERS = {
    PARAM_TEXT: _check_text_param,
    PARAM_TOOL_NAME: _check_tool_param,
    PARAM_STEP_COUNT: _check_step_count_param,
    PARAM_USD: _check_usd_param,
}


def _parse_check(path: Path, where: str, index: int, raw: Any) -> EvalCheck:
    """Validate one check and return it (Req 14.3, Req 14.10, Req 14.13)."""
    position = f"check at position {index + 1}"
    if not isinstance(raw, dict):
        raise _task_fault(
            path,
            where,
            f"its {position} is {type(raw).__name__}, not an object.",
        )
    if CHECK_ID_KEY not in raw:
        raise _task_fault(
            path, where, f"its {position} has no {CHECK_ID_KEY!r}."
        )
    check_id = _check_identifier(
        path, where, f"{position} identifier", raw[CHECK_ID_KEY]
    )
    if CHECK_TYPE_KEY not in raw:
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} declares no {CHECK_TYPE_KEY!r}; the supported "
            f"types are {_quote_list(CHECK_TYPES)}.",
        )
    declared_type = raw[CHECK_TYPE_KEY]
    if not isinstance(declared_type, str) or declared_type not in CHECK_SPECS:
        raise _task_fault(
            path,
            where,
            f"check {check_id!r} declares the unsupported check type "
            f"{declared_type!r}; the supported types are "
            f"{_quote_list(CHECK_TYPES)}.",
        )
    spec = CHECK_SPECS[declared_type]

    allowed = (CHECK_ID_KEY, CHECK_TYPE_KEY, *spec.param_names)
    _reject_unknown_keys(path, where, raw, allowed, f"check {check_id!r}")

    params: dict[str, Any] = {}
    for name, kind in spec.required_params.items():
        if name not in raw:
            raise _task_fault(
                path,
                where,
                f"check {check_id!r} of type {declared_type!r} omits its "
                f"required parameter {name!r}; that type requires "
                f"{_expected_keys(spec.param_names)}.",
            )
        params[name] = _PARAM_CHECKERS[kind](path, where, check_id, name, raw[name])

    return EvalCheck(
        id=check_id, type=declared_type, params=MappingProxyType(params)
    )


def _parse_task(path: Path, index: int, raw: Any) -> EvalTask:
    """Validate one task and return it (Req 14.2, Req 14.13)."""
    # Until the identifier is known, a defect can only be located by position --
    # which is still "naming the affected task" as precisely as the file allows.
    where = f"the task at position {index + 1}"
    if not isinstance(raw, dict):
        raise _fault(
            path,
            f"{where} is {type(raw).__name__}, not an object.",
        )

    missing = [key for key in TASK_KEYS if key not in raw]
    if TASK_ID_KEY in missing:
        raise _task_fault(
            path,
            where,
            f"it has no {TASK_ID_KEY!r}. Every task needs "
            f"{_expected_keys(TASK_KEYS)}.",
        )
    task_id = _check_identifier(path, where, TASK_ID_KEY, raw[TASK_ID_KEY])
    where = f"task {task_id!r}"
    if missing:
        raise _task_fault(
            path,
            where,
            f"it omits {_quote_list([k for k in TASK_KEYS if k in missing])}. "
            f"Every task needs {_expected_keys(TASK_KEYS)}.",
        )

    _reject_unknown_keys(path, where, raw, TASK_KEYS, "it")

    category = raw[TASK_CATEGORY_KEY]
    if not isinstance(category, str) or category not in CATEGORY_COMPOSITION:
        raise _task_fault(
            path,
            where,
            f"its {TASK_CATEGORY_KEY!r} is {category!r}, which is not one of "
            f"{_quote_list(CATEGORIES)}.",
        )

    prompt = raw[TASK_PROMPT_KEY]
    if not isinstance(prompt, str):
        raise _task_fault(
            path,
            where,
            f"its {TASK_PROMPT_KEY!r} is {type(prompt).__name__}, not a string.",
        )
    if len(prompt) < PROMPT_MIN_CHARS or len(prompt) > PROMPT_MAX_CHARS:
        raise _task_fault(
            path,
            where,
            f"its {TASK_PROMPT_KEY!r} is {len(prompt)} characters long; a "
            f"prompt must be {PROMPT_MIN_CHARS} to {PROMPT_MAX_CHARS} "
            "characters.",
        )
    if not prompt.strip():
        # The loop rejects a whitespace-only task as an invalid task (Req 4.12),
        # so a suite holding one would spend a run to learn what the file could
        # have said at load time.
        raise _task_fault(
            path,
            where,
            f"its {TASK_PROMPT_KEY!r} is whitespace only; the agent would "
            "reject it as an invalid task.",
        )

    max_steps = raw[TASK_MAX_STEPS_KEY]
    if not _is_int(max_steps):
        raise _task_fault(
            path,
            where,
            f"its {TASK_MAX_STEPS_KEY!r} is {max_steps!r}, which is not an "
            "integer.",
        )
    if max_steps < 1 or max_steps > MAX_STEPS:
        raise _task_fault(
            path,
            where,
            f"its {TASK_MAX_STEPS_KEY!r} is {max_steps}; it must be between 1 "
            f"and MAX_STEPS ({MAX_STEPS}), the bound every run is held to.",
        )

    raw_checks = raw[TASK_CHECKS_KEY]
    if not isinstance(raw_checks, list):
        raise _task_fault(
            path,
            where,
            f"its {TASK_CHECKS_KEY!r} is {type(raw_checks).__name__}, not a "
            "list.",
        )
    if not raw_checks:
        raise _task_fault(
            path,
            where,
            f"its {TASK_CHECKS_KEY!r} list is empty; every task needs at least "
            "one check, or its run asserts nothing.",
        )

    checks: list[EvalCheck] = []
    seen: dict[str, int] = {}
    for check_index, raw_check in enumerate(raw_checks):
        check = _parse_check(path, where, check_index, raw_check)
        if check.id in seen:
            raise _task_fault(
                path,
                where,
                f"it declares two checks with the identifier {check.id!r} "
                f"(positions {seen[check.id] + 1} and {check_index + 1}); a "
                "report row names a failing check by identifier, so they must "
                "be distinct within a task.",
            )
        seen[check.id] = check_index
        checks.append(check)

    return EvalTask(
        id=task_id,
        category=category,
        prompt=prompt,
        max_steps=max_steps,
        checks=tuple(checks),
    )


def parse_suite(document: Any, *, path: Path) -> EvalSuite:
    """Validate an already-parsed suite document (Req 14.2, 14.10, 14.13).

    Split out from :func:`load_suite` so the rules can be exercised against a
    literal dict, and so the property test for Property 24 can corrupt an
    in-memory suite without a file. ``path`` only names the origin in messages.

    :raises EvalSuiteError: on any structural defect, an unsupported check type,
        a missing required check parameter, or any value out of its documented
        range. Nothing partial is returned: either the whole suite validates or
        no suite is produced, which is what lets the runner promise that no task
        executed (Req 14.13).
    """
    if not isinstance(document, dict):
        raise _fault(
            path, f"the top level is {type(document).__name__}, not an object."
        )
    _reject_unknown_keys(path, None, document, SUITE_KEYS, "the top level")

    version = document.get(VERSION_KEY, SUITE_VERSION)
    if not _is_int(version) or version < 1:
        raise _fault(
            path,
            f"its {VERSION_KEY!r} is {version!r}, which is not a positive "
            "integer.",
        )
    if version != SUITE_VERSION:
        raise _fault(
            path,
            f"it declares schema version {version}, and this build reads "
            f"version {SUITE_VERSION}.",
        )

    if TASKS_KEY not in document:
        raise _fault(path, f"the top-level {TASKS_KEY!r} list is missing.")
    raw_tasks = document[TASKS_KEY]
    if not isinstance(raw_tasks, list):
        raise _fault(
            path,
            f"{TASKS_KEY!r} is {type(raw_tasks).__name__}, not a list of task "
            "objects.",
        )
    if not raw_tasks:
        raise _fault(path, f"{TASKS_KEY!r} holds no tasks.")

    tasks: list[EvalTask] = []
    seen: dict[str, int] = {}
    for index, raw_task in enumerate(raw_tasks):
        task = _parse_task(path, index, raw_task)
        if task.id in seen:
            raise _task_fault(
                path,
                f"task {task.id!r}",
                f"its identifier is declared twice (positions "
                f"{seen[task.id] + 1} and {index + 1}); Req 14.2 requires task "
                "identifiers to be unique within the suite.",
            )
        seen[task.id] = index
        tasks.append(task)

    return EvalSuite(version=version, tasks=tuple(tasks), source_path=path)


def validate_composition(suite: EvalSuite) -> None:
    """Assert the published size and category split (Req 14.1, Req 14.4-14.8, 14.12).

    Kept apart from :func:`parse_suite` on purpose. Shape validation is what
    Req 14.13 makes the runner's precondition, and a test or an experiment may
    legitimately load a two-task suite; the exact 12-task composition is a
    property of the *published* suite, asserted by the resource lint and by the
    runner when it loads the shipped file.

    :raises EvalSuiteError: when the suite does not hold exactly
        :data:`SUITE_SIZE` tasks in the :data:`CATEGORY_COMPOSITION` split.
    """
    path = suite.source_path or Path(DEFAULT_TASKS_PATH)
    if len(suite) != SUITE_SIZE:
        raise _fault(
            path,
            f"it holds {len(suite)} tasks; the published suite holds exactly "
            f"{SUITE_SIZE}.",
        )
    counts = suite.counts_by_category()
    wrong = [
        f"{category!r} has {counts[category]} (expected {expected})"
        for category, expected in CATEGORY_COMPOSITION.items()
        if counts[category] != expected
    ]
    if wrong:
        raise _fault(
            path,
            "its category split is wrong: " + "; ".join(wrong) + ".",
        )


def load_suite(
    path: Path | str | None = None,
    *,
    require_composition: bool = False,
) -> EvalSuite:
    """Read and validate the eval suite from disk (Req 14.9, Req 14.13).

    :param path: the suite to read; defaults to :data:`DEFAULT_TASKS_PATH`.
    :param require_composition: also assert the published 12-task composition
        via :func:`validate_composition`.
    :raises EvalSuiteError: if the file is absent, unreadable, not UTF-8, not
        well-formed JSON, or fails any check in :func:`parse_suite`. The
        exception carries ``exit_code == exit_codes.EVAL_FAILURE``, and no task
        has run and no report has been touched when it is raised.
    """
    resolved = Path(path) if path is not None else DEFAULT_TASKS_PATH
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _fault(
            resolved,
            "the file does not exist. Agent Lab declares its evaluation tasks "
            f"in {TASKS_DIRNAME}/{TASKS_FILENAME}; restore it or pass a "
            "different path.",
        ) from None
    except IsADirectoryError:
        raise _fault(resolved, "it is a directory, not a file.") from None
    except UnicodeDecodeError:
        raise _fault(resolved, "it is not valid UTF-8 text.") from None
    except OSError as exc:
        raise _fault(resolved, f"reading it failed ({exc.strerror or exc}).") from None

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _fault(
            resolved,
            f"it is not well-formed JSON (line {exc.lineno}, column "
            f"{exc.colno}: {exc.msg}).",
        ) from None

    suite = parse_suite(document, path=resolved)
    if require_composition:
        validate_composition(suite)
    return suite


__all__ = [
    "CATEGORIES",
    "CATEGORY_ABSENT_ANSWER",
    "CATEGORY_COMPOSITION",
    "CATEGORY_COST_GUARD",
    "CATEGORY_INJECTION",
    "CATEGORY_MULTI_TOOL",
    "CATEGORY_NO_TOOL",
    "CATEGORY_SINGLE_TOOL",
    "CATEGORY_TOOL_ERROR",
    "CHECK_ANSWER_CONTAINS",
    "CHECK_ANSWER_NOT_CONTAINS",
    "CHECK_COST_AT_MOST",
    "CHECK_ENDED_WITH_FINAL_ANSWER",
    "CHECK_ID_KEY",
    "CHECK_NOTES_NOT_CONTAINS",
    "CHECK_NO_TOOL_CALLED",
    "CHECK_SPECS",
    "CHECK_STEPS_AT_MOST",
    "CHECK_TOOL_CALLED",
    "CHECK_TYPES",
    "CHECK_TYPE_KEY",
    "DEFAULT_TASKS_PATH",
    "ID_MAX_CHARS",
    "ID_MIN_CHARS",
    "PARAM_STEP_COUNT",
    "PARAM_TEXT",
    "PARAM_TOOL_NAME",
    "PARAM_USD",
    "PROJECT_ROOT",
    "PROMPT_MAX_CHARS",
    "PROMPT_MIN_CHARS",
    "SUITE_SIZE",
    "SUITE_VERSION",
    "TASKS_FILENAME",
    "TASKS_KEY",
    "TASK_CATEGORY_KEY",
    "TASK_CHECKS_KEY",
    "TASK_ID_KEY",
    "TASK_KEYS",
    "TASK_MAX_STEPS_KEY",
    "TASK_PROMPT_KEY",
    "TEXT_MAX_CHARS",
    "TEXT_MIN_CHARS",
    "USD_MAX",
    "USD_MIN",
    "VERSION_KEY",
    "CheckSpec",
    "EvalCheck",
    "EvalSuite",
    "EvalSuiteError",
    "EvalTask",
    "load_suite",
    "parse_suite",
    "validate_composition",
]
