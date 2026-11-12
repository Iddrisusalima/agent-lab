"""Execute the eval suite, print the results table, write the report (Req 15).

The runner is the project's integration tier: it performs real runs against a
real provider, so every decision here is about spending money predictably and
about leaving the operator's own files alone.

**It does not import the CLI.** Req 1.1 requires every source file -- the eval
runner named among them -- to be importable without importing the CLI entry
point, so nothing here comes from ``cli.py``. What it reuses instead are the
same wiring functions the CLI calls: :func:`~agent_lab.config.load_config`,
:func:`~agent_lab.prompts.load_system_prompt`,
:func:`~agent_lab.pricing.price_table`,
:func:`~agent_lab.retrieval.chroma_index.select_retriever`,
:func:`~agent_lab.tools.factory.build_tool_registry`,
:func:`~agent_lab.providers.build_model_client` and
:func:`~agent_lab.render.build_renderer`. The ``eval`` subcommand (task 11.9) is
a thin handler in ``cli.py`` that calls :func:`run_eval`; the dependency points
that way and never back.

**Nothing of the operator's is written to.** A run's notes path and trace path
are configuration, and the suite redirects both: every task writes notes to one
evaluation notes file and its trace to ``<trace dir>/<task id>.jsonl``
(Req 15.3). Those paths are compared against the configured ones after
``Path.resolve()`` and after ``os.path.normcase``, so ``./notes.md``, an
absolute spelling of it, and a differing case on Windows all collide as they
should; a collision executes no task and raises :class:`EvalRunnerError`
(Req 15.4). The evaluation notes file is truncated to empty before each task, so
``read_notes`` during task 9 cannot see what task 3 saved (Req 15.5).

**No operator input is read.** Each task's configuration carries
``interactive=False`` and the gate is an
:class:`~agent_lab.confirmation.AutoApproveGate`, whose constructor refuses to
exist unless non-interactive mode was actually selected (Req 15.2). A suite run
therefore cannot block on a terminal prompt half way through.

**One failing task is not a failing suite.** A check that does not hold fails
its task and the suite goes on; an exception raised anywhere inside a task's
wiring or run fails that task with the message truncated to 500 characters and
the suite goes on (Req 15.6, Req 15.7). What *does* stop the suite is money:
before each task the cumulative cost is compared against the suite ceiling
(``EVAL_COST_CEILING``, default 5.00), and once it is reached every remaining
task is recorded as not run and the exit code is non-zero (Req 15.8).

**The model client is injectable.** :func:`run_suite` takes ``model_client`` or
``model_client_factory``, so the whole suite can be driven offline by the test
suite's ``FakeModelClient`` with no provider adapter built and no network call
issued (Req 16.8). The default factory builds the configured provider's client,
one per task, exactly as the CLI builds one per run.

**The report writer is the third redaction sink.** Req 1.11 covers terminal
output, trace lines *and* the eval report, so every byte this module writes --
to a stream or to ``evals/results.md`` -- goes through
:func:`~agent_lab.redaction.redact` first.

**The report keeps its hand-written half.** Req 15.11 is a product of one run
and Req 15.12 is a record of an improvement cycle, so the writer regenerates
everything above the ``## Before and after`` heading and copies that heading and
everything after it through unchanged (task 11.10 fills it in by hand).

**The second model is optional and invisible when unconfigured.** Req 18.5 lets
an operator name one comparison model identifier in ``COMPARE_MODEL_ID``, and
:attr:`~agent_lab.config.Config.comparison_model_ids` is the whole of that
configuration surface: a 1-tuple unconfigured, a 2-tuple otherwise, never more.
With one identifier the code path is the one that was always here --
:func:`run_suite` once, :func:`render_report` once, the same tables under the
same headings -- because Req 18.6 and Req 18.7 require a report a reader cannot
tell the feature from. With two, :func:`run_comparison` runs the suite once per
identifier and composes the per-model reports: the model dimension lives in
:class:`ComparisonReport`, not inside :class:`SuiteReport`, so nothing a
single-model run produces changes shape. Two details are worth stating:

* **Trace paths.** Req 15.3 names a trace file after its task, and two models
  running the same twelve tasks would write the same twelve paths -- which
  :func:`verify_paths` is right to refuse (Req 15.4). A comparison run therefore
  gives each model its own subdirectory of the trace directory, named after a
  path-safe form of its model identifier: ``<trace dir>/<model>/<task
  id>.jsonl``. With one model the path is unchanged: ``<trace dir>/<task
  id>.jsonl``.
* **The cost ceiling.** ``EVAL_COST_CEILING`` bounds the invocation, not the
  model (Req 15.8), so each model's suite is started with what the previous one
  already spent (``initial_spend``). Two models cannot spend twice the ceiling;
  the second model's tasks are recorded not run once the ceiling is reached.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable, Mapping, Sequence

from agent_lab import exit_codes
from agent_lab.agent_loop import AgentLoop, RunOutcome, RunResult
from agent_lab.config import Config, load_config
from agent_lab.confirmation import AutoApproveGate
from agent_lab.cost import CostTracker
from agent_lab.evals import checks as check_module
from agent_lab.evals.schema import (
    PROJECT_ROOT,
    EvalSuite,
    EvalSuiteError,
    EvalTask,
    load_suite,
)
from agent_lab.model_client import ModelClient
from agent_lab.pricing import PriceTable, price_table
from agent_lab.prompts import load_system_prompt
from agent_lab.providers import build_model_client
from agent_lab.redaction import redact
from agent_lab.render import build_renderer
from agent_lab.retrieval.base import Retriever
from agent_lab.retrieval.chroma_index import select_retriever
from agent_lab.tools.dispatcher import ToolDispatcher
from agent_lab.tools.factory import build_tool_registry
from agent_lab.trace import JsonlTraceLogger

# --------------------------------------------------------------------------- #
# Where the suite's own files live
# --------------------------------------------------------------------------- #

#: The directory holding ``tasks.json`` and ``results.md`` (Req 14.9, Req 15.11).
EVALS_DIR = PROJECT_ROOT / "evals"

#: The report. Req 15.11 names the file; Req 15.12 keeps part of it by hand.
DEFAULT_REPORT_PATH = EVALS_DIR / "results.md"

#: Where a suite run puts the files it writes. Named ``traces`` because
#: ``.gitignore`` already excludes any directory of that name at any depth, so
#: the notes file and the per-task traces of an eval run can never be committed
#: (Req 1.4) and a reviewer's clone stays clean (Req 16.9).
DEFAULT_EVAL_RUN_DIR = EVALS_DIR / "traces"

#: The evaluation notes file: one file, truncated before each task (Req 15.5).
DEFAULT_EVAL_NOTES_PATH = DEFAULT_EVAL_RUN_DIR / "eval_notes.md"

#: Per-task trace file names are ``<task id><TRACE_SUFFIX>`` (Req 15.3). The
#: identifier is safe in a path segment because ``schema.py`` restricts it to
#: letters, digits, ``.``, ``_`` and ``-`` for exactly this reason.
TRACE_SUFFIX = ".jsonl"

# Req 15.7: an error that fails a task is recorded at most this long.
ERROR_MESSAGE_MAX_CHARS = 500

# Req 15.8: stop one tolerance short of the ceiling rather than one float
# wobble past it. The same tolerance the cost checks compare with.
COST_TOLERANCE_USD = check_module.COST_TOLERANCE_USD

#: The heading whose section the writer never regenerates (Req 15.12).
PRESERVED_HEADING = "## Before and after"

#: The stub written under that heading the first time a report is generated, so
#: task 11.10 has the shape to fill in rather than a blank page.
PRESERVED_STUB = """## Before and after

Hand-maintained (Req 15.12). The runner never rewrites anything from this
heading down; the sections above it are regenerated on every run.

- **Before:** _pass rate_ % -- model `_model id_`, measured _YYYY-MM-DD_
- **After:** _pass rate_ % -- model `_model id_`, measured _YYYY-MM-DD_

### Changes made between the two measurements

- _each change to a tool description, to the system prompt, or to code_
"""


# --------------------------------------------------------------------------- #
# The error
# --------------------------------------------------------------------------- #


class EvalRunnerError(Exception):
    """The suite could not be run at all: no task executed, no report written.

    Raised for an unknown task identifier (Req 15.14) and for an evaluation path
    that collides with one of the operator's configured paths (Req 15.4). Both
    are refusals *before* the first task, which is why they are exceptions
    rather than rows in a report: there is nothing to report yet, and Req 15.4
    and Req 14.13 both require an existing report to be left as it was.

    :attr:`exit_code` carries :data:`~agent_lab.exit_codes.EVAL_FAILURE`, the
    same code :class:`~agent_lab.evals.schema.EvalSuiteError` carries, so a
    caller reports either without deciding anything.
    """

    exit_code = exit_codes.EVAL_FAILURE


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class EvalOutcome(str, Enum):
    """The three outcomes Req 15.9 permits in the table's outcome column."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not run"


@dataclass(frozen=True)
class CheckResult:
    """One evaluated check: which one, whether it held, and what it compared."""

    id: str
    type: str
    passed: bool
    reason: str

    def describe(self) -> str:
        """``id (type): reason`` -- the line a failing-check listing prints."""
        return f"{self.id} ({self.type}): {self.reason}"


@dataclass(frozen=True)
class TaskResult:
    """What one row of the results table reports (Req 15.9).

    ``steps`` and ``cost_usd`` are zero for a task that never started, which is
    what makes a not-run row distinguishable from a free one: ``outcome`` says
    which it is, and the aggregates of Req 15.10 average over executed tasks
    only.
    """

    task_id: str
    category: str
    outcome: EvalOutcome
    steps: int = 0
    cost_usd: float = 0.0
    run_outcome: RunOutcome | None = None
    checks: tuple[CheckResult, ...] = ()
    error: str | None = None
    trace_path: str = ""
    final_answer: str | None = None

    @property
    def executed(self) -> bool:
        """Whether a run was actually performed for this task."""
        return self.outcome is not EvalOutcome.NOT_RUN

    @property
    def passed(self) -> bool:
        return self.outcome is EvalOutcome.PASSED

    @property
    def failing_checks(self) -> tuple[CheckResult, ...]:
        """Every check of this task that did not hold (Req 15.6)."""
        return tuple(check for check in self.checks if not check.passed)

    @property
    def detail(self) -> str:
        """The last column of the table: failing check ids, or the error.

        Req 15.9 asks for "the identifier of each failing Eval_Check or the
        truncated error message". A raised error has no check identifiers to
        report, so the message stands in for them; a task that passed has
        neither and prints a dash so the column is never empty.
        """
        if self.error is not None:
            return self.error
        failing = self.failing_checks
        if failing:
            return ", ".join(check.id for check in failing)
        return "-"


@dataclass(frozen=True)
class SuiteReport:
    """Everything one suite execution produced (Req 15.9, 15.10, 15.11, 15.15).

    ``suite_size`` is the number of tasks in the *suite*, not in the selection,
    because Req 15.10 divides the passed count by the suite total. When one task
    was selected, :attr:`selected_pass_rate` reports the selection's own rate as
    well, so a single-task run of Req 15.13 prints a number that means something.
    """

    model_id: str
    started_at: datetime
    results: tuple[TaskResult, ...]
    suite_size: int
    cost_ceiling: float
    ceiling_reached: bool = False
    selected_task_id: str | None = None
    report_path: Path | None = None
    suite_path: Path | None = None
    warnings: tuple[str, ...] = ()

    # -- counts ----------------------------------------------------------- #

    @property
    def executed(self) -> tuple[TaskResult, ...]:
        return tuple(result for result in self.results if result.executed)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.outcome is EvalOutcome.FAILED)

    @property
    def not_run_count(self) -> int:
        return sum(1 for result in self.results if result.outcome is EvalOutcome.NOT_RUN)

    @property
    def executed_count(self) -> int:
        return len(self.executed)

    # -- aggregates (Req 15.10) ------------------------------------------- #

    @property
    def pass_rate(self) -> float:
        """Passed tasks over the suite's total task count, as a percentage."""
        if self.suite_size <= 0:
            return 0.0
        return self.passed_count / self.suite_size * 100.0

    @property
    def selected_pass_rate(self) -> float:
        """Passed tasks over the number of tasks selected, as a percentage."""
        if not self.results:
            return 0.0
        return self.passed_count / len(self.results) * 100.0

    @property
    def mean_steps(self) -> float:
        """Mean steps per executed task; 0.0 when nothing executed."""
        executed = self.executed
        if not executed:
            return 0.0
        return sum(result.steps for result in executed) / len(executed)

    @property
    def mean_cost_usd(self) -> float:
        """Mean dollar cost per executed task; 0.0 when nothing executed."""
        executed = self.executed
        if not executed:
            return 0.0
        return sum(result.cost_usd for result in executed) / len(executed)

    @property
    def total_cost_usd(self) -> float:
        """The suite total: what this execution spent (Req 15.11)."""
        return sum(result.cost_usd for result in self.results)

    # -- the verdict (Req 15.15) ------------------------------------------ #

    @property
    def all_passed(self) -> bool:
        """Whether every selected task was executed and recorded as passed."""
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def exit_code(self) -> int:
        """0 only on an all-pass suite; 7 in every other case (Req 15.15)."""
        return exit_codes.FINAL_ANSWER if self.all_passed else exit_codes.EVAL_FAILURE

    @property
    def started_at_utc(self) -> str:
        """The run start as the report records it (Req 15.11)."""
        return _format_timestamp(self.started_at)

    def result_for(self, task_id: str) -> TaskResult | None:
        for result in self.results:
            if result.task_id == task_id:
                return result
        return None


@dataclass(frozen=True)
class ModelSummary:
    """The four figures Req 18.5 states for one model identifier.

    Read off a :class:`SuiteReport` rather than recomputed, so the comparison
    section and that model's own aggregates section can never disagree.
    """

    model_id: str
    passed_count: int
    suite_size: int
    pass_rate: float
    mean_cost_usd: float

    @classmethod
    def of(cls, report: SuiteReport) -> "ModelSummary":
        return cls(
            model_id=report.model_id,
            passed_count=report.passed_count,
            suite_size=report.suite_size,
            pass_rate=report.pass_rate,
            mean_cost_usd=report.mean_cost_usd,
        )


@dataclass(frozen=True)
class ComparisonReport:
    """What one invocation produced, one :class:`SuiteReport` per model (Req 18.5).

    The model dimension lives here and not inside :class:`SuiteReport`: a
    single-model run produces exactly the report it always did, which is what
    Req 18.6 and Req 18.7 require of an unconfigured stretch feature. A
    comparison run holds at most two reports, because
    :attr:`~agent_lab.config.Config.comparison_model_ids` yields at most two
    identifiers.
    """

    reports: tuple[SuiteReport, ...]
    started_at: datetime
    cost_ceiling: float
    report_path: Path | None = None

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(report.model_id for report in self.reports)

    @property
    def model_count(self) -> int:
        return len(self.reports)

    @property
    def primary(self) -> SuiteReport:
        """The first model's report: the one a single-model run would produce."""
        return self.reports[0]

    @property
    def suite_size(self) -> int:
        return self.reports[0].suite_size if self.reports else 0

    @property
    def summaries(self) -> tuple[ModelSummary, ...]:
        """One :class:`ModelSummary` per model, in configuration order."""
        return tuple(ModelSummary.of(report) for report in self.reports)

    @property
    def total_cost_usd(self) -> float:
        """What the whole invocation spent, across every model (Req 15.8)."""
        return sum(report.total_cost_usd for report in self.reports)

    @property
    def ceiling_reached(self) -> bool:
        return any(report.ceiling_reached for report in self.reports)

    @property
    def all_passed(self) -> bool:
        """Whether every task passed, for every model that was run."""
        return bool(self.reports) and all(report.all_passed for report in self.reports)

    @property
    def exit_code(self) -> int:
        """0 only when every task passed for every model; 7 otherwise."""
        return exit_codes.FINAL_ANSWER if self.all_passed else exit_codes.EVAL_FAILURE

    @property
    def started_at_utc(self) -> str:
        return _format_timestamp(self.started_at)

    def report_for(self, model_id: str) -> SuiteReport | None:
        for report in self.reports:
            if report.model_id == model_id:
                return report
        return None


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalPaths:
    """The three paths a suite run writes to, none of them the operator's.

    Held as one value so the collision check of Req 15.4 has a single object to
    inspect and so a test can point the whole suite at ``tmp_path`` in one
    argument (Req 16.9).
    """

    notes_file: Path
    trace_dir: Path
    report_file: Path

    def trace_file_for(self, task_id: str) -> Path:
        """The trace path of one task, embedding its identifier (Req 15.3)."""
        return self.trace_dir / f"{task_id}{TRACE_SUFFIX}"


def build_paths(
    *,
    notes_file: Path | str | None = None,
    trace_dir: Path | str | None = None,
    report_file: Path | str | None = None,
) -> EvalPaths:
    """The evaluation paths, each defaulting to its committed location."""
    return EvalPaths(
        notes_file=Path(notes_file) if notes_file is not None else DEFAULT_EVAL_NOTES_PATH,
        trace_dir=Path(trace_dir) if trace_dir is not None else DEFAULT_EVAL_RUN_DIR,
        report_file=Path(report_file) if report_file is not None else DEFAULT_REPORT_PATH,
    )


#: Anything a model identifier may contain that a path segment should not.
#: Identifiers are commonly ``vendor/model:tag``, and both separators would make
#: a directory where one segment was meant.
_UNSAFE_IN_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def model_path_segment(model_id: str) -> str:
    """``model_id`` as one path segment: ``openai/gpt-4o`` -> ``openai-gpt-4o``.

    The restricted set is the one ``schema.py`` already allows in a task
    identifier, for the same reason: a trace path is built from it (Req 15.3).
    An identifier that reduces to nothing becomes ``model``.
    """
    cleaned = _UNSAFE_IN_SEGMENT.sub("-", model_id).strip("-._")
    return cleaned or "model"


def model_path_segments(model_ids: Sequence[str]) -> tuple[str, ...]:
    """One distinct path segment per identifier, in order.

    Two different identifiers can reduce to the same segment (``a/b`` and
    ``a:b``), and two models writing one directory is the collision Req 15.4
    exists to prevent, so a repeat is suffixed with its position.
    """
    segments: list[str] = []
    for position, model_id in enumerate(model_ids, start=1):
        segment = model_path_segment(model_id)
        if segment in segments:
            segment = f"{segment}-{position}"
        segments.append(segment)
    return tuple(segments)


def paths_for_model(paths: EvalPaths, segment: str) -> EvalPaths:
    """``paths`` with the trace directory moved under ``segment`` (Req 18.5).

    Only the trace directory moves. The report is one document covering both
    models, and the evaluation notes file is truncated before every task
    (Req 15.5), so the models sharing it cannot let one see the other's note --
    they run one after the other, never at once.
    """
    return replace(paths, trace_dir=paths.trace_dir / segment)


def _key(path: Path) -> str:
    """The comparison form of a path: resolved, then case-normalised.

    ``Path.resolve()`` answers the "``./notes.md`` versus an absolute spelling"
    half of Req 15.4 and works on a path that does not exist yet.
    ``os.path.normcase`` answers the other half: on Windows ``NOTES.MD`` and
    ``notes.md`` are the same file, and an eval run that wrote to one while the
    operator's configuration named the other would not leave their file
    byte-identical.
    """
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - an unresolvable path
        resolved = path.absolute()
    return os.path.normcase(str(resolved))


def verify_paths(config: Config, paths: EvalPaths, task_ids: Sequence[str]) -> None:
    """Refuse the run if any evaluation path is one of the operator's (Req 15.4).

    Checked for the notes file and for the trace path of every selected task,
    before the first task starts, so a collision means no task executed, no
    operator file touched, and no report rewritten.

    :raises EvalRunnerError: naming the colliding pair of paths.
    """
    operator = {
        _key(Path(config.notes_file)): ("the configured notes file", Path(config.notes_file)),
        _key(Path(config.trace_file)): ("the configured trace file", Path(config.trace_file)),
    }

    candidates: list[tuple[str, Path]] = [("the evaluation notes file", paths.notes_file)]
    candidates.extend(
        (f"the evaluation trace file of task {task_id!r}", paths.trace_file_for(task_id))
        for task_id in task_ids
    )

    for label, candidate in candidates:
        match = operator.get(_key(candidate))
        if match is None:
            continue
        operator_label, operator_path = match
        raise EvalRunnerError(
            f"the evaluation paths collide with the configured paths of the "
            f"operator: {label} ({candidate}) resolves to the same path as "
            f"{operator_label} ({operator_path}). No evaluation task was run and "
            f"no file was written. Point NOTES_FILE and TRACE_FILE somewhere "
            f"else, or pass different evaluation paths."
        )


# --------------------------------------------------------------------------- #
# Selection (Req 15.1, 15.13, 15.14)
# --------------------------------------------------------------------------- #


def select_tasks(suite: EvalSuite, task_id: str | None = None) -> tuple[EvalTask, ...]:
    """The tasks to execute: all of them in declaration order, or exactly one.

    :raises EvalRunnerError: when ``task_id`` names no task in the suite
        (Req 15.14). Nothing has run when this is raised.
    """
    if task_id is None:
        return tuple(suite)
    task = suite.get(task_id)
    if task is None:
        known = ", ".join(repr(known_id) for known_id in suite.ids()) or "none"
        raise EvalRunnerError(
            f"unknown evaluation task identifier {task_id!r}. No task was run. "
            f"The suite declares: {known}."
        )
    return (task,)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

#: What builds the model client for one task. Injectable so the suite runs
#: offline against a scripted client (Req 16.8).
ModelClientFactory = Callable[[EvalTask, Config], ModelClient]


def _default_client_factory(task: EvalTask, config: Config) -> ModelClient:
    """One real client per task, as the CLI builds one per run (Req 2.8)."""
    return build_model_client(config)


def run_suite(
    config: Config,
    *,
    suite: EvalSuite | None = None,
    suite_path: Path | str | None = None,
    require_composition: bool = False,
    task_id: str | None = None,
    model_client: ModelClient | None = None,
    model_client_factory: ModelClientFactory | None = None,
    retriever: Retriever | None = None,
    system_prompt: str | None = None,
    prices: PriceTable | None = None,
    paths: EvalPaths | None = None,
    notes_file: Path | str | None = None,
    trace_dir: Path | str | None = None,
    report_file: Path | str | None = None,
    cost_ceiling: float | None = None,
    initial_spend: float = 0.0,
    write_report: bool = True,
    narrate: bool = False,
    progress: bool = True,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    started_at: datetime | None = None,
) -> SuiteReport:
    """Execute the selected tasks, print the results, write the report (Req 15).

    :param config: the validated configuration of the operator. Its notes and
        trace paths are *not* written to: they are what the evaluation paths are
        checked against (Req 15.3, Req 15.4).
    :param suite: an already-loaded suite; loaded from ``suite_path`` otherwise.
    :param suite_path: where to read the suite from, defaulting to the committed
        ``evals/tasks.json``.
    :param require_composition: also assert the published 12-task split while
        loading (Req 14.1).
    :param task_id: run only this task (Req 15.13); all of them when ``None``.
    :param model_client: one client used for every task -- the offline path.
    :param model_client_factory: a per-task factory ``(task, config) -> client``.
        Mutually exclusive with ``model_client``; the configured provider's
        client is built per task when neither is given.
    :param retriever: what ``search_docs`` searches. Selected once from the
        configuration when absent, and shared by every task: it is read-only and
        selecting it may open the vector store.
    :param system_prompt: the prompt text; loaded from its versioned resource
        when absent (Req 12.10).
    :param prices: the price table; loaded when absent (Req 3.2).
    :param paths: the evaluation paths as one value, overriding the three
        individual arguments.
    :param cost_ceiling: the suite ceiling in dollars, defaulting to
        ``config.eval_cost_ceiling`` (Req 15.8).
    :param initial_spend: dollars already spent by this invocation before the
        first task of *this* suite execution. Zero for a single-model run; the
        previous model's total in a comparison run, which is what keeps the
        ceiling a bound on the invocation rather than on each model (Req 18.5).
    :param write_report: pass ``False`` to print the results without touching
        ``evals/results.md``.
    :param narrate: whether each task's own steps are narrated (Req 13.1).
        Off by default: twelve narrated runs bury the table.
    :param progress: whether one line per finished task is printed as the suite
        goes, which a real paid run wants and a test does not need.
    :param started_at: the run start recorded in the report; ``now`` in UTC when
        absent (Req 15.11).

    :raises EvalRunnerError: on an unknown identifier (Req 15.14) or a path
        collision (Req 15.4) -- in both cases before any task runs.
    :raises EvalSuiteError: when the suite is defective (Req 14.13). No task has
        run and the report is untouched.
    """
    if model_client is not None and model_client_factory is not None:
        raise ValueError(
            "pass either model_client or model_client_factory, not both: the "
            "first serves every task, the second builds one per task"
        )

    loaded = (
        suite
        if suite is not None
        else load_suite(suite_path, require_composition=require_composition)
    )
    selected = select_tasks(loaded, task_id)

    resolved_paths = paths if paths is not None else build_paths(
        notes_file=notes_file, trace_dir=trace_dir, report_file=report_file
    )
    verify_paths(config, resolved_paths, [task.id for task in selected])

    factory: ModelClientFactory
    if model_client is not None:
        factory = lambda _task, _config: model_client  # noqa: E731 - one expression
    else:
        factory = model_client_factory or _default_client_factory

    prompt = system_prompt if system_prompt is not None else load_system_prompt()[0]
    table = prices if prices is not None else price_table()
    search = retriever if retriever is not None else select_retriever(config)
    ceiling = float(cost_ceiling if cost_ceiling is not None else config.eval_cost_ceiling)
    start = started_at if started_at is not None else datetime.now(timezone.utc)
    model_id = getattr(model_client, "model_id", None) or config.model_id

    results: list[TaskResult] = []
    warnings: list[str] = []
    spent = float(initial_spend)
    ceiling_reached = False

    for position, task in enumerate(selected, start=1):
        # Req 15.8: the comparison happens *before* the task starts, so the
        # ceiling bounds what the suite spends rather than reporting it.
        if spent >= ceiling - COST_TOLERANCE_USD:
            ceiling_reached = True
            results.append(
                TaskResult(
                    task_id=task.id,
                    category=task.category,
                    outcome=EvalOutcome.NOT_RUN,
                    error=None,
                )
            )
            continue

        result = _execute_task(
            task,
            config=config,
            paths=resolved_paths,
            system_prompt=prompt,
            prices=table,
            retriever=search,
            client_factory=factory,
            stdout=stdout,
            stderr=stderr,
            narrate=narrate,
        )
        results.append(result)
        spent += result.cost_usd
        if progress:
            _write(
                stdout,
                f"[{position}/{len(selected)}] {task.id}: {result.outcome.value}  "
                f"steps={result.steps}  cost={_usd(result.cost_usd)}  "
                f"suite={_usd(spent)}",
                config.api_key,
            )

    if ceiling_reached:
        warnings.append(
            f"the suite cost ceiling of {_usd(ceiling)} USD was reached after "
            f"{sum(1 for r in results if r.executed)} task(s); "
            f"{sum(1 for r in results if not r.executed)} task(s) were not run."
        )

    report = SuiteReport(
        model_id=model_id,
        started_at=start,
        results=tuple(results),
        suite_size=len(loaded),
        cost_ceiling=ceiling,
        ceiling_reached=ceiling_reached,
        selected_task_id=task_id,
        report_path=resolved_paths.report_file if write_report else None,
        suite_path=loaded.source_path,
        warnings=tuple(warnings),
    )

    _print_report(report, stdout=stdout, secret=config.api_key)
    if write_report:
        written, problem = write_report_file(
            report, resolved_paths.report_file, secret=config.api_key
        )
        if not written and problem is not None:
            # A report that could not be written must not change the exit code:
            # the tasks ran and their outcomes were printed. Said once, on
            # stderr, like every other warning (Req 13.8).
            _write(stderr or sys.stderr, f"warning: {problem}", config.api_key)
    return report


def run_comparison(
    config: Config,
    *,
    model_ids: Sequence[str] | None = None,
    paths: EvalPaths | None = None,
    notes_file: Path | str | None = None,
    trace_dir: Path | str | None = None,
    report_file: Path | str | None = None,
    cost_ceiling: float | None = None,
    write_report: bool = True,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    started_at: datetime | None = None,
    **options: Any,
) -> ComparisonReport:
    """Run the suite once per configured model identifier (Req 18.5).

    One identifier is the ordinary case and produces exactly what
    :func:`run_suite` followed by :func:`write_report_file` produces: the same
    sections, the same trace paths, no comparison table (Req 18.6, Req 18.7).
    Two identifiers run the suite twice and write one report stating, for each
    identifier, that identifier, its passed count, its pass rate over the
    suite's task count, and its mean cost per task.

    :param model_ids: the identifiers to run, defaulting to
        ``config.comparison_model_ids`` -- at most two, by construction. Each
        model's suite runs with ``config.model_id`` replaced by its identifier,
        so an injected ``model_client_factory`` can tell the models apart from
        the configuration it is handed.
    :param cost_ceiling: the ceiling for the *invocation*, not per model
        (Req 15.8). Each model's suite is started with what the previous models
        already spent, so a two-model run cannot spend twice the ceiling.

    ``options`` are passed through to :func:`run_suite` unchanged.

    :raises ValueError: when a single ``model_client`` is given for more than
        one model -- that client reports one model identifier for every run,
        which would make the comparison a comparison of nothing.
    """
    ids = tuple(model_ids) if model_ids is not None else config.comparison_model_ids
    if not ids:
        raise EvalRunnerError(
            "no model identifier was configured, so there is nothing to run. "
            "Set MODEL_ID."
        )
    if len(ids) > 1 and options.get("model_client") is not None:
        raise ValueError(
            "a comparison run of more than one model identifier needs "
            "model_client_factory, not model_client: one client serves every "
            f"task with one model identifier, and {len(ids)} were configured"
        )

    resolved_paths = paths if paths is not None else build_paths(
        notes_file=notes_file, trace_dir=trace_dir, report_file=report_file
    )
    ceiling = float(cost_ceiling if cost_ceiling is not None else config.eval_cost_ceiling)
    start = started_at if started_at is not None else datetime.now(timezone.utc)
    segments = model_path_segments(ids)

    reports: list[SuiteReport] = []
    spent = 0.0
    for model_id, segment in zip(ids, segments):
        # Req 15.3, Req 15.4: two models running the same tasks need two sets of
        # trace files, so each model's go in their own subdirectory. One model
        # keeps the path it always had.
        model_paths = resolved_paths if len(ids) == 1 else paths_for_model(resolved_paths, segment)
        # `compare_model_id` is cleared so that nothing downstream can read this
        # per-model configuration as a request for a further comparison.
        model_config = replace(config, model_id=model_id, compare_model_id="")
        if len(ids) > 1:
            _write(stdout, "", config.api_key)
            _write(stdout, f"=== model {model_id} ===", config.api_key)
        report = run_suite(
            model_config,
            paths=model_paths,
            cost_ceiling=ceiling,
            initial_spend=spent,
            write_report=False,
            stdout=stdout,
            stderr=stderr,
            started_at=start,
            **options,
        )
        reports.append(report)
        spent += report.total_cost_usd

    comparison = ComparisonReport(
        reports=tuple(reports),
        started_at=start,
        cost_ceiling=ceiling,
        report_path=resolved_paths.report_file if write_report else None,
    )

    if comparison.model_count > 1:
        _print_comparison(comparison, stdout=stdout, secret=config.api_key)
    if write_report:
        written, problem = write_comparison_report_file(
            comparison, resolved_paths.report_file, secret=config.api_key
        )
        if not written and problem is not None:
            _write(stderr or sys.stderr, f"warning: {problem}", config.api_key)
    return comparison


def _execute_task(
    task: EvalTask,
    *,
    config: Config,
    paths: EvalPaths,
    system_prompt: str,
    prices: PriceTable,
    retriever: Retriever,
    client_factory: ModelClientFactory,
    stdout: IO[str] | None,
    stderr: IO[str] | None,
    narrate: bool,
) -> TaskResult:
    """Run one task and evaluate its checks (Req 15.2, 15.3, 15.5, 15.6, 15.7).

    Everything is wired per task: a fresh cost tracker, so the row's cost is
    this task's; a fresh trace logger, so the trace file holds one task; a fresh
    client by default, because a client holds the tool-call identifiers it
    issued. The registry is rebuilt too, because ``save_note`` is given its
    notes path and its gate at construction and both are per-task values.

    The whole body is guarded: Req 15.7 asks that a raised error fail *this*
    task with a truncated message and the suite continue, and the wiring can
    raise as readily as the run can.
    """
    trace_path = paths.trace_file_for(task.id)
    # Req 15.2, 15.3: the operator's paths are replaced and the mode is
    # non-interactive. `Config` is frozen, so this is a new value and the
    # operator's own configuration is unchanged for the next task.
    task_config = replace(
        config,
        notes_file=paths.notes_file,
        trace_file=trace_path,
        interactive=False,
    )
    tracker = CostTracker(prices=prices)

    try:
        # Req 15.5: empty before the task begins, so nothing an earlier task
        # saved is readable by `read_notes` during this one.
        _reset_notes(paths.notes_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        renderer = build_renderer(
            task_config,
            stdout=stdout,
            stderr=stderr,
            narrate=narrate,
            prefer_rich=False,
        )
        # Req 15.2: this constructor refuses to exist unless non-interactive
        # mode was selected, which is the structural half of "no operator input
        # is read".
        gate = AutoApproveGate(task_config)
        registry = build_tool_registry(task_config, gate=gate, retriever=retriever)
        dispatcher = ToolDispatcher(registry, task_config)
        client = client_factory(task, task_config)
        trace_sink = JsonlTraceLogger(
            trace_path,
            api_key=task_config.api_key,
            stream=stderr,
        )
        loop = AgentLoop(
            client,
            dispatcher,
            registry,
            tracker,
            trace_sink=trace_sink,
            renderer=renderer,
            system_prompt=system_prompt,
            # Req 14.2: the task's own step budget, which Req 14.7's check
            # asserts the run stayed inside.
            max_steps=task.max_steps,
            cost_cap=task_config.cost_cap,
            notes_file=paths.notes_file,
        )
        run = loop.run(task.prompt)
    except Exception as exc:  # noqa: BLE001 - Req 15.7 is the reason
        totals = tracker.totals()
        return TaskResult(
            task_id=task.id,
            category=task.category,
            outcome=EvalOutcome.FAILED,
            steps=totals.steps,
            cost_usd=totals.usd,
            error=_truncate(f"{type(exc).__name__}: {exc}"),
            trace_path=str(trace_path),
        )

    evaluated = _evaluate_checks(task, run)
    return TaskResult(
        task_id=task.id,
        category=task.category,
        # Req 15.6: passed only when every check held.
        outcome=EvalOutcome.PASSED
        if all(check.passed for check in evaluated)
        else EvalOutcome.FAILED,
        steps=run.steps,
        cost_usd=run.cost_usd,
        run_outcome=run.outcome,
        checks=evaluated,
        trace_path=run.trace_path or str(trace_path),
        final_answer=run.final_answer,
    )


def _evaluate_checks(task: EvalTask, run: RunResult) -> tuple[CheckResult, ...]:
    """Evaluate every check of ``task`` against ``run`` (Req 15.6).

    Every check is evaluated, including the ones after the first failure: the
    report lists *each* failing identifier, and a reader fixing one assertion
    should not have to rerun the suite to discover the next.
    """
    evaluated: list[CheckResult] = []
    for check in task.checks:
        outcome = check_module.evaluate(check.type, check.params, run)
        evaluated.append(
            CheckResult(
                id=check.id,
                type=check.type,
                passed=outcome.passed,
                reason=outcome.reason,
            )
        )
    return tuple(evaluated)


def _reset_notes(path: Path) -> None:
    """Truncate the evaluation notes file to empty, creating it (Req 15.5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _truncate(message: str, limit: int = ERROR_MESSAGE_MAX_CHARS) -> str:
    """``message`` at no more than ``limit`` characters (Req 15.7)."""
    collapsed = " ".join(message.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)] + "..."


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

TABLE_HEADERS: tuple[str, ...] = (
    "Task",
    "Outcome",
    "Steps",
    "Cost (USD)",
    "Failing checks or error",
)


def _usd(amount: float) -> str:
    """A dollar figure to 4 decimal places (Req 15.9, 15.10, 15.11)."""
    return f"{amount:.4f}"


def _one_dp(value: float) -> str:
    """A figure to 1 decimal place (Req 15.10)."""
    return f"{value:.1f}"


def _format_timestamp(moment: datetime) -> str:
    """``moment`` as the UTC instant the report records (Req 15.11)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_cells(result: TaskResult) -> tuple[str, str, str, str, str]:
    return (
        result.task_id,
        result.outcome.value,
        str(result.steps),
        _usd(result.cost_usd),
        result.detail,
    )


def format_table(report: SuiteReport) -> list[str]:
    """The results table as plain text lines, one row per task (Req 15.9)."""
    rows = [_row_cells(result) for result in report.results]
    widths = [
        max(len(header), *(len(row[column]) for row in rows)) if rows else len(header)
        for column, header in enumerate(TABLE_HEADERS)
    ]
    # Steps and cost read as numbers, so they are right-aligned; the rest are
    # names and sentences, so they are not.
    aligns = ("<", "<", ">", ">", "<")

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            f"{cell:{align}{width}}" for cell, align, width in zip(cells, aligns, widths)
        ).rstrip()

    return [
        line(TABLE_HEADERS),
        "  ".join("-" * width for width in widths),
        *(line(row) for row in rows),
    ]


def format_aggregates(report: SuiteReport) -> list[str]:
    """The aggregate metrics as plain text lines (Req 15.10).

    The pass rate's denominator is the suite's task count, as Req 15.10 states
    it. A run that selected one task gets a second line for the selection's own
    rate, because "8.3%" alone would misdescribe a single passing task
    (Req 15.13).
    """
    lines = [
        f"Pass rate: {_one_dp(report.pass_rate)}% "
        f"({report.passed_count} of {report.suite_size} suite tasks passed)",
        f"Mean steps per executed task: {_one_dp(report.mean_steps)}",
        f"Mean cost per executed task: {_usd(report.mean_cost_usd)} USD",
        f"Suite total cost: {_usd(report.total_cost_usd)} USD",
    ]
    if len(report.results) != report.suite_size:
        lines.insert(
            1,
            f"Selected pass rate: {_one_dp(report.selected_pass_rate)}% "
            f"({report.passed_count} of {len(report.results)} selected)",
        )
    if report.ceiling_reached:
        lines.append(
            f"Suite cost ceiling of {_usd(report.cost_ceiling)} USD reached: "
            f"{report.not_run_count} task(s) were not run."
        )
    return lines


COMPARISON_HEADERS: tuple[str, ...] = (
    "Model identifier",
    "Tasks passed",
    "Pass rate",
    "Mean cost per task (USD)",
)


def _comparison_cells(summary: ModelSummary) -> tuple[str, str, str, str]:
    """One model's row: the four figures Req 18.5 names, in that order."""
    return (
        summary.model_id,
        f"{summary.passed_count} of {summary.suite_size}",
        f"{_one_dp(summary.pass_rate)}%",
        _usd(summary.mean_cost_usd),
    )


def format_comparison(comparison: ComparisonReport) -> list[str]:
    """The per-model comparison as plain text lines (Req 18.5).

    Printed only when more than one model ran: with one model there is nothing
    to compare and Req 18.6 wants the output a reader has always seen.
    """
    rows = [_comparison_cells(summary) for summary in comparison.summaries]
    widths = [
        max(len(header), *(len(row[column]) for row in rows)) if rows else len(header)
        for column, header in enumerate(COMPARISON_HEADERS)
    ]
    aligns = ("<", ">", ">", ">")

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            f"{cell:{align}{width}}" for cell, align, width in zip(cells, aligns, widths)
        ).rstrip()

    return [
        "Model comparison:",
        line(COMPARISON_HEADERS),
        "  ".join("-" * width for width in widths),
        *(line(row) for row in rows),
        f"Invocation total cost: {_usd(comparison.total_cost_usd)} USD "
        f"(ceiling {_usd(comparison.cost_ceiling)} USD for the whole invocation)",
    ]


def format_failing_checks(report: SuiteReport) -> list[str]:
    """One line per failing check, with the reason it did not hold.

    Not required by Req 15.9, which asks only for identifiers, and printed as a
    separate block for that reason. The improvement cycle of Req 15.12 starts
    from *why* a check failed, and "the final answer does not contain '49761'"
    is where that reading begins.
    """
    lines: list[str] = []
    for result in report.results:
        failing = result.failing_checks
        if not failing and result.error is None:
            continue
        lines.append(f"{result.task_id}:")
        if result.error is not None:
            lines.append(f"  error: {result.error}")
        lines.extend(f"  {check.describe()}" for check in failing)
    return lines


def _print_report(
    report: SuiteReport,
    *,
    stdout: IO[str] | None,
    secret: str | None,
) -> None:
    """Print the table, the failing checks and the aggregates (Req 15.9, 15.10)."""
    blocks: list[list[str]] = [format_table(report)]
    failing = format_failing_checks(report)
    if failing:
        blocks.append(["Failing checks:", *failing])
    blocks.append(format_aggregates(report))
    if report.warnings:
        blocks.append(list(report.warnings))

    for block in blocks:
        _write(stdout, "", secret)
        for text in block:
            _write(stdout, text, secret)


def _print_comparison(
    comparison: ComparisonReport,
    *,
    stdout: IO[str] | None,
    secret: str | None,
) -> None:
    """Print the per-model comparison after both models' own output (Req 18.5)."""
    _write(stdout, "", secret)
    for text in format_comparison(comparison):
        _write(stdout, text, secret)


def _write(stream: IO[str] | None, text: str, secret: str | None) -> None:
    """Write one redacted line, resolving ``sys.stdout`` at write time (Req 1.11).

    A closed or broken stream does not fail a suite that already ran: the tasks
    executed, the money was spent, and the report file is the durable record.
    """
    target = stream if stream is not None else sys.stdout
    try:
        target.write(redact(text, secret) + "\n")
        target.flush()
    except (OSError, ValueError):  # a closed or broken stream
        pass


# --------------------------------------------------------------------------- #
# The report file (Req 15.11, Req 15.12)
# --------------------------------------------------------------------------- #


def _md_cell(text: str) -> str:
    """``text`` as one markdown table cell: no pipes, no line breaks."""
    return " ".join(text.split()).replace("|", "\\|") or "-"


def _document_header() -> list[str]:
    """The report's title and its "what is regenerated" comment (Req 15.12)."""
    return [
        "# Eval results",
        "",
        "<!-- Generated by the eval runner (agent_lab.evals.runner). Every",
        f"     section above \"{PRESERVED_HEADING.lstrip('# ')}\" is rewritten on each run;",
        "     that section and everything after it is hand-maintained and kept. -->",
        "",
    ]


def _document(body_lines: Sequence[str], preserved: str | None) -> str:
    """The title, the generated sections, then the hand-maintained tail."""
    body = "\n".join([*_document_header(), *body_lines]).rstrip() + "\n"
    kept = (preserved if preserved is not None else PRESERVED_STUB).strip() + "\n"
    return f"{body}\n{kept}"


def _generated_sections(report: SuiteReport, *, depth: int = 2) -> list[str]:
    """One suite execution's generated sections: metadata, tasks, aggregates.

    ``depth`` is the markdown heading level of those sections. It is 2 for a
    single-model report -- the shape Req 15.11 fixed and Req 18.6 requires to
    stay untouched -- and 3 in a comparison report, where each model's sections
    sit under a level-2 heading naming its model identifier.
    """
    heading = "#" * depth
    subheading = "#" * (depth + 1)
    selected = (
        f"{len(report.results)} of {report.suite_size}"
        if report.selected_task_id is None
        else f"1 of {report.suite_size} (task `{report.selected_task_id}`)"
    )

    lines: list[str] = [
        f"{heading} Run metadata",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model identifier | `{_md_cell(report.model_id)}` |",
        f"| Run start (UTC) | {report.started_at_utc} |",
        f"| Tasks selected | {selected} |",
        f"| Suite total cost (USD) | {_usd(report.total_cost_usd)} |",
        f"| Suite cost ceiling (USD) | {_usd(report.cost_ceiling)} |",
        f"| Cost ceiling reached | {'yes' if report.ceiling_reached else 'no'} |",
        f"| Suite file | `{_md_cell(str(report.suite_path))}` |",
        "",
        f"{heading} Per-task results",
        "",
        "| " + " | ".join(TABLE_HEADERS) + " |",
        "| " + " | ".join("---" for _ in TABLE_HEADERS) + " |",
    ]
    for result in report.results:
        cells = _row_cells(result)
        lines.append(
            "| `"
            + _md_cell(cells[0])
            + "` | "
            + " | ".join(_md_cell(cell) for cell in cells[1:])
            + " |"
        )

    failing = format_failing_checks(report)
    if failing:
        lines += ["", f"{subheading} Why the failing checks did not hold", "", "```"]
        lines += failing
        lines.append("```")

    lines += [
        "",
        f"{heading} Aggregates",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Pass rate | {_one_dp(report.pass_rate)}% "
        f"({report.passed_count} of {report.suite_size}) |",
    ]
    if len(report.results) != report.suite_size:
        lines.append(
            f"| Selected pass rate | {_one_dp(report.selected_pass_rate)}% "
            f"({report.passed_count} of {len(report.results)}) |"
        )
    lines += [
        f"| Mean steps per executed task | {_one_dp(report.mean_steps)} |",
        f"| Mean cost per executed task (USD) | {_usd(report.mean_cost_usd)} |",
        f"| Tasks executed | {report.executed_count} |",
        f"| Tasks passed | {report.passed_count} |",
        f"| Tasks failed | {report.failed_count} |",
        f"| Tasks not run | {report.not_run_count} |",
    ]
    for warning in report.warnings:
        lines += ["", f"> {warning}"]
    return lines


def render_report(report: SuiteReport, *, preserved: str | None = None) -> str:
    """The full ``results.md`` text: generated sections, then the kept one.

    ``preserved`` is the ``## Before and after`` section as the existing file
    holds it, copied through verbatim; :data:`PRESERVED_STUB` stands in the
    first time a report is generated (Req 15.12).
    """
    return _document(_generated_sections(report, depth=2), preserved)


def render_comparison_report(
    comparison: ComparisonReport, *, preserved: str | None = None
) -> str:
    """The ``results.md`` text for a comparison run (Req 18.5).

    One model delegates to :func:`render_report`: the document is then the one
    Req 15.11 fixed, byte for byte, with no trace of the comparison feature
    (Req 18.6, Req 18.7).

    Two models get a ``## Model comparison`` table stating, per identifier, the
    identifier, the passed count, the pass rate over the suite's task count and
    the mean dollar cost per task, followed by each model's own metadata,
    per-task table and aggregates under a heading naming that model.
    """
    if comparison.model_count <= 1:
        return render_report(comparison.primary, preserved=preserved)

    lines: list[str] = [
        "## Model comparison",
        "",
        f"Two model identifiers were configured, so the suite ran once against "
        f"each (Req 18.5). Run start (UTC) {comparison.started_at_utc}. The cost "
        f"ceiling of {_usd(comparison.cost_ceiling)} USD bounds the whole "
        f"invocation, not each model.",
        "",
        "| " + " | ".join(COMPARISON_HEADERS) + " |",
        "| " + " | ".join("---" for _ in COMPARISON_HEADERS) + " |",
    ]
    for summary in comparison.summaries:
        cells = _comparison_cells(summary)
        lines.append(
            "| `"
            + _md_cell(cells[0])
            + "` | "
            + " | ".join(_md_cell(cell) for cell in cells[1:])
            + " |"
        )
    lines += [
        "",
        f"Invocation total cost (USD): {_usd(comparison.total_cost_usd)}",
    ]

    for report in comparison.reports:
        lines += [
            "",
            f"## Results for model `{_md_cell(report.model_id)}`",
            "",
            *_generated_sections(report, depth=3),
        ]
    return _document(lines, preserved)


def preserved_section(path: Path) -> str | None:
    """The hand-maintained tail of an existing report, or ``None`` (Req 15.12).

    Everything from the first ``## Before and after`` heading to the end of the
    file, exactly as written. The heading is matched case-insensitively on a
    line of its own, so a report whose author capitalised it differently still
    keeps its two measurements.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    wanted = PRESERVED_HEADING.casefold()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().casefold() == wanted:
            return "\n".join(lines[index:])
    return None


def write_report_file(
    report: SuiteReport,
    path: Path | str | None = None,
    *,
    secret: str | None = None,
) -> tuple[bool, str | None]:
    """Write ``results.md``, keeping its hand-maintained section (Req 15.11).

    Returns ``(written, problem)``: ``problem`` is a sentence naming the path
    and the reason when the write failed, and ``None`` otherwise. A failure is
    reported rather than raised -- the tasks have already run, and a read-only
    report file must not turn a completed suite into a crash.

    ``secret`` is the provider key, and the whole rendered text passes through
    :func:`~agent_lab.redaction.redact` before it reaches the file: Req 1.11
    names the eval report alongside terminal output and the trace.
    """
    target = Path(path) if path is not None else (report.report_path or DEFAULT_REPORT_PATH)
    text = render_report(report, preserved=preserved_section(target))
    return _write_report_text(target, text, secret=secret)


def write_comparison_report_file(
    comparison: ComparisonReport,
    path: Path | str | None = None,
    *,
    secret: str | None = None,
) -> tuple[bool, str | None]:
    """Write ``results.md`` for a comparison run (Req 18.5, Req 15.11, 15.12).

    Identical to :func:`write_report_file` in every respect a reader can
    observe, including keeping the hand-maintained ``## Before and after``
    section, and identical in output too when only one model ran.
    """
    target = (
        Path(path) if path is not None else (comparison.report_path or DEFAULT_REPORT_PATH)
    )
    text = render_comparison_report(comparison, preserved=preserved_section(target))
    return _write_report_text(target, text, secret=secret)


def _write_report_text(
    target: Path, text: str, *, secret: str | None
) -> tuple[bool, str | None]:
    """Redact ``text`` and write it to ``target`` (Req 1.11)."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(redact(text, secret), encoding="utf-8", newline="\n")
    except OSError as exc:
        return False, (
            f"the eval report at {target} could not be written "
            f"({exc.strerror or exc}); the results above were not recorded."
        )
    return True, None


# --------------------------------------------------------------------------- #
# The entry point the `eval` command calls (task 11.9)
# --------------------------------------------------------------------------- #


def run_eval(
    task_id: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    config: Config | None = None,
    **options: Any,
) -> int:
    """Load configuration, run the suite, and return the process exit code.

    The one function ``agentlab eval`` needs (task 11.9): it owns the parts a
    command handler would otherwise repeat -- reading configuration in
    non-interactive mode (Req 15.2), turning a suite defect or a refusal into
    one stderr line and exit 7 (Req 14.13, Req 15.4, Req 15.14), and returning
    0 only for an all-pass suite (Req 15.15).

    A :class:`~agent_lab.config.ConfigError` is deliberately *not* caught: it is
    exit 6 and the CLI already has one handler for it.

    ``options`` are passed through to :func:`run_suite`, which is how a test
    supplies a scripted client and temporary paths.
    """
    resolved = config
    if resolved is None:
        resolved = load_config(
            env if env is not None else os.environ,
            # Req 15.2: the suite reads no operator input, and this is the only
            # thing that selects that mode.
            interactive=False,
            stream=stderr,
        )
    # The shipped suite is the published one, so its 12-task composition is
    # asserted; a caller pointing at another file asks for shape validation only
    # (Req 14.1, Req 14.13).
    options.setdefault("require_composition", options.get("suite_path") is None)

    # Req 18.5: at most two identifiers, and exactly one unless the operator
    # configured a comparison. With one, the call below is the single-model path
    # that was always here -- no branch taken, no report section added,
    # nothing a reader could tell the feature from (Req 18.6, Req 18.7).
    model_ids = resolved.comparison_model_ids

    try:
        if len(model_ids) > 1:
            comparison = run_comparison(
                resolved,
                model_ids=model_ids,
                task_id=task_id,
                stdout=stdout,
                stderr=stderr,
                **options,
            )
            return comparison.exit_code
        report = run_suite(
            resolved,
            task_id=task_id,
            stdout=stdout,
            stderr=stderr,
            **options,
        )
    except (EvalSuiteError, EvalRunnerError) as exc:
        _write(stderr or sys.stderr, str(exc), resolved.api_key)
        return getattr(exc, "exit_code", exit_codes.EVAL_FAILURE)
    return report.exit_code


__all__ = [
    "COMPARISON_HEADERS",
    "COST_TOLERANCE_USD",
    "DEFAULT_EVAL_NOTES_PATH",
    "DEFAULT_EVAL_RUN_DIR",
    "DEFAULT_REPORT_PATH",
    "ERROR_MESSAGE_MAX_CHARS",
    "EVALS_DIR",
    "PRESERVED_HEADING",
    "PRESERVED_STUB",
    "TABLE_HEADERS",
    "TRACE_SUFFIX",
    "CheckResult",
    "ComparisonReport",
    "EvalOutcome",
    "EvalPaths",
    "EvalRunnerError",
    "ModelClientFactory",
    "ModelSummary",
    "SuiteReport",
    "TaskResult",
    "build_paths",
    "format_aggregates",
    "format_comparison",
    "format_failing_checks",
    "format_table",
    "model_path_segment",
    "model_path_segments",
    "paths_for_model",
    "preserved_section",
    "render_comparison_report",
    "render_report",
    "run_comparison",
    "run_eval",
    "run_suite",
    "select_tasks",
    "verify_paths",
    "write_comparison_report_file",
    "write_report_file",
]
