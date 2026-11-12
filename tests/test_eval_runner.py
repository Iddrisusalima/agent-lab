"""Unit tests for `evals/runner.py`, the suite execution tier (task 11.7, Req 15).

Every test here drives the *whole* runner -- selection, path verification, the
per-task wiring, the checks, the printed table, the report file -- and does it
offline. That is possible because :func:`run_suite` takes its model client as an
argument (Req 16.8): the script comes from ``conftest.FakeModelClient``, the
retriever is a two-line stand-in, the prices are a literal table, and every path
the suite writes to is under ``tmp_path`` (Req 16.9). No provider adapter is
built and no key leaves the fixtures, so the socket guard in ``conftest`` never
has to fire.

What each section asserts, in the order Req 15 states it:

* declaration order (15.1) -- read off the prompts the client was handed, not
  off the result list alone, so the claim is about what actually ran;
* no operator input (15.2) -- ``sys.stdin`` and ``builtins.input`` are replaced
  with objects that raise, and a task that saves a note still completes;
* one trace file per task, named for the task, with the operator's own notes
  and trace files byte-identical afterwards (15.3);
* a path collision refused before anything runs (15.4), including the
  ``..``-spelled form of the same file, since Req 15.4 is about paths that
  *resolve* together;
* the evaluation notes file reset between tasks (15.5), asserted from the far
  side: the ``read_notes`` result the second task's model was shown;
* passed only when every check held, with each failing identifier recorded
  (15.6, 15.9), and a task whose wiring raised failed with a truncated message
  while the suite carried on (15.7);
* the suite cost ceiling stopping the suite, the remainder recorded not run,
  and a non-zero exit (15.8, 15.15);
* the table and the aggregates (15.9, 15.10) and the report file that repeats
  them with the model identifier, the UTC start and the suite total (15.11);
* the hand-maintained ``## Before and after`` section surviving a regeneration
  (15.12);
* single-task selection (15.13) and an unknown identifier refused with nothing
  run (15.14);
* exit code 0 only on an all-pass suite (15.15);
* the API key absent from the report file and from the printed output, the
  report being the third sink Req 1.11 names.

The suites are built in memory from ``schema``'s frozen types rather than loaded
from ``evals/tasks.json``: this file is about the runner, the shipped suite has
its own validation, and a two-task suite keeps each assertion readable. Prices
are a literal table for the reason ``test_cost.py`` gives -- a cost assertion
computed from published prices would start failing when a provider changes them.
"""

from __future__ import annotations

import builtins
import dataclasses
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_lab import exit_codes
from agent_lab.agent_loop import RunOutcome
from agent_lab.config import Config
from agent_lab.evals.runner import (
    DEFAULT_EVAL_NOTES_PATH,
    DEFAULT_EVAL_RUN_DIR,
    DEFAULT_REPORT_PATH,
    ERROR_MESSAGE_MAX_CHARS,
    PRESERVED_HEADING,
    TABLE_HEADERS,
    TRACE_SUFFIX,
    EvalOutcome,
    EvalPaths,
    EvalRunnerError,
    build_paths,
    format_aggregates,
    format_table,
    preserved_section,
    run_eval,
    run_suite,
    select_tasks,
    verify_paths,
)
from agent_lab.evals.schema import (
    SUITE_VERSION,
    EvalCheck,
    EvalSuite,
    EvalTask,
)
from agent_lab.pricing import ModelPrice, PriceTable
from agent_lab.redaction import REDACTION_MARKER

from conftest import (
    FAKE_API_KEY,
    FAKE_MODEL_ID,
    FakeModelClient,
    make_reply,
    make_tool_call,
    make_usage,
)

# --------------------------------------------------------------------------- #
# Prices: $1 per million input tokens, $2 per million output tokens
# --------------------------------------------------------------------------- #

LITERAL_PRICES = PriceTable(
    {FAKE_MODEL_ID: ModelPrice(FAKE_MODEL_ID, 1.0, 2.0, "2026-10-31")}
)

#: Usage priced at exactly $0.10 a call, so a suite ceiling can be written as a
#: number of calls rather than as arithmetic.
DIME = make_usage(input_tokens=100_000, output_tokens=0)

SYSTEM_PROMPT = "You are a test fixture."


# --------------------------------------------------------------------------- #
# Stand-ins
# --------------------------------------------------------------------------- #


class FakeRetriever:
    """``search_docs``'s collaborator, so no vector store is opened (Req 7.7)."""

    def __init__(self, hits: list[tuple[str, str]] | None = None) -> None:
        self.hits = list(hits or [])
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        self.queries.append((query, k))
        return self.hits[:k]


class RefusingStdin:
    """A stdin replacement that fails the test that reads it (Req 15.2)."""

    def readline(self, *args: object) -> str:
        raise AssertionError(
            "the eval runner read operator input; Req 15.2 requires the suite "
            "to run in non-interactive mode and read none"
        )

    def read(self, *args: object) -> str:
        return self.readline()

    def isatty(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Suite builders
# --------------------------------------------------------------------------- #


def check(check_id: str, check_type: str, **params: object) -> EvalCheck:
    """One check, written as a literal."""
    return EvalCheck(id=check_id, type=check_type, params=params)


def task(
    task_id: str,
    *,
    prompt: str | None = None,
    category: str = "single_tool",
    max_steps: int = 4,
    checks: tuple[EvalCheck, ...] = (check("ended", "ended_with_final_answer"),),
) -> EvalTask:
    """One task, defaulting to the weakest check that still asserts something."""
    return EvalTask(
        id=task_id,
        category=category,
        prompt=prompt if prompt is not None else f"Task {task_id}: answer it.",
        max_steps=max_steps,
        checks=checks,
    )


def suite(*tasks: EvalTask, source_path: Path | None = None) -> EvalSuite:
    """A validated suite built in memory, in declaration order (Req 15.1)."""
    return EvalSuite(version=SUITE_VERSION, tasks=tasks, source_path=source_path)


def by_task(clients: dict[str, FakeModelClient]):
    """A ``model_client_factory`` that serves one scripted client per task."""

    def factory(eval_task: EvalTask, _config: Config) -> FakeModelClient:
        return clients[eval_task.id]

    return factory


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def eval_paths(tmp_path: Path) -> EvalPaths:
    """The three evaluation paths, none of them the operator's (Req 16.9).

    The notes file sits beside the trace directory rather than inside it, so a
    test can assert on the exact contents of the trace directory.
    """
    base = tmp_path / "evalrun"
    return EvalPaths(
        notes_file=base / "eval_notes.md",
        trace_dir=base / "traces",
        report_file=base / "results.md",
    )


class Harness:
    """One suite execution plus the streams and the client it was given."""

    def __init__(self, config: Config, paths: EvalPaths) -> None:
        self.config = config
        self.paths = paths
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.retriever = FakeRetriever([("Harbor depth is 14 metres.", "harbor.md")])

    def run(self, eval_suite: EvalSuite, **kwargs: object):
        kwargs.setdefault("write_report", False)
        return run_suite(
            self.config,
            suite=eval_suite,
            retriever=self.retriever,
            system_prompt=SYSTEM_PROMPT,
            prices=LITERAL_PRICES,
            paths=self.paths,
            progress=False,
            stdout=self.stdout,
            stderr=self.stderr,
            **kwargs,
        )

    @property
    def out(self) -> str:
        return self.stdout.getvalue()

    @property
    def err(self) -> str:
        return self.stderr.getvalue()


@pytest.fixture
def harness(tmp_config: Config, eval_paths: EvalPaths) -> Harness:
    return Harness(tmp_config, eval_paths)


def answered(text: str = "The answer is 49761.", *, usage=None) -> FakeModelClient:
    """A client that answers immediately, however many times it is called."""
    return FakeModelClient([make_reply(text, usage=usage)], repeat_last=True)


# --------------------------------------------------------------------------- #
# 1. Declaration order (Req 15.1)
# --------------------------------------------------------------------------- #


def test_every_task_runs_in_declaration_order(harness: Harness) -> None:
    # Req 15.1: no identifier argument executes every task, in the order the
    # suite declares them. Asserted from the prompts the model was handed, so
    # the claim is about execution rather than about the result list's sorting.
    clients = {name: answered() for name in ("gamma", "alpha", "beta")}
    declared = suite(task("gamma"), task("alpha"), task("beta"))

    report = harness.run(declared, model_client_factory=by_task(clients))

    assert [result.task_id for result in report.results] == ["gamma", "alpha", "beta"]
    assert report.suite_size == 3
    assert report.selected_task_id is None
    # Each client saw exactly its own task's prompt as the first user message.
    for name, client in clients.items():
        assert client.call_count == 1
        assert client.calls[0].messages[0].content == f"Task {name}: answer it."
        assert client.calls[0].system_prompt == SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 2. Non-interactive mode (Req 15.2)
# --------------------------------------------------------------------------- #


def test_suite_reads_no_operator_input(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Req 15.2: the gate operates in non-interactive mode, so a save_note step
    # cannot block the suite on a terminal prompt. Both ways a prompt could be
    # read are replaced with objects that fail the test instead of blocking.
    monkeypatch.setattr(sys, "stdin", RefusingStdin())
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the eval runner called input()")
        ),
    )
    # The operator's own configuration says interactive; the runner overrides it
    # per task, which is the only reason this run can finish at all.
    assert harness.config.interactive is True

    client = FakeModelClient(
        [
            make_reply(tool_calls=[make_tool_call("save_note", {"text": "alpha"})]),
            make_reply("Saved."),
        ]
    )
    report = harness.run(suite(task("notes")), model_client=client)

    result = report.result_for("notes")
    assert result.outcome is EvalOutcome.PASSED
    assert result.run_outcome is RunOutcome.FINAL_ANSWER
    # The write happened without an approval prompt, into the evaluation notes.
    assert "alpha" in harness.paths.notes_file.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 3. One trace file per task, and the operator's files left alone (Req 15.3)
# --------------------------------------------------------------------------- #


def test_each_task_writes_its_own_named_trace_file(harness: Harness) -> None:
    # Req 15.3: a trace path reserved for evaluation that includes the task
    # identifier -- one file per task, each holding only that task's steps.
    clients = {"alpha": answered(), "beta": answered()}

    report = harness.run(
        suite(task("alpha"), task("beta")), model_client_factory=by_task(clients)
    )

    written = sorted(path.name for path in harness.paths.trace_dir.iterdir())
    assert written == [f"alpha{TRACE_SUFFIX}", f"beta{TRACE_SUFFIX}"]
    for name in ("alpha", "beta"):
        path = harness.paths.trace_dir / f"{name}{TRACE_SUFFIX}"
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        # One line per step, and a single run behind them (Req 11.8).
        assert len(records) == 1
        assert len({record["run_id"] for record in records}) == 1
        assert report.result_for(name).trace_path == str(path)


def test_the_operator_notes_and_trace_files_are_byte_identical(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 15.3: whatever the suite writes, the configured notes file and trace
    # file of the operator are exactly as they were.
    notes_before = b"# my notes\n\n[2026-01-01T00:00:00Z] a note of my own\n"
    trace_before = b'{"run_id": "mine", "step": 1}\n'
    Path(tmp_config.notes_file).write_bytes(notes_before)
    Path(tmp_config.trace_file).parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_config.trace_file).write_bytes(trace_before)

    harness = Harness(tmp_config, eval_paths)
    client = FakeModelClient(
        [
            make_reply(tool_calls=[make_tool_call("save_note", {"text": "eval note"})]),
            make_reply(tool_calls=[make_tool_call("read_notes", {})]),
            make_reply("Done."),
        ]
    )
    harness.run(suite(task("writes", max_steps=4)), model_client=client)

    assert Path(tmp_config.notes_file).read_bytes() == notes_before
    assert Path(tmp_config.trace_file).read_bytes() == trace_before
    # And the configuration object itself was not mutated for the next caller.
    assert tmp_config.interactive is True
    assert Path(tmp_config.notes_file) != eval_paths.notes_file


# --------------------------------------------------------------------------- #
# 4. A path collision is refused before anything runs (Req 15.4)
# --------------------------------------------------------------------------- #


def collision_cases(config: Config, tmp_path: Path) -> dict[str, EvalPaths]:
    """Evaluation paths that each resolve onto one of the operator's."""
    operator_notes = Path(config.notes_file)
    operator_trace = Path(config.trace_file)
    return {
        # The same notes file, named directly.
        "notes": EvalPaths(
            notes_file=operator_notes,
            trace_dir=tmp_path / "evalrun" / "traces",
            report_file=tmp_path / "evalrun" / "results.md",
        ),
        # The same notes file, spelled through a parent hop: Req 15.4 is about
        # paths that *resolve* to the same file, not about equal strings.
        "notes-resolved": EvalPaths(
            notes_file=operator_notes.parent / "sub" / ".." / operator_notes.name,
            trace_dir=tmp_path / "evalrun" / "traces",
            report_file=tmp_path / "evalrun" / "results.md",
        ),
        # A trace directory in which this task's file is the operator's trace.
        "trace": EvalPaths(
            notes_file=tmp_path / "evalrun" / "eval_notes.md",
            trace_dir=operator_trace.parent,
            report_file=tmp_path / "evalrun" / "results.md",
        ),
    }


@pytest.mark.parametrize("case", ["notes", "notes-resolved", "trace"])
def test_path_collision_runs_nothing_and_writes_nothing(
    tmp_config: Config, tmp_path: Path, case: str
) -> None:
    # Req 15.4: no task executed, an error naming the collision, a non-zero
    # exit code -- and an existing report left as it was.
    paths = collision_cases(tmp_config, tmp_path)[case]
    report_file = tmp_path / "existing" / "results.md"
    report_file.parent.mkdir()
    report_file.write_text("the previous report\n", encoding="utf-8")
    paths = dataclasses.replace(paths, report_file=report_file)

    client = answered()
    harness = Harness(tmp_config, paths)
    # The operator's trace file is named `run.jsonl`, so the task identifier
    # `run` is what makes the trace case collide.
    declared = suite(task(Path(tmp_config.trace_file).stem))

    with pytest.raises(EvalRunnerError) as raised:
        harness.run(declared, model_client=client, write_report=True)

    message = str(raised.value)
    assert "collide" in message
    assert "No evaluation task was run" in message
    assert raised.value.exit_code == exit_codes.EVAL_FAILURE
    assert client.call_count == 0
    assert report_file.read_text(encoding="utf-8") == "the previous report\n"
    # Nothing was created either: the refusal precedes the first task's wiring.
    assert not paths.trace_dir.exists()
    assert not paths.notes_file.exists()


def test_verify_paths_accepts_paths_that_are_not_the_operators(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # The other half of Req 15.4: distinct paths are accepted, so the check
    # cannot be passing by refusing everything.
    verify_paths(tmp_config, eval_paths, ["alpha", "beta"])


# --------------------------------------------------------------------------- #
# 5. The evaluation notes file is reset between tasks (Req 15.5)
# --------------------------------------------------------------------------- #


def test_notes_are_reset_so_an_earlier_task_is_unreadable(harness: Harness) -> None:
    # Req 15.5: a note the first task saved must not be readable by read_notes
    # during the second. Asserted from what the second task's model was shown.
    writer = FakeModelClient(
        [
            make_reply(
                tool_calls=[make_tool_call("save_note", {"text": "SECRETFROMTASKONE"})]
            ),
            make_reply("Saved."),
        ]
    )
    reader = FakeModelClient(
        [make_reply(tool_calls=[make_tool_call("read_notes", {})]), make_reply("Done.")]
    )

    report = harness.run(
        suite(task("writer"), task("reader")),
        model_client_factory=by_task({"writer": writer, "reader": reader}),
    )

    assert report.result_for("writer").outcome is EvalOutcome.PASSED
    shown = "\n".join(reader.tool_results_submitted())
    assert "SECRETFROMTASKONE" not in shown
    # The read happened and reported the file as empty rather than silently
    # returning nothing (Req 10.3).
    assert "empty" in shown.lower()
    assert "SECRETFROMTASKONE" not in harness.paths.notes_file.read_text("utf-8")


# --------------------------------------------------------------------------- #
# 6. Checks decide the outcome (Req 15.6, 15.9)
# --------------------------------------------------------------------------- #


def test_a_task_passes_only_when_every_check_holds(harness: Harness) -> None:
    # Req 15.6: all four checks hold, so the task is passed and no failing
    # identifier is recorded.
    passing = task(
        "allgood",
        checks=(
            check("ended", "ended_with_final_answer"),
            check("value", "answer_contains", text="49761"),
            check("steps", "steps_at_most", count=4),
            check("notools", "no_tool_called"),
        ),
    )

    report = harness.run(suite(passing), model_client=answered())

    result = report.result_for("allgood")
    assert result.outcome is EvalOutcome.PASSED
    assert [c.id for c in result.checks] == ["ended", "value", "steps", "notools"]
    assert all(c.passed for c in result.checks)
    assert result.failing_checks == ()
    assert result.detail == "-"
    assert report.all_passed is True


def test_one_failing_check_fails_the_task_and_names_every_failure(
    harness: Harness,
) -> None:
    # Req 15.6 and Req 15.9: every check is evaluated even after the first
    # failure, and each failing identifier is recorded and printed.
    mixed = task(
        "mixed",
        checks=(
            check("ended", "ended_with_final_answer"),
            check("value", "answer_contains", text="an answer nobody gave"),
            check("tool", "tool_called", tool="calculator"),
        ),
    )

    report = harness.run(suite(mixed), model_client=answered())

    result = report.result_for("mixed")
    assert result.outcome is EvalOutcome.FAILED
    assert [c.id for c in result.failing_checks] == ["value", "tool"]
    # The passing check was still evaluated and recorded.
    assert len(result.checks) == 3
    assert result.detail == "value, tool"
    assert result.error is None
    # Req 15.9: the row and the identifiers reach the printed output.
    assert "value, tool" in harness.out
    # Req 15.15: a failed task is a non-zero exit.
    assert report.exit_code == exit_codes.EVAL_FAILURE


def test_a_raised_error_fails_one_task_and_the_suite_continues(
    harness: Harness,
) -> None:
    # Req 15.7: the error is recorded against that task, truncated to 500
    # characters, and the remaining tasks still run.
    survivor = answered()

    def factory(eval_task: EvalTask, _config: Config):
        if eval_task.id == "broken":
            raise RuntimeError("wiring failed because " + "x" * 800)
        return survivor

    report = harness.run(
        suite(task("broken"), task("works")), model_client_factory=factory
    )

    broken = report.result_for("broken")
    assert broken.outcome is EvalOutcome.FAILED
    assert broken.error is not None
    assert broken.error.startswith("RuntimeError: wiring failed because")
    assert len(broken.error) <= ERROR_MESSAGE_MAX_CHARS
    assert broken.error.endswith("...")
    assert broken.detail == broken.error
    # The suite carried on and the second task was executed normally.
    assert report.result_for("works").outcome is EvalOutcome.PASSED
    assert survivor.call_count == 1


# --------------------------------------------------------------------------- #
# 7. The suite cost ceiling (Req 15.8, Req 15.15)
# --------------------------------------------------------------------------- #


def test_cost_ceiling_stops_the_suite_and_records_the_rest_not_run(
    harness: Harness,
) -> None:
    # Req 15.8: one task costs $0.10, the ceiling is $0.05, so the first task
    # runs and every later one is recorded not run with no model call made.
    clients = {name: answered(usage=DIME) for name in ("one", "two", "three")}

    report = harness.run(
        suite(task("one"), task("two"), task("three")),
        model_client_factory=by_task(clients),
        cost_ceiling=0.05,
    )

    outcomes = [result.outcome for result in report.results]
    assert outcomes == [
        EvalOutcome.PASSED,
        EvalOutcome.NOT_RUN,
        EvalOutcome.NOT_RUN,
    ]
    assert clients["two"].call_count == 0
    assert clients["three"].call_count == 0
    # A not-run row carries no steps and no cost, and is not an executed task.
    for name in ("two", "three"):
        result = report.result_for(name)
        assert (result.steps, result.cost_usd, result.executed) == (0, 0.0, False)
    assert report.ceiling_reached is True
    assert report.not_run_count == 2
    assert report.total_cost_usd == pytest.approx(0.10)
    # Req 15.8: the ceiling is reported, and the suite exits non-zero.
    assert any("ceiling" in warning for warning in report.warnings)
    assert "ceiling" in harness.out
    assert report.exit_code == exit_codes.EVAL_FAILURE


def test_a_ceiling_above_the_whole_suite_runs_everything(harness: Harness) -> None:
    # The complement, so the ceiling test above is not passing because the
    # runner stops after one task regardless.
    clients = {name: answered(usage=DIME) for name in ("one", "two")}

    report = harness.run(
        suite(task("one"), task("two")),
        model_client_factory=by_task(clients),
        cost_ceiling=5.0,
    )

    assert report.ceiling_reached is False
    assert report.not_run_count == 0
    assert report.executed_count == 2
    assert report.total_cost_usd == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# 8. Selection (Req 15.13, Req 15.14)
# --------------------------------------------------------------------------- #


def test_one_identifier_runs_only_that_task(harness: Harness) -> None:
    # Req 15.13: exactly the named task runs, and the table and the aggregates
    # are printed for it.
    clients = {name: answered() for name in ("alpha", "beta", "gamma")}

    report = harness.run(
        suite(task("alpha"), task("beta"), task("gamma")),
        model_client_factory=by_task(clients),
        task_id="beta",
    )

    assert [result.task_id for result in report.results] == ["beta"]
    assert clients["alpha"].call_count == 0
    assert clients["gamma"].call_count == 0
    assert report.suite_size == 3
    assert report.selected_task_id == "beta"
    # Req 15.10's denominator is the suite; a one-task selection also reports
    # its own rate so "33.3%" is not the only number a reader sees.
    assert report.pass_rate == pytest.approx(100 / 3)
    assert report.selected_pass_rate == pytest.approx(100.0)
    assert "Selected pass rate: 100.0%" in harness.out
    # Only the selected task's trace file exists (Req 15.3).
    assert [p.name for p in harness.paths.trace_dir.iterdir()] == [
        f"beta{TRACE_SUFFIX}"
    ]
    # Req 15.15: a selection that passed is exit code 0.
    assert report.exit_code == exit_codes.FINAL_ANSWER


def test_unknown_identifier_runs_nothing(harness: Harness) -> None:
    # Req 15.14: the identifier is reported as unknown, no task runs, and the
    # refusal carries the non-zero exit code.
    client = answered()
    report_file = harness.paths.report_file
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("the previous report\n", encoding="utf-8")

    with pytest.raises(EvalRunnerError) as raised:
        harness.run(
            suite(task("alpha"), task("beta")),
            model_client=client,
            task_id="delta",
            write_report=True,
        )

    message = str(raised.value)
    assert "unknown evaluation task identifier 'delta'" in message
    assert "No task was run" in message
    # The message lists what the suite does declare, so the typo is fixable.
    assert "'alpha'" in message and "'beta'" in message
    assert raised.value.exit_code == exit_codes.EVAL_FAILURE
    assert client.call_count == 0
    assert not harness.paths.trace_dir.exists()
    assert report_file.read_text(encoding="utf-8") == "the previous report\n"


def test_select_tasks_is_the_whole_suite_or_exactly_one() -> None:
    # Req 15.1 and Req 15.13 in one place: the selection function itself.
    declared = suite(task("a"), task("b"))

    assert select_tasks(declared) == declared.tasks
    assert [t.id for t in select_tasks(declared, "b")] == ["b"]
    with pytest.raises(EvalRunnerError):
        select_tasks(declared, "B")  # identifiers are case-sensitive


def test_run_eval_turns_a_refusal_into_one_line_and_exit_seven(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 15.14 through the entry point the `eval` command calls: the message
    # goes to stderr and the exit code is 7, with nothing raised at the caller.
    stdout, stderr = io.StringIO(), io.StringIO()

    code = run_eval(
        "nope",
        config=tmp_config,
        stdout=stdout,
        stderr=stderr,
        suite=suite(task("alpha")),
        model_client=answered(),
        retriever=FakeRetriever(),
        system_prompt=SYSTEM_PROMPT,
        prices=LITERAL_PRICES,
        paths=eval_paths,
        progress=False,
        write_report=False,
    )

    assert code == exit_codes.EVAL_FAILURE
    assert "unknown evaluation task identifier 'nope'" in stderr.getvalue()
    assert stdout.getvalue() == ""


# --------------------------------------------------------------------------- #
# 9. The printed table and aggregates (Req 15.9, Req 15.10)
# --------------------------------------------------------------------------- #


def test_table_and_aggregates_are_printed(harness: Harness) -> None:
    # Req 15.9: one row per task with the identifier, the outcome word, the
    # step count and the cost to 4 decimal places. Req 15.10: the three
    # aggregate figures, the pass rate over the suite's task count.
    good = task("good", checks=(check("ended", "ended_with_final_answer"),))
    bad = task("bad", checks=(check("value", "answer_contains", text="absent text"),))
    clients = {"good": answered(usage=DIME), "bad": answered(usage=DIME)}

    report = harness.run(suite(good, bad), model_client_factory=by_task(clients))
    printed = harness.out

    for header in TABLE_HEADERS:
        assert header in printed
    assert "good" in printed and "passed" in printed
    assert "bad" in printed and "failed" in printed
    # The cost column is the 4-decimal figure Req 15.9 asks for.
    assert "0.1000" in printed
    assert "Pass rate: 50.0% (1 of 2 suite tasks passed)" in printed
    assert "Mean steps per executed task: 1.0" in printed
    assert "Mean cost per executed task: 0.1000 USD" in printed
    assert "Suite total cost: 0.2000 USD" in printed
    # The same lines are what the formatters produce, called directly.
    assert format_table(report)[0].startswith("Task")
    assert len(format_table(report)) == 2 + len(report.results)
    assert format_aggregates(report)[0].startswith("Pass rate: 50.0%")
    # Req 15.9's "why" block names the failing check and its reason.
    assert "value (answer_contains)" in printed


# --------------------------------------------------------------------------- #
# 10. The report file (Req 15.11, Req 15.12)
# --------------------------------------------------------------------------- #

STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_report_file_records_the_run(harness: Harness) -> None:
    # Req 15.11: the per-task outcomes, the aggregates, the model identifier,
    # the UTC start, and the suite total cost rounded to 4 decimal places.
    good = task("good", checks=(check("ended", "ended_with_final_answer"),))
    bad = task("bad", checks=(check("value", "answer_contains", text="absent text"),))
    clients = {"good": answered(usage=DIME), "bad": answered(usage=DIME)}

    report = harness.run(
        suite(good, bad, source_path=Path("evals/tasks.json")),
        model_client_factory=by_task(clients),
        write_report=True,
        started_at=STARTED_AT,
    )

    text = harness.paths.report_file.read_text(encoding="utf-8")
    assert text.startswith("# Eval results")
    assert FAKE_MODEL_ID in text
    assert "2026-01-02T03:04:05Z" in text
    assert "| Suite total cost (USD) | 0.2000 |" in text
    # One markdown row per task, carrying the same cells as the printed table.
    assert "| `good` | passed |" in text
    assert "| `bad` | failed |" in text
    assert "| Pass rate | 50.0% (1 of 2) |" in text
    assert "| Mean steps per executed task | 1.0 |" in text
    assert "| Mean cost per executed task (USD) | 0.1000 |" in text
    assert report.report_path == harness.paths.report_file
    # Req 15.12: the hand-maintained section exists from the first write, as a
    # stub for task 11.10 to fill in.
    assert PRESERVED_HEADING in text


def test_the_before_and_after_section_survives_regeneration(harness: Harness) -> None:
    # Req 15.12: everything from the `## Before and after` heading down is
    # hand-maintained, so a regeneration copies it through unchanged.
    kept = (
        f"{PRESERVED_HEADING}\n\n"
        "- **Before:** 58.3% -- model `gpt-4o-mini`, measured 2026-02-01\n"
        "- **After:** 83.3% -- model `gpt-4o-mini`, measured 2026-02-08\n\n"
        "### Changes made between the two measurements\n\n"
        "- sharpened the `search_docs` description\n"
    )
    harness.paths.report_file.parent.mkdir(parents=True, exist_ok=True)
    harness.paths.report_file.write_text(
        "# Eval results\n\nstale generated content from the previous run\n\n" + kept,
        encoding="utf-8",
    )

    harness.run(
        suite(task("alpha")),
        model_client=answered(),
        write_report=True,
        started_at=STARTED_AT,
    )

    text = harness.paths.report_file.read_text(encoding="utf-8")
    assert kept.strip() in text
    assert "stale generated content" not in text
    assert "## Per-task results" in text
    assert text.index("## Per-task results") < text.index(PRESERVED_HEADING)
    # The reader of the kept section agrees with what was written.
    assert preserved_section(harness.paths.report_file).startswith(PRESERVED_HEADING)


def test_an_unwritable_report_warns_without_changing_the_exit_code(
    harness: Harness,
) -> None:
    # The report is the durable record, but the tasks have already run: a write
    # failure is one warning on stderr, not a crash and not a failed suite.
    blocked = dataclasses.replace(
        harness.paths, report_file=harness.paths.trace_dir / "results.md"
    )
    harness.paths = blocked
    # A directory where the file should be makes the write fail for a reason
    # the operator can act on.
    blocked.report_file.mkdir(parents=True, exist_ok=True)

    report = harness.run(suite(task("alpha")), model_client=answered(), write_report=True)

    assert report.exit_code == exit_codes.FINAL_ANSWER
    assert "warning:" in harness.err
    assert str(blocked.report_file) in harness.err


# --------------------------------------------------------------------------- #
# 11. Redaction: the report is the third sink (Req 1.11)
# --------------------------------------------------------------------------- #


def test_the_api_key_never_reaches_the_report_or_the_terminal(
    harness: Harness,
) -> None:
    # Req 1.11 names the eval report alongside terminal output and the trace.
    # The model answers with the key, a failing check quotes the answer into
    # its reason, and that reason is printed and written -- so both sinks are
    # exercised by one run.
    leaky = answered(f"The configured key is {FAKE_API_KEY} and the total is 3.")
    declared = suite(
        task("leaky", checks=(check("value", "answer_contains", text="absent text"),))
    )

    harness.run(declared, model_client=leaky, write_report=True)

    report_text = harness.paths.report_file.read_text(encoding="utf-8")
    assert FAKE_API_KEY not in report_text
    assert REDACTION_MARKER in report_text
    assert FAKE_API_KEY not in harness.out
    assert FAKE_API_KEY not in harness.err
    assert REDACTION_MARKER in harness.out
    # And not in the trace either, which is the sink the logger owns.
    trace = (harness.paths.trace_dir / f"leaky{TRACE_SUFFIX}").read_text("utf-8")
    assert FAKE_API_KEY not in trace


# --------------------------------------------------------------------------- #
# 12. The exit code (Req 15.15)
# --------------------------------------------------------------------------- #


def test_exit_code_is_zero_only_when_every_selected_task_passed(
    harness: Harness,
) -> None:
    # Req 15.15: 0 for an all-pass suite.
    clients = {name: answered() for name in ("one", "two")}
    report = harness.run(
        suite(task("one"), task("two")), model_client_factory=by_task(clients)
    )

    assert report.all_passed is True
    assert report.passed_count == 2
    assert report.exit_code == exit_codes.FINAL_ANSWER


def test_run_eval_returns_the_reports_exit_code(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # The same verdict through the entry point, for a pass and for a failure.
    def invoke(declared: EvalSuite) -> int:
        return run_eval(
            config=tmp_config,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            suite=declared,
            model_client=answered(),
            retriever=FakeRetriever(),
            system_prompt=SYSTEM_PROMPT,
            prices=LITERAL_PRICES,
            paths=eval_paths,
            progress=False,
            write_report=False,
        )

    passing = task("ok", checks=(check("ended", "ended_with_final_answer"),))
    failing = task("no", checks=(check("v", "answer_contains", text="absent text"),))

    assert invoke(suite(passing)) == exit_codes.FINAL_ANSWER
    assert invoke(suite(passing, failing)) == exit_codes.EVAL_FAILURE


# --------------------------------------------------------------------------- #
# 13. The committed defaults, asserted without writing to them
# --------------------------------------------------------------------------- #


def test_the_default_paths_are_the_committed_ones() -> None:
    # Nothing in this file writes to these, which is the point: the defaults
    # are asserted as values so a redirected test cannot drift from them.
    defaults = build_paths()

    assert defaults.notes_file == DEFAULT_EVAL_NOTES_PATH
    assert defaults.trace_dir == DEFAULT_EVAL_RUN_DIR
    assert defaults.report_file == DEFAULT_REPORT_PATH
    assert DEFAULT_REPORT_PATH.name == "results.md"
    # Req 1.4: an eval run's own files sit under a `traces` directory, which
    # `.gitignore` already excludes at any depth.
    assert DEFAULT_EVAL_RUN_DIR.name == "traces"
    assert DEFAULT_EVAL_NOTES_PATH.parent == DEFAULT_EVAL_RUN_DIR
    # Req 15.3: the trace file of a task is named for that task.
    assert defaults.trace_file_for("t1").name == f"t1{TRACE_SUFFIX}"
