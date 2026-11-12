"""Unit tests for the optional second-model comparison (task 14.4, Req 18.5).

The feature has two halves and this file is mostly about the first one.

**Unconfigured, nothing may change.** Req 18.6 and Req 18.7 say a stretch
feature that was not switched on leaves the behaviour of Requirements 1-17
exactly as it was, so the first test here renders a report through the
comparison entry point with ``COMPARE_MODEL_ID`` unset and compares it
*character for character* against the report the single-model path produces.
That is a stronger claim than "no comparison heading appears", and it is the
claim that matters: a reader of ``evals/results.md`` must not be able to tell
the feature exists.

**Configured, the report gains one section and nothing else.** The remaining
tests assert what Req 18.5 states -- every task of the suite run once against
each of the at most two identifiers, and per identifier the identifier, the
passed count, the pass rate over the suite's task count and the mean dollar cost
per task -- plus the two things the feature could plausibly break: the per-task
trace files of Req 15.3, which two models would otherwise write to the same
twelve paths, and the cost ceiling of Req 15.8, which bounds the invocation and
not the model.

Everything runs offline against ``conftest.FakeModelClient``, one scripted
client per model, selected by the model identifier in the configuration the
runner hands the factory. Every path is under ``tmp_path``: the repository's own
``evals/`` directory is never written to, and the hand-maintained section of the
real report is never at risk (Req 16.9).

Prices are a literal table for the reason the other eval tests give -- an
assertion computed from published prices would start failing when a provider
changes them -- and the two models are priced differently so that a mean cost
per task reported against the wrong model would be visible.
"""

from __future__ import annotations

import dataclasses
import io
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_lab import exit_codes
from agent_lab.config import Config
from agent_lab.evals.runner import (
    PRESERVED_HEADING,
    EvalPaths,
    ModelSummary,
    build_paths,
    model_path_segment,
    model_path_segments,
    render_comparison_report,
    render_report,
    run_comparison,
    run_eval,
    run_suite,
)
from agent_lab.evals.schema import (
    SUITE_VERSION,
    EvalCheck,
    EvalSuite,
    EvalTask,
    load_suite,
)
from agent_lab.pricing import ModelPrice, PriceTable

from conftest import FAKE_MODEL_ID, FakeModelClient, make_reply, make_usage

# --------------------------------------------------------------------------- #
# The two models and their prices
# --------------------------------------------------------------------------- #

#: The configured model: $1 per million input tokens.
PRIMARY_MODEL = FAKE_MODEL_ID

#: The comparison model. Spelled with a slash on purpose: a model identifier is
#: commonly ``vendor/model``, and the trace directory naming has to cope with it.
SECOND_MODEL = "vendor/model-b"

#: $2 per million input tokens for the second model, so the two mean costs per
#: task differ and a figure attributed to the wrong model is visible.
LITERAL_PRICES = PriceTable(
    {
        PRIMARY_MODEL: ModelPrice(PRIMARY_MODEL, 1.0, 0.0, "2026-10-31"),
        SECOND_MODEL: ModelPrice(SECOND_MODEL, 2.0, 0.0, "2026-10-31"),
    }
)

#: 100,000 input tokens: $0.10 a call on the primary model, $0.20 on the second.
DIME = make_usage(input_tokens=100_000, output_tokens=0)

SYSTEM_PROMPT = "You are a test fixture."

FIXED_START = datetime(2026, 11, 5, 9, 30, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Stand-ins and builders
# --------------------------------------------------------------------------- #


class FakeRetriever:
    """``search_docs``'s collaborator, so no vector store is opened (Req 7.7)."""

    def __init__(self, hits: list[tuple[str, str]] | None = None) -> None:
        self.hits = list(hits or [])

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        return self.hits[:k]


def check(check_id: str, check_type: str, **params: object) -> EvalCheck:
    return EvalCheck(id=check_id, type=check_type, params=params)


def task(task_id: str, *, category: str = "single_tool") -> EvalTask:
    """One task whose single check holds for any run that answered."""
    return EvalTask(
        id=task_id,
        category=category,
        prompt=f"Task {task_id}: answer it.",
        max_steps=4,
        checks=(check("ended", "ended_with_final_answer"),),
    )


def suite_of(count: int) -> EvalSuite:
    """A validated in-memory suite of ``count`` tasks, in declaration order."""
    return EvalSuite(
        version=SUITE_VERSION,
        tasks=tuple(task(f"task-{index:02d}") for index in range(1, count + 1)),
        source_path=Path("tests") / "in-memory-suite.json",
    )


def client_for(model_id: str, *, text: str = "The answer is 49761.") -> FakeModelClient:
    """A client that answers immediately, priced as ``model_id`` (Req 3.2)."""
    return FakeModelClient(
        [make_reply(text, usage=DIME)], model_id=model_id, repeat_last=True
    )


def by_model(clients: dict[str, FakeModelClient]):
    """A ``model_client_factory`` selecting a client by configured model id.

    This is the seam Req 18.5 is driven through offline: :func:`run_comparison`
    replaces ``config.model_id`` per model before calling the factory, so the
    factory can tell which model's suite execution it is being asked for.
    """

    def factory(_task: EvalTask, config: Config) -> FakeModelClient:
        return clients[config.model_id]

    return factory


def comparing(config: Config, second: str = SECOND_MODEL) -> Config:
    """``config`` with a comparison model identifier configured (Req 18.5)."""
    return dataclasses.replace(config, compare_model_id=second)


@pytest.fixture
def eval_paths(tmp_path: Path) -> EvalPaths:
    """The three evaluation paths, all under ``tmp_path`` (Req 16.9)."""
    base = tmp_path / "evalrun"
    return EvalPaths(
        notes_file=base / "eval_notes.md",
        trace_dir=base / "traces",
        report_file=base / "results.md",
    )


def compare(
    config: Config,
    eval_suite: EvalSuite,
    paths: EvalPaths,
    clients: dict[str, FakeModelClient],
    **kwargs: object,
):
    """Run the comparison entry point offline, returning its report."""
    kwargs.setdefault("write_report", False)
    kwargs.setdefault("started_at", FIXED_START)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("stdout", io.StringIO())
    kwargs.setdefault("stderr", io.StringIO())
    return run_comparison(
        config,
        suite=eval_suite,
        model_client_factory=by_model(clients),
        retriever=FakeRetriever([("Harbor depth is 14 metres.", "harbor.md")]),
        system_prompt=SYSTEM_PROMPT,
        prices=LITERAL_PRICES,
        paths=paths,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1. Unconfigured: one model, and a report identical in shape (Req 18.6, 18.7)
# --------------------------------------------------------------------------- #


def test_unconfigured_configuration_names_one_model(tmp_config: Config) -> None:
    # The config surface is the whole switch: unset means one identifier.
    assert tmp_config.compare_model_id == ""
    assert tmp_config.comparison_model_ids == (PRIMARY_MODEL,)


def test_unconfigured_report_is_byte_identical_to_the_single_model_report(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 18.6, Req 18.7: with the feature unconfigured the report a reader
    # opens must be the one the single-model path has always written. Not
    # "similar" -- the same characters.
    eval_suite = suite_of(3)
    clients = {PRIMARY_MODEL: client_for(PRIMARY_MODEL)}

    comparison = compare(tmp_config, eval_suite, eval_paths, clients)
    assert comparison.model_count == 1

    single = run_suite(
        tmp_config,
        suite=eval_suite,
        model_client_factory=by_model({PRIMARY_MODEL: client_for(PRIMARY_MODEL)}),
        retriever=FakeRetriever(),
        system_prompt=SYSTEM_PROMPT,
        prices=LITERAL_PRICES,
        paths=eval_paths,
        progress=False,
        write_report=False,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        started_at=FIXED_START,
    )

    assert render_comparison_report(comparison) == render_report(single)


def test_unconfigured_report_has_no_comparison_section(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # The headings Req 15.11 fixed, and none of the comparison ones.
    comparison = compare(
        tmp_config, suite_of(2), eval_paths, {PRIMARY_MODEL: client_for(PRIMARY_MODEL)}
    )
    text = render_comparison_report(comparison)

    assert "## Run metadata" in text
    assert "## Per-task results" in text
    assert "## Aggregates" in text
    assert "Model comparison" not in text
    assert "Results for model" not in text


def test_unconfigured_run_writes_the_ordinary_report_file(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # The same claim, through the writer rather than the renderer.
    compare(
        tmp_config,
        suite_of(2),
        eval_paths,
        {PRIMARY_MODEL: client_for(PRIMARY_MODEL)},
        write_report=True,
    )
    text = eval_paths.report_file.read_text(encoding="utf-8")

    assert text.startswith("# Eval results")
    assert "Model comparison" not in text
    assert "| Model identifier | `gpt-4o-mini` |" in text


def test_compare_model_id_equal_to_the_model_id_collapses_to_one_model(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 18.5 permits at most 2 identifiers, and the same identifier twice is
    # one model. Asserted end to end, not just on the config property.
    config = comparing(tmp_config, PRIMARY_MODEL)
    assert config.comparison_model_ids == (PRIMARY_MODEL,)

    comparison = compare(
        config,
        suite_of(2),
        eval_paths,
        {PRIMARY_MODEL: client_for(PRIMARY_MODEL)},
        write_report=True,
    )

    assert comparison.model_ids == (PRIMARY_MODEL,)
    text = eval_paths.report_file.read_text(encoding="utf-8")
    assert "Model comparison" not in text
    # One model keeps the trace path of Req 15.3 exactly: no model segment.
    assert (eval_paths.trace_dir / "task-01.jsonl").is_file()


# --------------------------------------------------------------------------- #
# 2. Configured: every task, once per identifier (Req 18.5)
# --------------------------------------------------------------------------- #


def test_each_of_the_two_models_runs_every_task_of_the_shipped_suite(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 18.5 names the whole suite -- all 12 tasks -- once per identifier.
    # The shipped suite is loaded with its composition asserted, so "12" here is
    # the published twelve and not a number this test made up.
    shipped = load_suite(require_composition=True)
    assert len(shipped) == 12

    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    comparison = compare(comparing(tmp_config), shipped, eval_paths, clients)

    assert comparison.model_ids == (PRIMARY_MODEL, SECOND_MODEL)
    for report in comparison.reports:
        assert report.suite_size == 12
        assert tuple(result.task_id for result in report.results) == shipped.ids()
        assert report.executed_count == 12


def test_both_models_are_driven_through_the_injected_factory(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # One scripted client per model, each called once per task: this is what
    # makes "ran against each identifier" an assertion about what executed.
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    comparison = compare(comparing(tmp_config), suite_of(4), eval_paths, clients)

    assert clients[PRIMARY_MODEL].call_count == 4
    assert clients[SECOND_MODEL].call_count == 4
    assert [report.model_id for report in comparison.reports] == [
        PRIMARY_MODEL,
        SECOND_MODEL,
    ]


# --------------------------------------------------------------------------- #
# 3. What the report states per identifier (Req 18.5)
# --------------------------------------------------------------------------- #


def test_report_states_identifier_passed_count_pass_rate_and_mean_cost_per_model(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 18.5, the whole of it: for each identifier, that identifier, the
    # passed count, the pass rate as the count over the suite's task count, and
    # the mean dollar cost per task. The second model answers with empty text,
    # so it fails every task and the two rows must differ.
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: FakeModelClient(
            [make_reply("", usage=DIME)], model_id=SECOND_MODEL, repeat_last=True
        ),
    }
    comparison = compare(
        comparing(tmp_config), suite_of(4), eval_paths, clients, write_report=True
    )

    assert comparison.summaries == (
        ModelSummary(PRIMARY_MODEL, 4, 4, 100.0, pytest.approx(0.10)),
        ModelSummary(SECOND_MODEL, 0, 4, 0.0, pytest.approx(0.20)),
    )

    text = eval_paths.report_file.read_text(encoding="utf-8")
    assert "## Model comparison" in text
    # Identifier, passed count, pass rate over the suite total, mean cost.
    assert f"| `{PRIMARY_MODEL}` | 4 of 4 | 100.0% | 0.1000 |" in text
    assert f"| `{SECOND_MODEL}` | 0 of 4 | 0.0% | 0.2000 |" in text
    # And each model keeps its own full section underneath.
    assert f"## Results for model `{PRIMARY_MODEL}`" in text
    assert f"## Results for model `{SECOND_MODEL}`" in text
    assert text.count("### Run metadata") == 2
    assert text.count("### Per-task results") == 2
    assert text.count("### Aggregates") == 2


def test_pass_rate_denominator_is_the_suite_task_count(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 18.5 divides by the number of tasks in the suite, which for the
    # shipped suite is 12. A 12-task suite where one task fails per model is the
    # readable form of that: 11 of 12, 91.7%.
    failing = suite_of(12).tasks
    eval_suite = EvalSuite(
        version=SUITE_VERSION,
        tasks=failing[:-1]
        + (
            dataclasses.replace(
                failing[-1],
                checks=(check("impossible", "answer_contains", text="never-emitted"),),
            ),
        ),
        source_path=Path("tests") / "in-memory-suite.json",
    )
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    comparison = compare(
        comparing(tmp_config), eval_suite, eval_paths, clients, write_report=True
    )

    text = eval_paths.report_file.read_text(encoding="utf-8")
    for summary in comparison.summaries:
        assert summary.passed_count == 11
        assert summary.suite_size == 12
        assert summary.pass_rate == pytest.approx(11 / 12 * 100.0)
        assert f"| `{summary.model_id}` | 11 of 12 | 91.7% |" in text


def test_the_comparison_is_printed_as_well_as_written(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # The terminal gets the same four figures, so an operator who never opens
    # the file still sees the comparison (Req 15.9's sibling for Req 18.5).
    stdout = io.StringIO()
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    compare(
        comparing(tmp_config),
        suite_of(2),
        eval_paths,
        clients,
        stdout=stdout,
    )

    printed = stdout.getvalue()
    assert "Model comparison:" in printed
    assert PRIMARY_MODEL in printed
    assert SECOND_MODEL in printed
    assert "2 of 2" in printed
    assert "0.1000" in printed and "0.2000" in printed


# --------------------------------------------------------------------------- #
# 4. Trace files: two sets, no collision (Req 15.3, Req 15.4)
# --------------------------------------------------------------------------- #


def test_model_path_segments_are_safe_and_distinct() -> None:
    assert model_path_segment("vendor/model-b") == "vendor-model-b"
    assert model_path_segment("vendor:model b") == "vendor-model-b"
    assert model_path_segment("///") == "model"
    # Two identifiers that reduce to the same segment still get two directories.
    assert model_path_segments(("a/b", "a:b")) == ("a-b", "a-b-2")


def test_each_model_writes_its_own_trace_files(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 15.3 names a trace file after its task, so two models running the same
    # tasks need two sets or they overwrite each other -- and Req 15.4 would
    # refuse the run outright if the second set reused the first set's paths.
    # The scheme: <trace dir>/<model segment>/<task id>.jsonl.
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    comparison = compare(comparing(tmp_config), suite_of(3), eval_paths, clients)

    written = [
        Path(result.trace_path)
        for report in comparison.reports
        for result in report.results
    ]
    assert len(written) == 6
    assert len({str(path).lower() for path in written}) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)

    for segment, model_id in (("gpt-4o-mini", PRIMARY_MODEL), ("vendor-model-b", SECOND_MODEL)):
        model_dir = eval_paths.trace_dir / segment
        assert sorted(path.name for path in model_dir.iterdir()) == [
            "task-01.jsonl",
            "task-02.jsonl",
            "task-03.jsonl",
        ]
        assert comparison.report_for(model_id) is not None


# --------------------------------------------------------------------------- #
# 5. The cost ceiling bounds the invocation, not the model (Req 15.8)
# --------------------------------------------------------------------------- #


def test_cost_ceiling_bounds_the_whole_invocation(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Each call costs $0.10 on the primary model. With a $0.25 ceiling the
    # primary model may start a task while under it -- three tasks, $0.30 -- and
    # the second model must then start nothing at all. A ceiling applied per
    # model would have let this invocation spend up to $0.50.
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    comparison = compare(
        comparing(tmp_config),
        suite_of(6),
        eval_paths,
        clients,
        cost_ceiling=0.25,
    )

    first, second = comparison.reports
    assert first.executed_count == 3
    assert second.executed_count == 0
    assert second.ceiling_reached
    assert comparison.ceiling_reached
    # The invocation total, which is the figure the ceiling is about.
    assert comparison.total_cost_usd == pytest.approx(0.30)
    assert comparison.total_cost_usd < 2 * 0.25
    # Nothing of the second model ran, so nothing of it was charged.
    assert second.total_cost_usd == 0.0
    assert clients[SECOND_MODEL].call_count == 0


# --------------------------------------------------------------------------- #
# 6. The verdict (Req 15.15 across models)
# --------------------------------------------------------------------------- #


def test_exit_code_is_zero_only_when_every_model_passed_every_task(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    both_pass = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    passing = compare(comparing(tmp_config), suite_of(2), eval_paths, both_pass)
    assert passing.all_passed
    assert passing.exit_code == exit_codes.FINAL_ANSWER

    one_fails = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: FakeModelClient(
            [make_reply("", usage=DIME)], model_id=SECOND_MODEL, repeat_last=True
        ),
    }
    mixed = compare(comparing(tmp_config), suite_of(2), eval_paths, one_fails)
    assert mixed.reports[0].all_passed
    assert not mixed.all_passed
    assert mixed.exit_code == exit_codes.EVAL_FAILURE


def test_run_eval_takes_the_comparison_path_when_configured(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # The entry point the `eval` command calls dispatches on the configuration
    # alone, so no caller has to know the feature exists.
    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    code = run_eval(
        config=comparing(tmp_config),
        suite=suite_of(2),
        model_client_factory=by_model(clients),
        retriever=FakeRetriever(),
        system_prompt=SYSTEM_PROMPT,
        prices=LITERAL_PRICES,
        paths=eval_paths,
        progress=False,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        started_at=FIXED_START,
    )

    assert code == exit_codes.FINAL_ANSWER
    text = eval_paths.report_file.read_text(encoding="utf-8")
    assert "## Model comparison" in text
    assert clients[SECOND_MODEL].call_count == 2


# --------------------------------------------------------------------------- #
# 7. The hand-maintained section (Req 15.12)
# --------------------------------------------------------------------------- #


def test_comparison_run_keeps_the_hand_maintained_section(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # Req 15.12 is not weakened by the extra section: everything from the
    # `## Before and after` heading down is copied through unchanged.
    kept = (
        f"{PRESERVED_HEADING}\n\n"
        "- **Before:** 75.0% -- model `openai/gpt-oss-120b`, measured 2026-11-04\n"
        "- **After:** 91.7% -- model `openai/gpt-oss-120b`, measured 2026-11-04\n"
    )
    eval_paths.report_file.parent.mkdir(parents=True, exist_ok=True)
    eval_paths.report_file.write_text(f"# Eval results\n\nstale\n\n{kept}", encoding="utf-8")

    clients = {
        PRIMARY_MODEL: client_for(PRIMARY_MODEL),
        SECOND_MODEL: client_for(SECOND_MODEL),
    }
    compare(
        comparing(tmp_config), suite_of(2), eval_paths, clients, write_report=True
    )

    text = eval_paths.report_file.read_text(encoding="utf-8")
    assert "## Model comparison" in text
    assert kept.strip() in text
    assert "75.0%" in text and "91.7%" in text
    assert "stale" not in text


# --------------------------------------------------------------------------- #
# 8. Refusals
# --------------------------------------------------------------------------- #


def test_a_single_client_is_refused_for_a_two_model_comparison(
    tmp_config: Config, eval_paths: EvalPaths
) -> None:
    # One client reports one model identifier for every run it serves, which
    # would make the comparison a comparison of one model with itself.
    with pytest.raises(ValueError, match="model_client_factory"):
        run_comparison(
            comparing(tmp_config),
            suite=suite_of(2),
            model_client=client_for(PRIMARY_MODEL),
            retriever=FakeRetriever(),
            system_prompt=SYSTEM_PROMPT,
            prices=LITERAL_PRICES,
            paths=eval_paths,
            progress=False,
            write_report=False,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )


def test_default_paths_are_untouched_by_this_file() -> None:
    # A guard on the tests themselves: the committed report and trace directory
    # are what build_paths returns by default, and nothing here uses them.
    defaults = build_paths()
    assert defaults.report_file.name == "results.md"
    assert defaults.trace_dir.name == "traces"
