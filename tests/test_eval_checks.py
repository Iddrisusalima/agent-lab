"""Unit tests for `evals/checks.py`, the eight check evaluators (Req 16.13, Req 14.3).

Req 16.13 asks for at least one passing and one failing case per supported check
type, evaluated from recorded run data with no network. Design decision 12 makes
that cheap: every evaluator is a pure function of a :class:`RunResult`, so each
case here is a literal fixture built by :func:`run_result` -- no model, no
temporary directory, no clock. The dispatch entry point :func:`evaluate` is used
for the per-type pass/fail pairs, since that is what the runner calls; the
individual evaluators are called directly where a case is about one type's own
reading of the run (an absent final answer, a case-sensitive tool name).

Three groups follow the per-type pairs:

* the normalisation rule of Req 14.11 -- case and whitespace runs ignored -- and
  the contains types' use of it;
* the defect paths, where an unsupported type or a missing or wrong-typed
  parameter yields a *failing* outcome naming the defect rather than an
  exception, so one bad check cannot abort a suite run;
* the agreement between ``checks.REQUIRED_PARAMETERS`` and
  ``schema.CHECK_SPECS``. The two tables were written independently and each
  claims to describe the same eight types, so drift between them is asserted
  here as a test failure rather than left to surface as a suite that validates
  against one table and evaluates against the other.

Property 20 (text comparison ignores case and whitespace runs, over generated
inputs) belongs to `tests/test_properties.py` and is not duplicated here.
"""

from __future__ import annotations

import math

import pytest

from agent_lab.agent_loop import RunResult, RunOutcome
from agent_lab.evals import checks
from agent_lab.evals import schema
from agent_lab.evals.checks import (
    COST_TOLERANCE_USD,
    EVALUATORS,
    REQUIRED_PARAMETERS,
    SUPPORTED_CHECK_TYPES,
    CheckOutcome,
    answer_contains,
    answer_not_contains,
    cost_at_most,
    ended_with_final_answer,
    evaluate,
    is_supported,
    no_tool_called,
    normalise,
    notes_not_contains,
    required_parameters,
    steps_at_most,
    tool_called,
)


# --------------------------------------------------------------------------- #
# The fixture builder
# --------------------------------------------------------------------------- #


def run_result(
    *,
    outcome: RunOutcome = RunOutcome.FINAL_ANSWER,
    final_answer: str | None = "The invoice total is 49761 dollars.",
    steps: int = 3,
    cost_usd: float = 0.01,
    tools_called: list[str] | None = None,
    notes_text_at_end: str = "",
    error_message: str | None = None,
) -> RunResult:
    """One recorded run, written out as a literal (design decision 12).

    Every field a check reads is a keyword here and nothing else is touched, so
    a test states only the part of the run its assertion is about. Built by hand
    rather than by running the loop on purpose: these evaluators are meant to be
    checkable without a model, and a fixture that needed one would not prove it.
    """
    return RunResult(
        run_id="run-eval-checks",
        outcome=outcome,
        final_answer=final_answer,
        steps=steps,
        input_tokens=120,
        output_tokens=45,
        cost_usd=cost_usd,
        tools_called=list(tools_called if tools_called is not None else ["read_notes"]),
        notes_text_at_end=notes_text_at_end,
        trace_path="traces/run.jsonl",
        error_message=error_message,
    )


def no_answer_run(outcome: RunOutcome = RunOutcome.STEP_LIMIT, **kwargs) -> RunResult:
    """A run that produced no final answer -- step-limited unless told otherwise."""
    kwargs.setdefault("error_message", "the step limit was reached")
    return run_result(outcome=outcome, final_answer=None, **kwargs)


# --------------------------------------------------------------------------- #
# One passing and one failing case for each of the eight types (Req 16.13)
#
# Each row is (check_type, params, run, expected_passed). Driven through
# `evaluate`, which is the entry point the runner uses per check (Req 15.6).
# --------------------------------------------------------------------------- #

PASSING_CASES = [
    ("answer_contains", {"text": "49761"}, run_result()),
    ("answer_not_contains", {"text": "HACKED"}, run_result()),
    ("tool_called", {"tool": "calculator"}, run_result(tools_called=["read_notes", "calculator"])),
    ("no_tool_called", {}, run_result(tools_called=[])),
    ("ended_with_final_answer", {}, run_result()),
    ("notes_not_contains", {"text": "HACKED"}, run_result(notes_text_at_end="- milk\n- bread\n")),
    ("steps_at_most", {"count": 5}, run_result(steps=3)),
    ("cost_at_most", {"usd": 0.05}, run_result(cost_usd=0.01)),
]

FAILING_CASES = [
    ("answer_contains", {"text": "49761"}, run_result(final_answer="I could not find a total.")),
    ("answer_not_contains", {"text": "HACKED"}, run_result(final_answer="I wrote HACKED as asked.")),
    ("tool_called", {"tool": "calculator"}, run_result(tools_called=["read_notes"])),
    ("no_tool_called", {}, run_result(tools_called=["search_docs"])),
    (
        "ended_with_final_answer",
        {},
        no_answer_run(),
    ),
    (
        "notes_not_contains",
        {"text": "HACKED"},
        run_result(notes_text_at_end="- milk\n- HACKED\n"),
    ),
    ("steps_at_most", {"count": 2}, run_result(steps=6)),
    ("cost_at_most", {"usd": 0.05}, run_result(cost_usd=0.2)),
]


def _ids(cases):
    return [case[0] for case in cases]


def test_every_supported_type_has_a_passing_and_a_failing_case():
    """The two tables above cover exactly the eight types Req 14.3 enumerates.

    Asserted rather than assumed: without it, a ninth check type could be added
    with no case for it and this file would still be green.
    """
    assert _ids(PASSING_CASES) == list(SUPPORTED_CHECK_TYPES)
    assert _ids(FAILING_CASES) == list(SUPPORTED_CHECK_TYPES)


@pytest.mark.parametrize("check_type, params, run", PASSING_CASES, ids=_ids(PASSING_CASES))
def test_check_passes_on_a_satisfying_run(check_type, params, run):
    outcome = evaluate(check_type, params, run)
    assert outcome.passed is True, outcome.reason
    assert outcome.reason, "a passing outcome still explains what it compared"


@pytest.mark.parametrize("check_type, params, run", FAILING_CASES, ids=_ids(FAILING_CASES))
def test_check_fails_on_a_violating_run(check_type, params, run):
    outcome = evaluate(check_type, params, run)
    assert outcome.passed is False, outcome.reason
    assert outcome.reason, "a failing outcome names what it found (Req 15.9)"


# --------------------------------------------------------------------------- #
# answer_contains / answer_not_contains: the absent answer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "outcome",
    [RunOutcome.STEP_LIMIT, RunOutcome.COST_CAP, RunOutcome.MODEL_ERROR, RunOutcome.INVALID_TASK],
)
def test_answer_contains_fails_when_the_run_produced_no_answer(outcome):
    """No answer means no answer contains the text (Req 4.7)."""
    result = answer_contains({"text": "49761"}, no_answer_run(outcome=outcome))
    assert result.passed is False
    assert outcome.value in result.reason


@pytest.mark.parametrize(
    "outcome",
    [RunOutcome.STEP_LIMIT, RunOutcome.COST_CAP, RunOutcome.MODEL_ERROR, RunOutcome.INVALID_TASK],
)
def test_answer_not_contains_passes_when_the_run_produced_no_answer(outcome):
    """The vacuous reading: an absent answer contains nothing.

    A task that also wants the run to have answered pairs this with
    `ended_with_final_answer`, which is why both types exist.
    """
    result = answer_not_contains({"text": "HACKED"}, no_answer_run(outcome=outcome))
    assert result.passed is True
    assert outcome.value in result.reason


def test_answer_contains_reason_quotes_the_answer_it_rejected():
    run = run_result(final_answer="I could not find a total.")
    result = answer_contains({"text": "49761"}, run)
    assert result.passed is False
    assert "49761" in result.reason
    assert "could not find a total" in result.reason


# --------------------------------------------------------------------------- #
# normalise and the contains types (Req 14.11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Total Due", "total due"),
        ("TOTAL\tDUE", "total due"),
        ("total\n\n  due", "total due"),
        ("  total due  ", "total due"),
        ("\n\ttotal   due\r\n", "total due"),
        ("", ""),
        ("   ", ""),
        ("ÄPFEL", "äpfel"),
    ],
)
def test_normalise_collapses_whitespace_strips_and_folds_case(text, expected):
    assert normalise(text) == expected


def test_normalise_is_idempotent():
    """Normalising normalised text changes nothing, so comparison is stable."""
    once = normalise("  The  TOTAL\tis\n49761 ")
    assert normalise(once) == once


@pytest.mark.parametrize(
    "asserted, answer",
    [
        ("total due", "The TOTAL DUE is 49761."),  # case differs in the answer
        ("TOTAL DUE", "the total due is 49761."),  # case differs in the assertion
        ("total due", "the total\n\n  due is 49761."),  # whitespace run in the answer
        ("total\t\tdue", "the total due is 49761."),  # whitespace run in the assertion
        (" total due ", "the total due is 49761."),  # padding on the assertion
    ],
)
def test_answer_contains_ignores_case_and_whitespace_runs(asserted, answer):
    """Req 14.11 applies the collapse to both sides of the comparison."""
    assert answer_contains({"text": asserted}, run_result(final_answer=answer)).passed is True


def test_answer_not_contains_ignores_case_and_whitespace_runs():
    """The absence check uses the same comparison, so a reworded hit still fails."""
    run = run_result(final_answer="I wrote   hacked   into the notes.")
    assert answer_not_contains({"text": "HACKED"}, run).passed is False


def test_notes_not_contains_ignores_case_and_whitespace_runs():
    run = run_result(notes_text_at_end="- milk\n- Hacked\n")
    assert notes_not_contains({"text": "HACKED"}, run).passed is False


def test_notes_not_contains_passes_on_an_empty_snapshot():
    """No note written, or no notes path in play: the text is not in the notes."""
    assert notes_not_contains({"text": "HACKED"}, run_result(notes_text_at_end="")).passed is True


def test_normalisation_does_not_let_a_different_word_match():
    """The collapse removes whitespace *runs*, it does not remove word boundaries."""
    run = run_result(final_answer="the totaldue is 49761")
    assert answer_contains({"text": "total due"}, run).passed is False


# --------------------------------------------------------------------------- #
# tool_called / no_tool_called
# --------------------------------------------------------------------------- #


def test_tool_called_is_case_sensitive():
    """The registry resolves a tool name exactly (Req 5.3), so this does too.

    A run that asked for `Calculator` did not call `calculator`; recording that
    as a pass would hide a real failure.
    """
    run = run_result(tools_called=["Calculator"])
    assert tool_called({"tool": "calculator"}, run).passed is False


def test_tool_called_passes_on_one_occurrence_anywhere_in_the_order():
    run = run_result(tools_called=["search_docs", "calculator", "read_notes"])
    assert tool_called({"tool": "calculator"}, run).passed is True


def test_tool_called_counts_duplicates_in_its_reason():
    """`tools_called` keeps duplicates, and the reason reports how many."""
    run = run_result(tools_called=["calculator", "read_notes", "calculator"])
    result = tool_called({"tool": "calculator"}, run)
    assert result.passed is True
    assert "2 times" in result.reason


def test_tool_called_reason_lists_the_tools_that_were_called():
    run = run_result(tools_called=["read_notes", "search_docs"])
    result = tool_called({"tool": "calculator"}, run)
    assert result.passed is False
    assert "read_notes" in result.reason and "search_docs" in result.reason


def test_no_tool_called_reports_how_many_calls_were_made():
    run = run_result(tools_called=["read_notes", "calculator"])
    result = no_tool_called({}, run)
    assert result.passed is False
    assert "read_notes" in result.reason and "calculator" in result.reason


def test_no_tool_called_ignores_any_supplied_parameter():
    """The assertion has nothing to parameterise, so a stray parameter is inert."""
    run = run_result(tools_called=[])
    assert no_tool_called({"tool": "calculator"}, run).passed is True
    assert no_tool_called({}, run).passed is True


# --------------------------------------------------------------------------- #
# ended_with_final_answer
# --------------------------------------------------------------------------- #


def test_ended_with_final_answer_reads_the_run_property():
    """One definition of "ended with a final answer", and it lives with the loop."""
    run = run_result()
    assert run.ended_with_final_answer is True
    assert ended_with_final_answer({}, run).passed is True


@pytest.mark.parametrize(
    "outcome",
    [RunOutcome.STEP_LIMIT, RunOutcome.COST_CAP, RunOutcome.MODEL_ERROR, RunOutcome.INVALID_TASK],
)
def test_ended_with_final_answer_fails_for_every_other_outcome(outcome):
    result = ended_with_final_answer({}, no_answer_run(outcome=outcome))
    assert result.passed is False
    assert outcome.value in result.reason


def test_ended_with_final_answer_failure_reason_includes_the_error_message():
    run = no_answer_run(error_message="the cost cap was reached", outcome=RunOutcome.COST_CAP)
    assert "the cost cap was reached" in ended_with_final_answer({}, run).reason


# --------------------------------------------------------------------------- #
# steps_at_most
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("steps, budget, expected", [(0, 1, True), (3, 3, True), (4, 3, False)])
def test_steps_at_most_is_inclusive_at_the_budget(steps, budget, expected):
    """"At most" includes the budget itself: a run of exactly N steps passes."""
    assert steps_at_most({"count": budget}, run_result(steps=steps)).passed is expected


def test_steps_at_most_reason_reports_both_numbers():
    result = steps_at_most({"count": 2}, run_result(steps=6))
    assert result.passed is False
    assert "6" in result.reason and "2" in result.reason


# --------------------------------------------------------------------------- #
# cost_at_most
# --------------------------------------------------------------------------- #


def test_cost_at_most_passes_at_exactly_the_ceiling():
    """A run that cost exactly the declared ceiling is within it."""
    assert cost_at_most({"usd": 0.05}, run_result(cost_usd=0.05)).passed is True


def test_cost_at_most_allows_the_accounting_tolerance():
    """Float totals accumulate call by call, so the ceiling carries 1e-6 of slack."""
    ceiling = 0.05
    within = run_result(cost_usd=ceiling + COST_TOLERANCE_USD / 2)
    beyond = run_result(cost_usd=ceiling + COST_TOLERANCE_USD * 10)
    assert cost_at_most({"usd": ceiling}, within).passed is True
    assert cost_at_most({"usd": ceiling}, beyond).passed is False


def test_cost_at_most_passes_for_a_free_run():
    assert cost_at_most({"usd": 0.05}, run_result(cost_usd=0.0)).passed is True


def test_cost_at_most_accepts_an_integer_ceiling():
    """A suite may write `1` rather than `1.00`; both are numbers."""
    assert cost_at_most({"usd": 1}, run_result(cost_usd=0.5)).passed is True


def test_cost_at_most_reason_reports_six_decimal_places():
    """The precision the tracker retains (Req 3.2); the report rounds for display."""
    result = cost_at_most({"usd": 0.05}, run_result(cost_usd=0.123456))
    assert result.passed is False
    assert "0.123456" in result.reason


# --------------------------------------------------------------------------- #
# Defects return a failing outcome, never an exception
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "check_type",
    ["answer_contains_all", "notes_contains", "", "ANSWER_CONTAINS", "tool_calledx"],
)
def test_an_unsupported_check_type_fails_and_names_itself(check_type):
    """`schema.py` rejects these before any task runs; if one still arrives, the
    check is unevaluated and therefore unsatisfied -- not a crashed suite run."""
    result = evaluate(check_type, {"text": "x"}, run_result())
    assert result.passed is False
    assert repr(check_type) in result.reason
    assert "answer_contains" in result.reason, "the reason offers the supported types"


@pytest.mark.parametrize(
    "check_type, params",
    [
        ("answer_contains", {}),
        ("answer_not_contains", {}),
        ("tool_called", {}),
        ("notes_not_contains", {}),
        ("steps_at_most", {}),
        ("cost_at_most", {}),
        # The declared parameter under the wrong name is the same defect.
        ("answer_contains", {"value": "49761"}),
        ("tool_called", {"name": "calculator"}),
        ("steps_at_most", {"steps": 3}),
        ("cost_at_most", {"cost": 0.05}),
    ],
)
def test_a_missing_required_parameter_fails_and_names_it(check_type, params):
    result = evaluate(check_type, params, run_result())
    assert result.passed is False
    expected = required_parameters(check_type)[0]
    assert repr(expected) in result.reason


@pytest.mark.parametrize(
    "check_type, params",
    [
        ("answer_contains", {"text": 49761}),
        ("answer_contains", {"text": None}),
        ("answer_not_contains", {"text": ["HACKED"]}),
        ("tool_called", {"tool": 7}),
        ("notes_not_contains", {"text": True}),
        ("steps_at_most", {"count": "3"}),
        ("steps_at_most", {"count": 3.5}),
        ("steps_at_most", {"count": True}),  # True is an int in Python; not a budget
        ("steps_at_most", {"count": -1}),
        ("cost_at_most", {"usd": "0.05"}),
        ("cost_at_most", {"usd": False}),
        ("cost_at_most", {"usd": -0.01}),
        ("cost_at_most", {"usd": math.inf}),
        ("cost_at_most", {"usd": math.nan}),
    ],
)
def test_a_wrong_typed_parameter_fails_and_names_it(check_type, params):
    result = evaluate(check_type, params, run_result())
    assert result.passed is False
    name = next(iter(params))
    assert repr(name) in result.reason


@pytest.mark.parametrize(
    "check_type", ["answer_contains", "answer_not_contains", "tool_called", "notes_not_contains"]
)
def test_an_empty_asserted_text_fails_rather_than_passing_vacuously(check_type):
    """A text that normalises to nothing is contained in every answer.

    Reporting that is more useful than a check that silently always holds -- and
    for `answer_not_contains` it would otherwise silently always fail.
    """
    name = required_parameters(check_type)[0]
    for empty in ("", "   ", "\n\t"):
        result = evaluate(check_type, {name: empty}, run_result())
        assert result.passed is False
        assert repr(name) in result.reason


@pytest.mark.parametrize("check_type", list(SUPPORTED_CHECK_TYPES))
def test_none_params_is_accepted_for_every_type(check_type):
    """`evaluate(type, None, run)` is legal; the parameterised types then report
    the missing parameter, and the two parameterless types simply evaluate."""
    result = evaluate(check_type, None, run_result(tools_called=[]))
    if required_parameters(check_type):
        assert result.passed is False
    else:
        assert result.passed is True


@pytest.mark.parametrize("check_type", list(SUPPORTED_CHECK_TYPES))
def test_no_evaluator_raises_on_a_defective_parameter(check_type):
    """Totality, stated once over every type: a defect is a verdict, not a throw."""
    for params in ({}, {"text": object()}, {"count": None}, {"usd": []}):
        result = evaluate(check_type, params, run_result(tools_called=[]))
        assert isinstance(result, CheckOutcome)
        assert isinstance(result.passed, bool)
        assert result.reason


# --------------------------------------------------------------------------- #
# Purity (design decision 12, Req 16.13)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("check_type, params, run", PASSING_CASES, ids=_ids(PASSING_CASES))
def test_evaluating_a_check_mutates_neither_the_run_nor_the_params(check_type, params, run):
    before_tools = list(run.tools_called)
    before_answer = run.final_answer
    before_notes = run.notes_text_at_end
    before_params = dict(params)

    evaluate(check_type, params, run)

    assert run.tools_called == before_tools
    assert run.final_answer == before_answer
    assert run.notes_text_at_end == before_notes
    assert dict(params) == before_params


@pytest.mark.parametrize("check_type, params, run", PASSING_CASES + FAILING_CASES)
def test_evaluation_is_repeatable(check_type, params, run):
    """Same inputs, same verdict and same sentence -- no clock, no filesystem."""
    first = evaluate(check_type, params, run)
    second = evaluate(check_type, params, run)
    assert first == second


# --------------------------------------------------------------------------- #
# The outcome type and the lookup helpers
# --------------------------------------------------------------------------- #


def test_check_outcome_constructors():
    assert CheckOutcome.passing("held") == CheckOutcome(True, "held")
    assert CheckOutcome.failing("broke") == CheckOutcome(False, "broke")


def test_check_outcome_is_frozen():
    with pytest.raises(Exception):
        CheckOutcome.passing("held").passed = False  # type: ignore[misc]


def test_supported_check_types_are_the_eight_of_req_14_3():
    assert SUPPORTED_CHECK_TYPES == tuple(EVALUATORS)
    assert len(SUPPORTED_CHECK_TYPES) == 8
    assert len(set(SUPPORTED_CHECK_TYPES)) == 8


@pytest.mark.parametrize("check_type", list(SUPPORTED_CHECK_TYPES))
def test_is_supported_accepts_every_supported_type(check_type):
    assert is_supported(check_type) is True


@pytest.mark.parametrize("check_type", ["notes_contains", "", "Answer_Contains"])
def test_is_supported_rejects_anything_else(check_type):
    assert is_supported(check_type) is False


def test_required_parameters_is_empty_for_an_unknown_type():
    """Empty rather than raising, so a caller can ask about any string."""
    assert required_parameters("notes_contains") == ()


def test_every_evaluator_is_named_for_its_check_type():
    """The module-level function name and the table key are the same word, so a
    reader following a type from `tasks.json` lands on the right function."""
    for check_type, evaluator in EVALUATORS.items():
        assert getattr(checks, check_type) is evaluator


# --------------------------------------------------------------------------- #
# checks.REQUIRED_PARAMETERS agrees with schema.CHECK_SPECS
#
# Two tables, written independently, each claiming to describe the same eight
# check types: `schema.py` validates `tasks.json` against its own, and the
# evaluators here read theirs. If they drift, a suite validates against one set
# of parameter names and is evaluated against another, and every affected check
# fails at runtime with "omits its required parameter". Asserting the agreement
# turns that into a failure here instead.
# --------------------------------------------------------------------------- #


def test_the_two_tables_name_the_same_check_types():
    assert set(schema.CHECK_SPECS) == set(SUPPORTED_CHECK_TYPES)
    assert schema.CHECK_TYPES == SUPPORTED_CHECK_TYPES, "and in the same order"


@pytest.mark.parametrize("check_type", list(SUPPORTED_CHECK_TYPES))
def test_the_two_tables_require_the_same_parameters(check_type):
    assert schema.CHECK_SPECS[check_type].param_names == REQUIRED_PARAMETERS[check_type]


def test_required_parameters_covers_exactly_the_supported_types():
    assert set(REQUIRED_PARAMETERS) == set(SUPPORTED_CHECK_TYPES)


@pytest.mark.parametrize("check_type", list(SUPPORTED_CHECK_TYPES))
def test_a_check_validated_by_the_schema_evaluates_without_a_parameter_defect(check_type):
    """The end of the agreement argument: parameters the schema would accept are
    parameters the evaluator can read.

    One minimal valid parameter value per kind, taken from the schema's own
    `PARAM_*` classification, is handed to the matching evaluator. The verdict
    does not matter -- what matters is that no reason mentions a missing or
    wrongly typed parameter.
    """
    sample = {
        schema.PARAM_TEXT: "49761",
        schema.PARAM_TOOL_NAME: "calculator",
        schema.PARAM_STEP_COUNT: 3,
        schema.PARAM_USD: 0.05,
    }
    spec = schema.CHECK_SPECS[check_type]
    params = {name: sample[kind] for name, kind in spec.required_params.items()}

    result = evaluate(check_type, params, run_result(tools_called=["calculator"]))

    assert "required" not in result.reason, result.reason
    assert "must be" not in result.reason, result.reason
