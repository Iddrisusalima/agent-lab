"""Unit tests for the cost tracker (Req 16.12, Req 3.1 - Req 3.4, Req 3.6).

Every test builds its own literal :class:`PriceTable`. The committed table in
`pricing/prices.json` is deliberately not used here: its numbers change
whenever a provider changes its prices, and a cost assertion baked against
them would start failing for a reason that has nothing to do with the
accounting. `tests/test_pricing.py` is where the committed file is checked.

LITERAL is the table the formula assertions are computed from by hand: one
dollar per million input tokens and two per million output tokens, chosen so
the expected figure of each call can be read off the test.
"""

import pytest

from agent_lab.cost import COST_DECIMALS, CostTracker, Totals
from agent_lab.model_client import Usage
from agent_lab.pricing import ModelPrice, PriceTable

PRICED = "m"
UNPRICED = "mystery-model-9000"

LITERAL = PriceTable(
    {
        PRICED: ModelPrice(PRICED, 1.0, 2.0, "2026-10-31"),
        "cheap": ModelPrice("cheap", 0.15, 0.60, "2026-10-31"),
    }
)


def tracker():
    return CostTracker(prices=LITERAL)


# --------------------------------------------------------------------------- #
# Accumulation across calls (Req 3.1)
# --------------------------------------------------------------------------- #


def test_token_counts_accumulate_across_calls():
    # Req 3.1: both counts of every call enter the run totals.
    t = tracker()
    t.record(Usage(100, 10), PRICED)
    t.record(Usage(250, 40), PRICED)
    t.record(Usage(1, 2), PRICED)

    assert t.steps == 3
    assert t.input_tokens == 351
    assert t.output_tokens == 52
    assert t.total_tokens == 403


def test_totals_snapshot_agrees_with_the_properties():
    t = tracker()
    t.record(Usage(100, 10), PRICED)
    t.record(Usage(5, 5), PRICED)

    snapshot = t.totals()
    assert snapshot == Totals(
        steps=t.steps,
        input_tokens=t.input_tokens,
        output_tokens=t.output_tokens,
        usd=t.total_usd,
    )
    assert snapshot.total_tokens == snapshot.input_tokens + snapshot.output_tokens


def test_a_fresh_tracker_reports_zero_for_everything():
    t = tracker()
    assert t.totals() == Totals()
    assert t.total_usd == 0.0
    assert t.warnings() == []
    assert t.unpriced_model_ids() == ()
    assert t.unavailable_count == 0
    assert t.estimated_count == 0


def test_zero_token_call_still_counts_as_a_step():
    # A step with no tokens is still a completed call, and Req 3.7 ties the
    # step count to the trace line count.
    t = tracker()
    assert t.record(Usage(0, 0), PRICED) == 0.0
    assert t.steps == 1
    assert t.total_tokens == 0


# --------------------------------------------------------------------------- #
# The per-call formula, against the literal table (Req 3.2)
# --------------------------------------------------------------------------- #


def test_per_call_cost_follows_the_requirement_formula():
    # Req 3.2: in/1e6 * in_price + out/1e6 * out_price. At $1.00 per million
    # input and $2.00 per million output: 1,000,000 in = $1.00 and 500,000 out
    # = $1.00.
    t = tracker()
    cost = t.record(Usage(1_000_000, 500_000), PRICED)
    assert cost == pytest.approx(2.0)
    assert t.total_usd == pytest.approx(2.0)


def test_per_call_cost_of_a_small_call():
    t = tracker()
    cost = t.record(Usage(3, 7), PRICED)
    # 3/1e6 * 1.0 + 7/1e6 * 2.0
    assert cost == pytest.approx(0.000003 + 0.000014, abs=1e-12)


def test_sub_microdollar_cost_is_not_rounded_away():
    # Req 3.2 asks for at least 6 decimal places. 10 input tokens at $0.15 per
    # million is $0.0000015, which 6 places would flatten to 0.000002 and a
    # lazier rounding to 0.0.
    t = CostTracker(prices=LITERAL)
    cost = t.record(Usage(10, 0), "cheap")
    assert cost > 0.0
    assert cost == pytest.approx(1.5e-06, rel=1e-9)
    assert round(cost, COST_DECIMALS) == cost


def test_each_model_is_priced_from_its_own_row():
    t = tracker()
    expensive = t.record(Usage(1_000_000, 0), PRICED)
    cheap = t.record(Usage(1_000_000, 0), "cheap")
    assert expensive == pytest.approx(1.0)
    assert cheap == pytest.approx(0.15)


def test_the_run_total_is_the_sum_of_the_per_call_costs():
    # Req 3.7 / Req 16.12: the figure the summary prints equals the sum of the
    # per-call figures the trace carries, because `record` accumulates exactly
    # what it returns.
    t = tracker()
    calls = [(7, 3), (1_234, 567), (0, 1), (99_999, 11), (5, 0)]
    per_call = [t.record(Usage(i, o), PRICED) for i, o in calls]
    assert t.total_usd == pytest.approx(sum(per_call), abs=1e-06)


# --------------------------------------------------------------------------- #
# An unpriced model identifier (Req 3.3)
# --------------------------------------------------------------------------- #


def test_unpriced_model_costs_zero_but_still_accumulates_tokens():
    t = tracker()
    cost = t.record(Usage(500, 200), UNPRICED)

    assert cost == 0.0
    assert t.total_usd == 0.0
    assert t.steps == 1
    assert t.input_tokens == 500
    assert t.output_tokens == 200


def test_unpriced_model_earns_exactly_one_warning_naming_it():
    # Req 3.3: one warning for that identifier "no matter how many calls used
    # that identifier".
    t = tracker()
    for _ in range(4):
        t.record(Usage(10, 10), UNPRICED)

    assert t.unpriced_model_ids() == (UNPRICED,)
    naming = [line for line in t.warnings() if UNPRICED in line]
    assert len(naming) == 1
    assert len(t.warnings()) == 1


def test_two_unpriced_models_each_earn_their_own_warning_in_sorted_order():
    t = tracker()
    t.record(Usage(1, 1), "zeta-model")
    t.record(Usage(1, 1), "alpha-model")
    t.record(Usage(1, 1), "zeta-model")

    assert t.unpriced_model_ids() == ("alpha-model", "zeta-model")
    lines = t.warnings()
    assert len(lines) == 2
    assert "alpha-model" in lines[0]
    assert "zeta-model" in lines[1]


def test_a_priced_call_is_unaffected_by_an_unpriced_one():
    t = tracker()
    t.record(Usage(1_000_000, 0), UNPRICED)
    priced = t.record(Usage(1_000_000, 0), PRICED)

    assert priced == pytest.approx(1.0)
    assert t.total_usd == pytest.approx(1.0)
    assert t.input_tokens == 2_000_000
    assert t.unpriced_model_ids() == (UNPRICED,)


# --------------------------------------------------------------------------- #
# Missing token counts (Req 3.4)
# --------------------------------------------------------------------------- #


def test_absent_usage_records_zero_and_leaves_earlier_totals_untouched():
    # Req 3.4: the missing counts record as 0, the call costs 0.0, and every
    # previously accumulated total is left exactly as it was.
    t = tracker()
    t.record(Usage(1_000, 500), PRICED)
    before = t.totals()

    cost = t.record(None, PRICED)
    after = t.totals()

    assert cost == 0.0
    assert after.input_tokens == before.input_tokens
    assert after.output_tokens == before.output_tokens
    assert after.usd == before.usd
    # The call completed, so it is still a step (Req 3.7 ties steps to trace
    # lines).
    assert after.steps == before.steps + 1
    assert t.unavailable_count == 1


def test_a_partial_usage_block_takes_the_same_path():
    # record_counts is the optional-bearing form: one count arrived, the other
    # did not. Pricing from half the tokens would understate the call, so the
    # whole call takes the Req 3.4 path.
    t = CostTracker(prices=LITERAL)
    assert t.record_counts(None, 7, PRICED) == 0.0
    assert t.record_counts(7, None, PRICED) == 0.0
    assert t.total_tokens == 0
    assert t.total_usd == 0.0
    assert t.steps == 2
    assert t.unavailable_count == 2


def test_unavailable_counts_warn_with_the_number_of_calls_affected():
    # Req 3.4: the summary states that counts were unavailable and how many
    # calls that affected.
    t = tracker()
    t.record(None, PRICED)
    t.record(Usage(10, 10), PRICED)
    t.record(None, PRICED)

    lines = [line for line in t.warnings() if "unavailable" in line]
    assert len(lines) == 1
    assert "2" in lines[0]


def test_estimated_counts_are_counted_and_warned_about():
    # Req 2.3: a cost resting on estimated counts must not read as measured.
    t = tracker()
    t.record(Usage(100, 50, estimated=True), PRICED)
    t.record(Usage(100, 50), PRICED)

    assert t.estimated_count == 1
    assert len([line for line in t.warnings() if "estimated" in line]) == 1


def test_a_non_usage_argument_is_a_caller_defect():
    t = tracker()
    with pytest.raises(ValueError, match="Usage"):
        t.record({"input_tokens": 1}, PRICED)
    with pytest.raises(ValueError, match="model_id"):
        t.record(Usage(1, 1), 123)


# --------------------------------------------------------------------------- #
# Monotonicity across successive reads (Req 3.5, Req 3.6)
# --------------------------------------------------------------------------- #

MONOTONICITY_CALLS = [
    (Usage(120, 30), PRICED),
    (None, PRICED),
    (Usage(0, 0), PRICED),
    (Usage(4_000, 1_000), UNPRICED),
    (Usage(75, 75, estimated=True), "cheap"),
    (Usage(1_000_000, 500_000), PRICED),
]


def test_every_total_is_non_decreasing_across_successive_reads():
    # Req 3.6: for all pairs of successive reads, each field of the later read
    # is >= the same field of the earlier one.
    t = tracker()
    reads = [t.totals()]
    for usage, model_id in MONOTONICITY_CALLS:
        t.record(usage, model_id)
        reads.append(t.totals())

    for earlier, later in zip(reads, reads[1:]):
        assert later.steps >= earlier.steps
        assert later.input_tokens >= earlier.input_tokens
        assert later.output_tokens >= earlier.output_tokens
        assert later.total_tokens >= earlier.total_tokens
        assert later.usd >= earlier.usd


def test_a_held_snapshot_does_not_change_when_a_later_call_is_recorded():
    # Req 3.5/3.6 can only be observed if a read is a frozen snapshot rather
    # than a live view of the tracker.
    t = tracker()
    t.record(Usage(10, 10), PRICED)
    held = t.totals()
    t.record(Usage(1_000_000, 1_000_000), PRICED)

    assert held.steps == 1
    assert held.input_tokens == 10
    assert held.usd < t.total_usd
    assert t.totals() is not held
    with pytest.raises(Exception):
        held.steps = 99  # frozen


def test_a_read_reflects_only_calls_already_completed():
    # Req 3.5: no call still in flight appears in a read.
    t = tracker()
    for index in range(1, 4):
        assert t.totals().steps == index - 1
        t.record(Usage(index, index), PRICED)
        assert t.totals().steps == index
