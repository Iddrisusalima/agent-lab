"""Unit tests for the price table loader (Req 3.8).

Two halves. The first asserts things about the table actually committed in
`pricing/prices.json`, so a sloppy edit to that file fails the suite rather than
quietly corrupting every cost figure in the eval report. The second drives the
validator with malformed tables written to `tmp_path`.
"""

import json

import pytest

from agent_lab.config import ConfigError, DEFAULTS
from agent_lab.pricing import (
    DEFAULT_PRICES_PATH,
    ModelPrice,
    PriceTable,
    PriceTableError,
    load_price_table,
    parse_price_table,
    price_table,
)

# A minimal well-formed document, copied and mutated by the fault tests.
GOOD = {
    "_comment": "ignored",
    "models": {
        "model-a": {
            "input_per_mtok": 0.15,
            "output_per_mtok": 0.6,
            "sourced_on": "2026-10-31",
        }
    },
}


def write(tmp_path, document):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The committed table
# --------------------------------------------------------------------------- #


def test_committed_table_loads_and_validates():
    table = load_price_table()
    assert len(table) >= 1
    assert table.source_path == DEFAULT_PRICES_PATH


def test_committed_table_prices_the_default_model():
    # The model an operator gets with no MODEL_ID set must be priced, or the
    # default run reports a cost of 0.0 with a warning (Req 3.3) instead of a
    # real figure.
    table = load_price_table()
    default_model = DEFAULTS["MODEL_ID"]
    priced = table.lookup(default_model)
    assert priced is not None, f"{default_model} is missing from the price table"
    assert priced.input_per_mtok > 0
    assert priced.output_per_mtok > 0


def test_committed_table_entries_all_carry_a_sourcing_date():
    # Req 3.8: every entry states when its prices were read.
    table = load_price_table()
    for model_id in table.model_ids():
        entry = table.lookup(model_id)
        assert entry is not None
        assert len(entry.sourced_on) == 10


# --------------------------------------------------------------------------- #
# Lookup behaviour
# --------------------------------------------------------------------------- #


def test_unpriced_model_returns_none_rather_than_raising(tmp_path):
    # Req 3.3 rests on this: an absent model id is a tolerated condition the
    # cost tracker turns into one warning, not a run-ending error.
    table = load_price_table(write(tmp_path, GOOD))
    assert table.lookup("no-such-model") is None
    assert table.is_priced("no-such-model") is False
    assert table.is_priced("model-a") is True


def test_lookup_is_case_sensitive(tmp_path):
    table = load_price_table(write(tmp_path, GOOD))
    assert table.lookup("MODEL-A") is None


def test_model_ids_are_reported_in_file_order(tmp_path):
    document = {
        "models": {
            "zeta": GOOD["models"]["model-a"],
            "alpha": GOOD["models"]["model-a"],
        }
    }
    table = load_price_table(write(tmp_path, document))
    assert table.model_ids() == ("zeta", "alpha")


def test_cost_formula_matches_the_requirement():
    # Req 3.2: in/1e6 * in_price + out/1e6 * out_price.
    price = ModelPrice("m", 0.15, 0.60, "2026-10-31")
    assert price.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.75)
    assert price.cost_usd(2_000, 500) == pytest.approx(0.0003 + 0.0003, abs=1e-9)
    assert price.cost_usd(0, 0) == 0.0


def test_a_table_can_be_built_in_memory():
    # Task 5.4 needs a literal table with no file behind it.
    table = PriceTable({"m": ModelPrice("m", 1.0, 2.0, "2026-10-31")})
    assert table.source_path is None
    assert table.lookup("m").output_per_mtok == 2.0


def test_the_table_cannot_be_mutated_through_its_mapping():
    # Req 12.9 in miniature: nothing downstream can rewrite a price.
    table = PriceTable({"m": ModelPrice("m", 1.0, 2.0, "2026-10-31")})
    with pytest.raises(TypeError):
        table._prices["m"] = ModelPrice("m", 0.0, 0.0, "2026-10-31")


# --------------------------------------------------------------------------- #
# Load-time validation
# --------------------------------------------------------------------------- #


def test_absent_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(PriceTableError) as excinfo:
        load_price_table(missing)
    assert str(missing) in str(excinfo.value)


def test_unparseable_json_is_reported_with_a_position(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"models": {', encoding="utf-8")
    with pytest.raises(PriceTableError) as excinfo:
        load_price_table(path)
    assert "not valid JSON" in str(excinfo.value)


def test_missing_models_object_is_rejected(tmp_path):
    with pytest.raises(PriceTableError, match="'models' object is missing"):
        load_price_table(write(tmp_path, {"_comment": "only a comment"}))


def test_empty_models_object_is_rejected(tmp_path):
    with pytest.raises(PriceTableError, match="no entries"):
        load_price_table(write(tmp_path, {"models": {}}))


@pytest.mark.parametrize("missing", ["input_per_mtok", "output_per_mtok", "sourced_on"])
def test_an_entry_missing_any_required_field_is_rejected(tmp_path, missing):
    entry = dict(GOOD["models"]["model-a"])
    del entry[missing]
    with pytest.raises(PriceTableError) as excinfo:
        load_price_table(write(tmp_path, {"models": {"model-a": entry}}))
    message = str(excinfo.value)
    # The message must name the offender and the field, so the fix is obvious.
    assert "model-a" in message
    assert missing in message


@pytest.mark.parametrize("bad", ["free", None, True, float("inf"), -0.5, 20_000])
def test_a_non_numeric_or_out_of_range_price_is_rejected(tmp_path, bad):
    entry = dict(GOOD["models"]["model-a"])
    entry["input_per_mtok"] = bad
    document = {"models": {"model-a": entry}}
    path = tmp_path / "prices.json"
    # json.dumps writes Infinity, which json.loads accepts, so the value
    # survives the round trip and reaches the validator.
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PriceTableError, match="input_per_mtok"):
        load_price_table(path)


@pytest.mark.parametrize("bad", ["", "31-10-2026", "2026-10", "2026-02-30", 20261031])
def test_a_malformed_sourcing_date_is_rejected(tmp_path, bad):
    entry = dict(GOOD["models"]["model-a"])
    entry["sourced_on"] = bad
    with pytest.raises(PriceTableError, match="sourced_on"):
        load_price_table(write(tmp_path, {"models": {"model-a": entry}}))


def test_a_misspelled_field_is_rejected_rather_than_ignored(tmp_path):
    entry = dict(GOOD["models"]["model-a"])
    entry["input_per_mtokens"] = 0.15
    with pytest.raises(PriceTableError, match="unrecognised"):
        load_price_table(write(tmp_path, {"models": {"model-a": entry}}))


def test_underscore_keys_are_treated_as_comments(tmp_path):
    entry = dict(GOOD["models"]["model-a"], _note="checked against the invoice")
    document = {"_comment": "x", "models": {"model-a": entry, "_retired": {}}}
    table = load_price_table(write(tmp_path, document))
    assert table.model_ids() == ("model-a",)


def test_a_non_object_entry_is_rejected(tmp_path):
    with pytest.raises(PriceTableError, match="not an object"):
        load_price_table(write(tmp_path, {"models": {"model-a": 0.15}}))


def test_the_error_is_a_configuration_error(tmp_path):
    # So the CLI's single ConfigError path prints it and exits 6.
    assert issubclass(PriceTableError, ConfigError)
    with pytest.raises(ConfigError):
        load_price_table(tmp_path / "nope.json")


def test_parse_validates_without_touching_the_filesystem(tmp_path):
    table = parse_price_table(GOOD, path=tmp_path / "virtual.json")
    assert table.lookup("model-a").input_per_mtok == 0.15


# --------------------------------------------------------------------------- #
# Reading once
# --------------------------------------------------------------------------- #


def test_price_table_reads_the_file_once_per_path(tmp_path):
    path = write(tmp_path, GOOD)
    price_table.cache_clear()
    try:
        first = price_table(path)
        second = price_table(path)
        # Same object, so the second call did not re-read the file (Req 3.8:
        # loaded once at startup).
        assert first is second
        # A later edit is not picked up until the cache is cleared, which is
        # what "loaded once" means.
        write(tmp_path, {"models": {"other": GOOD["models"]["model-a"]}})
        assert price_table(path) is first
        price_table.cache_clear()
        assert price_table(path).model_ids() == ("other",)
    finally:
        price_table.cache_clear()
