"""Unit tests for `config.py` (Req 1.5-1.10, Req 9.11).

The environment is always injected as a plain dict and every filesystem path
lives under `tmp_path`, so no test reads `os.environ` or touches the repository.
Range exhaustiveness is Property 19's job (task 1.6); these tests pin the
documented edges, the error message contents, and the ordering guarantees.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest

from agent_lab.config import (
    DECIMAL_RANGES,
    DEFAULTS,
    EXAMPLE_FILE,
    INT_RANGES,
    Config,
    ConfigError,
    api_key_var_for,
    load_config,
    probe_access,
)

KEY_VAR = "OPENAI_API_KEY"
SECRET = "sk-test-not-a-real-key"

REPO_ROOT = Path(__file__).resolve().parents[1]


def env(**overrides: str) -> dict[str, str]:
    """An environment holding a credential plus the given overrides."""
    base = {KEY_VAR: SECRET}
    base.update(overrides)
    return base


def paths_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    """An environment whose three probed paths all pass their access check."""
    docs = tmp_path / "corpus"
    docs.mkdir()
    (docs / "a.md").write_text("hello", encoding="utf-8")
    values = {
        "DOCS_DIR": str(docs),
        "NOTES_FILE": str(tmp_path / "notes.md"),
        "TRACE_FILE": str(tmp_path / "traces" / "run.jsonl"),
    }
    values.update(overrides)
    return env(**values)


# --------------------------------------------------------------------------- #
# Req 1.8: the provider credential
# --------------------------------------------------------------------------- #


def test_absent_api_key_raises_naming_the_variable_and_the_example_file():
    with pytest.raises(ConfigError) as excinfo:
        load_config({}, probe=False)
    message = str(excinfo.value)
    assert KEY_VAR in message
    assert EXAMPLE_FILE in message


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n", "   \n "])
def test_blank_api_key_is_treated_as_absent(blank: str):
    # Req 1.8 draws no distinction between "not set" and "set to nothing".
    with pytest.raises(ConfigError) as excinfo:
        load_config({KEY_VAR: blank}, probe=False)
    assert KEY_VAR in str(excinfo.value)


def test_api_key_has_no_default_entry():
    # A default for the credential would turn a fatal fault into a silent one.
    assert KEY_VAR not in DEFAULTS


def test_api_key_is_stripped_and_its_variable_name_is_carried():
    config = load_config(env(**{KEY_VAR: f"  {SECRET}  "}), probe=False)
    assert config.api_key == SECRET
    assert config.api_key_var == KEY_VAR


def test_api_key_variable_name_follows_the_provider():
    assert api_key_var_for("openai") == "OPENAI_API_KEY"


def test_missing_key_is_fatal_before_the_access_probe_runs(tmp_path: Path):
    # Req 1.8: the run stops before any provider request and before any trace
    # line. The probe is the first thing after validation that touches the
    # filesystem, so an empty stream is the observable form of "nothing
    # happened yet".
    stream = io.StringIO()
    bad_paths = {
        "DOCS_DIR": str(tmp_path / "missing"),
        "NOTES_FILE": str(tmp_path / "missing" / "notes.md"),
    }
    with pytest.raises(ConfigError):
        load_config(bad_paths, probe=True, stream=stream)
    assert stream.getvalue() == ""


def test_out_of_range_value_is_fatal_before_the_access_probe_runs(tmp_path: Path):
    stream = io.StringIO()
    bad = paths_env(tmp_path, MAX_STEPS="0", DOCS_DIR=str(tmp_path / "missing"))
    with pytest.raises(ConfigError):
        load_config(bad, probe=True, stream=stream)
    assert stream.getvalue() == ""


# --------------------------------------------------------------------------- #
# Req 1.9: range validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["0", "-1", "51", "100"])
def test_max_steps_outside_its_range_is_rejected(value: str):
    with pytest.raises(ConfigError) as excinfo:
        load_config(env(MAX_STEPS=value), probe=False)
    message = str(excinfo.value)
    # Req 1.9: the variable, the received value, and the accepted range.
    assert "MAX_STEPS" in message
    assert repr(value) in message
    assert "1" in message and "50" in message


@pytest.mark.parametrize("value", ["abc", "3.5", "8 steps", "eight", "1e1", "٨"])
def test_unparseable_max_steps_is_rejected(value: str):
    # "1e1" is ten but not an integer literal, and "٨" is an Arabic-Indic eight
    # that `int()` would happily accept.
    with pytest.raises(ConfigError) as excinfo:
        load_config(env(MAX_STEPS=value), probe=False)
    assert "MAX_STEPS" in str(excinfo.value)


@pytest.mark.parametrize("value, expected", [("1", 1), ("50", 50), ("+8", 8)])
def test_max_steps_accepts_its_documented_edges(value: str, expected: int):
    assert load_config(env(MAX_STEPS=value), probe=False).max_steps == expected


@pytest.mark.parametrize("value", ["0", "0.009", "100.01", "-0.25", "1000"])
def test_cost_cap_outside_its_range_is_rejected(value: str):
    with pytest.raises(ConfigError) as excinfo:
        load_config(env(COST_CAP=value), probe=False)
    message = str(excinfo.value)
    assert "COST_CAP" in message
    assert repr(value) in message
    assert "0.01" in message and "100.00" in message


@pytest.mark.parametrize("value", ["nan", "NaN", "Infinity", "inf", "1e999", "$0.25"])
def test_unparseable_cost_cap_is_rejected(value: str):
    # `Decimal` accepts "nan" and "Infinity"; the documented range does not.
    with pytest.raises(ConfigError) as excinfo:
        load_config(env(COST_CAP=value), probe=False)
    assert "COST_CAP" in str(excinfo.value)


@pytest.mark.parametrize(
    "value, expected", [("0.01", 0.01), ("100.00", 100.0), ("0.5", 0.5)]
)
def test_cost_cap_accepts_its_documented_edges(value: str, expected: float):
    assert load_config(env(COST_CAP=value), probe=False).cost_cap == expected


@pytest.mark.parametrize("name, bounds", sorted(INT_RANGES.items()))
def test_every_bounded_integer_rejects_both_sides_of_its_range(
    name: str, bounds: tuple[int, int]
):
    lo, hi = bounds
    for bad in (str(lo - 1), str(hi + 1)):
        with pytest.raises(ConfigError) as excinfo:
            load_config(env(**{name: bad}), probe=False)
        message = str(excinfo.value)
        assert name in message
        assert repr(bad) in message
        assert str(lo) in message and str(hi) in message


@pytest.mark.parametrize("name", sorted(DECIMAL_RANGES))
def test_every_bounded_decimal_rejects_text(name: str):
    with pytest.raises(ConfigError) as excinfo:
        load_config(env(**{name: "not-a-number"}), probe=False)
    assert name in str(excinfo.value)


def test_chunk_overlap_upper_bound_follows_chunk_size():
    # Req 7.2: 0 to CHUNK_SIZE-1, so the bound is only meaningful alongside the
    # size that produced it.
    accepted = load_config(env(CHUNK_SIZE="400", CHUNK_OVERLAP="399"), probe=False)
    assert accepted.chunk_overlap == 399

    with pytest.raises(ConfigError) as excinfo:
        load_config(env(CHUNK_SIZE="400", CHUNK_OVERLAP="400"), probe=False)
    message = str(excinfo.value)
    assert "CHUNK_OVERLAP" in message
    assert "400" in message and "399" in message


def test_negative_chunk_overlap_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        load_config(env(CHUNK_OVERLAP="-1"), probe=False)
    assert "CHUNK_OVERLAP" in str(excinfo.value)


def test_unknown_provider_is_rejected_naming_the_variable():
    # Req 2.9, raised from the same place and the same type as the rest.
    with pytest.raises(ConfigError) as excinfo:
        load_config({"PROVIDER": "anthropic", KEY_VAR: SECRET}, probe=False)
    message = str(excinfo.value)
    assert "PROVIDER" in message
    assert "'anthropic'" in message
    assert "openai" in message


def test_provider_matching_is_case_sensitive():
    with pytest.raises(ConfigError):
        load_config({"PROVIDER": "OpenAI", KEY_VAR: SECRET}, probe=False)


# --------------------------------------------------------------------------- #
# Req 1.6, Req 1.7: reading, and defaults for absent and empty values
# --------------------------------------------------------------------------- #


def test_defaults_are_applied_when_only_the_credential_is_set():
    config = load_config({KEY_VAR: SECRET}, probe=False)
    # Req 1.7 names these two values explicitly.
    assert config.max_steps == 8
    assert config.cost_cap == 0.25
    assert config.provider == DEFAULTS["PROVIDER"]
    assert config.model_id == DEFAULTS["MODEL_ID"]
    assert config.docs_dir == Path(DEFAULTS["DOCS_DIR"])
    assert config.notes_file == Path(DEFAULTS["NOTES_FILE"])
    assert config.trace_file == Path(DEFAULTS["TRACE_FILE"])
    assert config.chroma_dir == Path(DEFAULTS["CHROMA_DIR"])
    assert config.tool_timeout_seconds == float(DEFAULTS["TOOL_TIMEOUT_SECONDS"])
    assert config.tool_result_max_chars == int(DEFAULTS["TOOL_RESULT_MAX_CHARS"])
    assert config.notes_read_budget == int(DEFAULTS["NOTES_READ_BUDGET"])
    assert config.display_max_chars == int(DEFAULTS["DISPLAY_MAX_CHARS"])
    assert config.chunk_size == int(DEFAULTS["CHUNK_SIZE"])
    assert config.chunk_overlap == int(DEFAULTS["CHUNK_OVERLAP"])
    assert config.eval_cost_ceiling == float(DEFAULTS["EVAL_COST_CEILING"])


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_empty_and_whitespace_values_fall_back_to_the_default(blank: str):
    # Req 1.7 treats "absent" and "holds an empty string" identically.
    blanked = {name: blank for name in DEFAULTS}
    config = load_config(env(**blanked), probe=False)
    reference = load_config({KEY_VAR: SECRET}, probe=False)
    assert config == reference


def test_supplied_values_override_every_default():
    config = load_config(
        env(
            PROVIDER="openai",
            MODEL_ID="gpt-4o",
            DOCS_DIR="docs",
            NOTES_FILE="out/n.md",
            TRACE_FILE="out/t.jsonl",
            CHROMA_DIR="out/.chroma",
            MAX_STEPS="12",
            COST_CAP="1.50",
            TOOL_TIMEOUT_SECONDS="30",
            TOOL_RESULT_MAX_CHARS="900",
            NOTES_READ_BUDGET="1200",
            DISPLAY_MAX_CHARS="120",
            CHUNK_SIZE="500",
            CHUNK_OVERLAP="50",
            EVAL_COST_CEILING="2.00",
        ),
        probe=False,
    )
    assert (config.model_id, config.max_steps, config.cost_cap) == ("gpt-4o", 12, 1.5)
    assert config.docs_dir == Path("docs")
    assert config.notes_file == Path("out/n.md")
    assert config.trace_file == Path("out/t.jsonl")
    assert config.chroma_dir == Path("out/.chroma")
    assert config.tool_timeout_seconds == 30.0
    assert config.tool_result_max_chars == 900
    assert config.notes_read_budget == 1200
    assert config.display_max_chars == 120
    assert (config.chunk_size, config.chunk_overlap) == (500, 50)
    assert config.eval_cost_ceiling == 2.0


def test_surrounding_whitespace_is_stripped_from_values():
    config = load_config(env(MAX_STEPS=" 12 ", MODEL_ID="  gpt-4o  "), probe=False)
    assert config.max_steps == 12
    assert config.model_id == "gpt-4o"


def test_unrelated_environment_variables_are_ignored():
    config = load_config(env(PATH="/usr/bin", HOME="/home/x"), probe=False)
    assert config == load_config({KEY_VAR: SECRET}, probe=False)


def test_env_example_documents_every_variable_and_its_default():
    # Req 1.5: one row per variable read, each stating its documented default.
    # Task 1.6 parses this file as a range oracle; here it only has to agree
    # with `DEFAULTS` on the values, including the two Req 1.7 names outright.
    text = (REPO_ROOT / EXAMPLE_FILE).read_text(encoding="utf-8")
    for name, default in DEFAULTS.items():
        assert f"# {name} --" in text, f"{name} is undocumented in {EXAMPLE_FILE}"
        assert f"# default: {default} |" in text, f"{name} default differs"
    assert f"# {KEY_VAR} --" in text
    assert "# default: required |" in text
    assert "# default: 8 |" in text  # MAX_STEPS
    assert "# default: 0.25 |" in text  # COST_CAP


# --------------------------------------------------------------------------- #
# Req 1.10: the non-fatal access probe
# --------------------------------------------------------------------------- #


def test_probe_is_silent_when_every_path_passes(tmp_path: Path):
    stream = io.StringIO()
    config = load_config(paths_env(tmp_path), probe=True, stream=stream)
    assert config.access_warnings == ()
    assert stream.getvalue() == ""


def test_unreadable_corpus_directory_warns_and_the_run_continues(tmp_path: Path):
    missing = tmp_path / "no-such-corpus"
    stream = io.StringIO()
    config = load_config(
        paths_env(tmp_path, DOCS_DIR=str(missing)), probe=True, stream=stream
    )
    assert len(config.access_warnings) == 1
    line = config.access_warnings[0]
    # Req 1.10: the variable, the configured path, and read-versus-write.
    assert "DOCS_DIR" in line
    assert str(missing) in line
    assert "read" in line
    assert line in stream.getvalue()
    # Non-fatal: a usable configuration still came back.
    assert config.max_steps == 8


def test_corpus_path_that_is_a_file_warns(tmp_path: Path):
    not_a_dir = tmp_path / "corpus.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    config = load_config(
        paths_env(tmp_path, DOCS_DIR=str(not_a_dir)), probe=True, stream=io.StringIO()
    )
    assert len(config.access_warnings) == 1
    assert "DOCS_DIR" in config.access_warnings[0]


def test_unwritable_notes_path_warns(tmp_path: Path):
    # A directory sitting where the notes file should be: the write can never
    # succeed, on any platform, without needing permission bits.
    occupied = tmp_path / "notes.md"
    occupied.mkdir()
    config = load_config(
        paths_env(tmp_path, NOTES_FILE=str(occupied)), probe=True, stream=io.StringIO()
    )
    assert len(config.access_warnings) == 1
    line = config.access_warnings[0]
    assert "NOTES_FILE" in line
    assert str(occupied) in line
    assert "write" in line


def test_trace_path_under_a_regular_file_warns(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    trace = blocker / "nested" / "run.jsonl"
    config = load_config(
        paths_env(tmp_path, TRACE_FILE=str(trace)), probe=True, stream=io.StringIO()
    )
    assert len(config.access_warnings) == 1
    line = config.access_warnings[0]
    assert "TRACE_FILE" in line
    assert "write" in line


def test_one_warning_line_per_failing_path(tmp_path: Path):
    notes = tmp_path / "notes.md"
    notes.mkdir()
    stream = io.StringIO()
    config = load_config(
        paths_env(
            tmp_path,
            DOCS_DIR=str(tmp_path / "gone"),
            NOTES_FILE=str(notes),
        ),
        probe=True,
        stream=stream,
    )
    assert len(config.access_warnings) == 2
    assert {"DOCS_DIR", "NOTES_FILE"} == {
        var
        for var in ("DOCS_DIR", "NOTES_FILE", "TRACE_FILE")
        if any(var in line for line in config.access_warnings)
    }
    emitted = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(emitted) == 2


def test_absent_notes_and_trace_files_are_not_warnings(tmp_path: Path):
    # Both are created on first write (Req 9.8, Req 11.10), so "does not exist
    # yet" must not be reported as a failure -- a clean clone has neither.
    config = load_config(
        paths_env(tmp_path, TRACE_FILE=str(tmp_path / "a" / "b" / "run.jsonl")),
        probe=True,
        stream=io.StringIO(),
    )
    assert config.access_warnings == ()
    assert not (tmp_path / "a").exists()  # the probe creates nothing


def test_probe_writes_to_stderr_by_default(tmp_path: Path, capsys):
    # Req 13.8: warnings go to stderr, narration to stdout.
    load_config(paths_env(tmp_path, DOCS_DIR=str(tmp_path / "gone")), probe=True)
    captured = capsys.readouterr()
    assert "DOCS_DIR" in captured.err
    assert captured.out == ""


def test_probe_disabled_leaves_no_warnings_and_prints_nothing(tmp_path: Path, capsys):
    config = load_config(
        paths_env(tmp_path, DOCS_DIR=str(tmp_path / "gone")), probe=False
    )
    assert config.access_warnings == ()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_probe_access_is_callable_on_its_own(tmp_path: Path):
    config = load_config(paths_env(tmp_path, DOCS_DIR=str(tmp_path / "gone")), probe=False)
    stream = io.StringIO()
    warnings = probe_access(config, stream)
    assert len(warnings) == 1
    assert "DOCS_DIR" in warnings[0]
    assert warnings[0] in stream.getvalue()


# --------------------------------------------------------------------------- #
# Req 9.11: non-interactive mode is selected only by the explicit option
# --------------------------------------------------------------------------- #


def test_interactive_is_the_default():
    assert load_config({KEY_VAR: SECRET}, probe=False).interactive is True


def test_non_interactive_comes_from_the_explicit_argument():
    config = load_config({KEY_VAR: SECRET}, interactive=False, probe=False)
    assert config.interactive is False


@pytest.mark.parametrize(
    "name",
    [
        "INTERACTIVE",
        "NON_INTERACTIVE",
        "AGENT_LAB_NON_INTERACTIVE",
        "AGENTLAB_YES",
        "YES",
        "CI",
        "TERM",
    ],
)
@pytest.mark.parametrize("value", ["1", "true", "false", "0", "yes", "dumb"])
def test_no_environment_variable_can_select_non_interactive(name: str, value: str):
    # Req 9.11: non-interactive is never an environment setting and never an
    # inference from the terminal. Whatever the environment says, the mode is
    # whatever the caller passed.
    assert load_config(env(**{name: value}), probe=False).interactive is True
    assert (
        load_config(env(**{name: value}), interactive=False, probe=False).interactive
        is False
    )


def test_interactive_is_keyword_only():
    # A positional second argument must not be able to silently become the mode.
    with pytest.raises(TypeError):
        load_config({KEY_VAR: SECRET}, False)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# The config object itself
# --------------------------------------------------------------------------- #


def test_config_is_frozen():
    # Req 12.9: no bound or permission on a loaded config can be rebound, which
    # is why tool result text cannot reach them.
    config = load_config({KEY_VAR: SECRET}, probe=False)
    for field, value in (
        ("max_steps", 999),
        ("cost_cap", 999.0),
        ("interactive", False),
        ("api_key", "other"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(config, field, value)


def test_two_configs_coexist_without_touching_process_state():
    first = load_config(env(MAX_STEPS="3"), probe=False)
    second = load_config(env(MAX_STEPS="9"), probe=False)
    assert (first.max_steps, second.max_steps) == (3, 9)
    assert isinstance(first, Config) and isinstance(second, Config)
