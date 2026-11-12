"""Environment reading, validation, and access probing (Req 1.6 - Req 1.10).

Three decisions shape this module.

**The environment arrives as a parameter.** ``load_config`` takes a
``Mapping[str, str]`` rather than reaching for ``os.environ`` itself, so a test
injects a dictionary instead of mutating process state, and two configurations
can exist side by side in one interpreter (design: Configuration).

**One table of defaults.** ``DEFAULTS`` holds every documented default exactly
once, as the string an operator would have written in ``.env``, and
``.env.example`` documents the same values. Task 1.6 parses ``.env.example`` as
the oracle for a property test, so any drift between the file and this dict is a
test failure rather than a silent inconsistency (Req 1.5, Req 1.7).

**Fatal and non-fatal are kept apart.** A missing credential or an out-of-range
value raises :class:`ConfigError`, which the CLI turns into exit code 6 before
any provider request is issued and before any trace line is appended (Req 1.8,
Req 1.9). A path that cannot be read or written only produces a warning line and
the run continues (Req 1.10).

Interactive mode is the default and is not configurable from the environment:
see :attr:`Config.interactive`.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO, Mapping

# --------------------------------------------------------------------------- #
# The documented default table
# --------------------------------------------------------------------------- #

# Req 1.7: the value applied when a non-credential variable is absent or holds
# an empty string. Values are the literal strings `.env.example` documents, so
# the comparison a test makes between the two is a string comparison with no
# formatting rules in between. The provider credential is deliberately absent
# from this table: it has no default, and its absence is fatal (Req 1.8).
DEFAULTS: dict[str, str] = {
    "PROVIDER": "openai",
    "MODEL_ID": "gpt-4o-mini",
    "DOCS_DIR": "sample_notes",
    "NOTES_FILE": "notes.md",
    "TRACE_FILE": "traces/run.jsonl",
    "CHROMA_DIR": ".chroma",
    "MAX_STEPS": "8",
    "COST_CAP": "0.25",
    "TOOL_TIMEOUT_SECONDS": "10",
    "TOOL_RESULT_MAX_CHARS": "4000",
    "NOTES_READ_BUDGET": "8000",
    "DISPLAY_MAX_CHARS": "500",
    "CHUNK_SIZE": "1000",
    "CHUNK_OVERLAP": "150",
    "EVAL_COST_CEILING": "5.00",
    # Req 18.6: every stretch feature is off unless its variable is set, so each
    # default below *is* the disabled state. A checkout with only a credential
    # set behaves exactly as it did before these four rows existed.
    "STREAM_OUTPUT": "off",
    "WEB_SEARCH": "off",
    "MEMORY_FILE": "",
    "COMPARE_MODEL_ID": "",
}

# Req 2.8, Req 2.9: the provider identifiers with a shipped adapter, compared
# case-sensitively. Adding a provider means adding a module, one entry here, and
# one entry in the model-client factory.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai",)

# Inclusive integer ranges, keyed by variable name (Req 1.9, 5.12, 10.6, 13.3,
# 7.2). CHUNK_OVERLAP is absent because its upper bound depends on CHUNK_SIZE.
INT_RANGES: dict[str, tuple[int, int]] = {
    "MAX_STEPS": (1, 50),
    "TOOL_RESULT_MAX_CHARS": (500, 20_000),
    "NOTES_READ_BUDGET": (500, 50_000),
    "DISPLAY_MAX_CHARS": (80, 10_000),
    "CHUNK_SIZE": (200, 4_000),
}

# Inclusive decimal ranges (Req 1.9, 5.8, 15.8). Bounds are Decimals built from
# the strings `.env.example` documents, so the accepted edge values are exactly
# the documented ones and the message quotes the documented text.
DECIMAL_RANGES: dict[str, tuple[Decimal, Decimal]] = {
    "COST_CAP": (Decimal("0.01"), Decimal("100.00")),
    "TOOL_TIMEOUT_SECONDS": (Decimal("1"), Decimal("60")),
    "EVAL_COST_CEILING": (Decimal("0.01"), Decimal("100.00")),
}

# The two spellings a stretch-feature gate accepts, compared case-sensitively,
# exactly as the provider identifier is (Req 2.8). "ON", "true", "1" and "yes"
# are all rejected with the Req 1.9 message rather than guessed at, so an
# operator never believes a feature is on while it is off.
GATE_VALUES: tuple[str, ...] = ("on", "off")
GATE_ON, GATE_OFF = GATE_VALUES

# Variables whose documented range is an enumeration handled by ``_parse_enum``.
# PROVIDER is deliberately not here: its accepted set is SUPPORTED_PROVIDERS,
# which also selects the credential variable, and resolving it stays in
# ``load_config`` ahead of the credential check (Req 2.9, Req 1.8).
ENUM_RANGES: dict[str, tuple[str, ...]] = {
    "STREAM_OUTPUT": GATE_VALUES,
    "WEB_SEARCH": GATE_VALUES,
}

# The documented file that every configuration error points the operator back to.
EXAMPLE_FILE = ".env.example"

# Accepted literal shapes. These are stricter than ``int()`` and ``Decimal()``,
# both of which accept non-ASCII decimal digits, and ``Decimal`` additionally
# accepts "nan" and "Infinity". The documented ranges admit none of those.
_INT_LITERAL = re.compile(r"^[+-]?[0-9]+$")
_DECIMAL_LITERAL = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)


class ConfigError(Exception):
    """A configuration fault that must stop the run before step 1.

    Raised for a missing or empty provider credential (Req 1.8), a value outside
    its documented range (Req 1.9), and an unresolved provider identifier
    (Req 2.9). The CLI catches this single type in ``main``, prints the message
    to stderr, and exits with ``exit_codes.CONFIGURATION_ERROR``.
    """


@dataclass(frozen=True)
class Config:
    """The validated configuration of one run. Frozen on purpose.

    Nothing reachable from model output or tool result text can rebind a field
    here, which is half of what makes Req 12.9 ("tool result text cannot change
    the run's bounds or permissions") structural rather than prompt-based.
    """

    # --- provider and model ---
    provider: str
    model_id: str
    api_key: str
    # The name of the credential variable, e.g. "OPENAI_API_KEY". Carried so
    # that error messages and the redaction wiring can name it without
    # re-deriving it from the provider id.
    api_key_var: str

    # --- paths ---
    docs_dir: Path
    notes_file: Path
    trace_file: Path
    chroma_dir: Path

    # --- run bounds ---
    max_steps: int
    cost_cap: float

    # --- tool execution limits ---
    tool_timeout_seconds: float
    tool_result_max_chars: int
    notes_read_budget: int

    # --- terminal output ---
    display_max_chars: int

    # --- retrieval chunking ---
    chunk_size: int
    chunk_overlap: int

    # --- evaluation suite ---
    eval_cost_ceiling: float

    # --- optional stretch features (Req 18) ---
    #
    # Each field below holds the raw documented value, and each default is the
    # disabled state (Req 18.6). They are plain fields rather than booleans and
    # paths because `.env.example` is the oracle Property 19 compares against:
    # the field named after a variable holds what that variable's documented
    # range denotes. The properties further down are the readable form the
    # features themselves use.

    # Req 18.2, Req 18.9: "on" streams output tokens as they arrive.
    stream_output: str = GATE_OFF
    # Req 18.3: "on" registers a fifth tool beside the four required ones.
    web_search: str = GATE_OFF
    # Req 18.4, Req 18.8: a file read into the initial conversation state at run
    # start. Empty means no memory file. A configured path that is absent or
    # unreadable is *not* a configuration fault: it warns and the run continues,
    # so nothing here touches the filesystem.
    memory_file: str = ""
    # Req 18.5: a second model identifier for the eval suite to compare against
    # ``model_id``. Empty means one model, which is the required behaviour.
    compare_model_id: str = ""

    # Req 9.11: interactive is the default, and only the explicit
    # `--non-interactive` / `--yes` startup option sets this to False. No
    # environment variable is read for it, and no condition is inferred from the
    # process environment -- in particular, "stdin is not a tty" does not select
    # non-interactive mode. A closed stdin while a prompt is pending is a
    # decline, which `confirmation.InteractiveGate` handles (Req 9.12).
    interactive: bool = True

    # Req 1.10: one line per path that failed its access check. Retained as well
    # as printed so the run summary can repeat them (Req 3.9, Req 13.7).
    access_warnings: tuple[str, ...] = ()

    # --- derived views of the stretch configuration ---------------------- #
    #
    # Computed, not stored: no field is added, so ``dataclasses.fields`` and
    # ``dataclasses.replace`` are unchanged for every existing caller, and the
    # frozen guarantee behind Req 12.9 still covers everything these read.

    @property
    def streaming_enabled(self) -> bool:
        """Req 18.2: whether output tokens are written as they arrive."""
        return self.stream_output == GATE_ON

    @property
    def web_search_enabled(self) -> bool:
        """Req 18.3: whether the optional fifth tool is registered."""
        return self.web_search == GATE_ON

    @property
    def memory_path(self) -> Path | None:
        """Req 18.4: the configured memory file, or None when unconfigured."""
        return Path(self.memory_file) if self.memory_file else None

    @property
    def comparison_model_ids(self) -> tuple[str, ...]:
        """The model identifiers the eval suite runs: at most 2 (Req 18.5).

        ``model_id`` first, then the comparison identifier when one is set and
        differs from it. Unconfigured, this is the single-model tuple the eval
        runner behaves as if it had always used.
        """
        if not self.compare_model_id or self.compare_model_id == self.model_id:
            return (self.model_id,)
        return (self.model_id, self.compare_model_id)


# --------------------------------------------------------------------------- #
# Reading and parsing
# --------------------------------------------------------------------------- #


def _raw(env: Mapping[str, str], name: str) -> str:
    """Return the value for ``name``, with the documented default substituted.

    Req 1.7 treats absent and empty identically: a variable left blank in `.env`
    is a variable the operator did not set. Whitespace is stripped first, so a
    value of " " is blank too.
    """
    value = env.get(name)
    if value is None or not value.strip():
        return DEFAULTS[name]
    return value.strip()


def _range_text_int(lo: int, hi: int) -> str:
    return f"an integer between {lo} and {hi} inclusive"


def _range_text_decimal(lo: Decimal, hi: Decimal) -> str:
    return f"a decimal value between {lo} and {hi} inclusive"


def _reject(name: str, received: str, accepted: str) -> ConfigError:
    """Build the Req 1.9 message: variable, received value, accepted range."""
    return ConfigError(
        f"{name} must be {accepted}, but received {received!r}. "
        f"See {EXAMPLE_FILE} for the accepted range and the default value "
        f"({DEFAULTS[name]})."
    )


def _parse_int(env: Mapping[str, str], name: str) -> int:
    lo, hi = INT_RANGES[name]
    raw = _raw(env, name)
    accepted = _range_text_int(lo, hi)
    if not _INT_LITERAL.match(raw):
        raise _reject(name, raw, accepted)
    value = int(raw)
    if not lo <= value <= hi:
        raise _reject(name, raw, accepted)
    return value


def _parse_decimal(env: Mapping[str, str], name: str) -> float:
    lo, hi = DECIMAL_RANGES[name]
    raw = _raw(env, name)
    accepted = _range_text_decimal(lo, hi)
    if not _DECIMAL_LITERAL.match(raw):
        raise _reject(name, raw, accepted)
    try:
        value = Decimal(raw)
    except InvalidOperation:  # pragma: no cover - the regex already excludes it
        raise _reject(name, raw, accepted) from None
    # The regex excludes "nan" and "inf", but a huge exponent such as "1e999999"
    # still parses, so finiteness is checked rather than assumed.
    if not value.is_finite() or not lo <= value <= hi:
        raise _reject(name, raw, accepted)
    # Compared as a Decimal against the documented bounds, then held as a float
    # for arithmetic; the 0.000001-dollar tolerance in Req 3.7 is what makes
    # float accumulation acceptable for money here.
    return float(value)


def _parse_enum(env: Mapping[str, str], name: str) -> str:
    """Req 1.9: one of the documented spellings, compared case-sensitively.

    Used for the stretch-feature gates (Req 18.6). An unrecognised spelling is
    fatal rather than read as "off", because silently disabling a feature an
    operator asked for is the one failure they cannot see.
    """
    values = ENUM_RANGES[name]
    raw = _raw(env, name)
    if raw not in values:
        raise _reject(name, raw, "one of " + ", ".join(repr(v) for v in values))
    return raw


def _parse_chunk_overlap(env: Mapping[str, str], chunk_size: int) -> int:
    """Req 7.2: overlap is 0 to one character less than the chunk size.

    The upper bound is the only one in the table that depends on another
    variable, so the message states the bound that was actually applied rather
    than a fixed number an operator could not reconcile with their settings.
    """
    name = "CHUNK_OVERLAP"
    raw = _raw(env, name)
    accepted = (
        f"an integer between 0 and {chunk_size - 1} inclusive "
        f"(0 to CHUNK_SIZE-1, with CHUNK_SIZE={chunk_size})"
    )
    if not _INT_LITERAL.match(raw):
        raise _reject(name, raw, accepted)
    value = int(raw)
    if not 0 <= value <= chunk_size - 1:
        raise _reject(name, raw, accepted)
    return value


def api_key_var_for(provider: str) -> str:
    """The credential variable name for a provider id: ``<PROVIDER>_API_KEY``."""
    return f"{provider.upper()}_API_KEY"


# --------------------------------------------------------------------------- #
# The access probe (non-fatal)
# --------------------------------------------------------------------------- #


def _nearest_existing(path: Path) -> Path | None:
    """The closest existing ancestor of ``path``, or None if none exists."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return None


def _check_readable_dir(path: Path) -> str | None:
    """Return a reason the directory is not readable, or None if it is."""
    if not path.exists():
        return "the path does not exist"
    if not path.is_dir():
        return "the path is not a directory"
    if not os.access(path, os.R_OK | os.X_OK):
        return "the directory is not readable"
    try:
        next(iter(os.scandir(path)), None)
    except OSError as exc:
        return f"listing the directory failed: {exc.strerror or exc}"
    return None


def _check_writable_file(path: Path) -> str | None:
    """Return a reason the file path is not writable, or None if it is.

    The file itself need not exist: ``save_note`` and the trace logger both
    create the file and any missing parent directory on first write (Req 9.8,
    Req 11.10). So the check is "could it be created", which means finding the
    nearest existing ancestor and asking whether that is a writable directory.
    """
    if path.exists():
        if path.is_dir():
            return "the path is a directory, not a file"
        if not os.access(path, os.W_OK):
            return "the file is not writable"
        return None
    anchor = _nearest_existing(path.parent if str(path.parent) else path)
    if anchor is None:
        return "no existing parent directory was found for the path"
    if not anchor.is_dir():
        return f"{anchor} is not a directory"
    if not os.access(anchor, os.W_OK | os.X_OK):
        return f"the parent directory {anchor} is not writable"
    return None


def probe_access(config: Config, stream: IO[str] | None = None) -> tuple[str, ...]:
    """Verify corpus read access and notes / trace write access (Req 1.10).

    One warning line per failing path, naming the environment variable, the
    configured path, and whether read or write access failed. The lines go to
    stderr, because they are warnings (Req 13.8), and are returned so the caller
    can repeat them in the run summary. Nothing here is fatal: a missing corpus
    directory is a legitimate state for a run that only uses the calculator, and
    the notes and trace files are created on first write.
    """
    out = stream if stream is not None else sys.stderr
    checks: tuple[tuple[str, Path, str, str | None], ...] = (
        ("DOCS_DIR", config.docs_dir, "read", _check_readable_dir(config.docs_dir)),
        (
            "NOTES_FILE",
            config.notes_file,
            "write",
            _check_writable_file(config.notes_file),
        ),
        (
            "TRACE_FILE",
            config.trace_file,
            "write",
            _check_writable_file(config.trace_file),
        ),
    )
    warnings: list[str] = []
    for var, path, mode, reason in checks:
        if reason is None:
            continue
        # The path is quoted but not ``repr``'d: on Windows ``repr`` doubles
        # every backslash, and Req 1.10 asks for the configured path, not an
        # escaped rendering of it.
        line = (
            f'warning: {var}: {mode} access check failed for path "{path}" '
            f"({reason}). Continuing."
        )
        warnings.append(line)
        print(line, file=out)
    return tuple(warnings)


# --------------------------------------------------------------------------- #
# The loader
# --------------------------------------------------------------------------- #


def load_config(
    env: Mapping[str, str],
    *,
    interactive: bool = True,
    probe: bool = True,
    stream: IO[str] | None = None,
) -> Config:
    """Read, default, and validate the configuration for one run.

    ``env`` is the whole environment as a mapping; the CLI passes ``os.environ``
    and tests pass a dictionary. ``interactive`` is supplied by the CLI from the
    presence of ``--non-interactive`` / ``--yes`` and comes from nowhere else
    (Req 9.11). ``probe`` runs the non-fatal access check (Req 1.10); a test that
    only cares about parsing can switch it off.

    The order below is the order the requirements demand:

    1. read every variable, substituting documented defaults (Req 1.6, Req 1.7);
    2. resolve the provider, which is what names the credential variable;
    3. a missing or empty credential is fatal (Req 1.8);
    4. a value outside its documented range is fatal (Req 1.9);
    5. the access probe warns and the run continues (Req 1.10).

    Steps 3 and 4 raise before anything in this process has contacted a provider
    or opened the trace file, which is the "before any provider request and
    before any trace line" half of Req 1.8.

    :raises ConfigError: on an unknown provider, a missing or empty credential,
        or any value outside its documented range.
    """
    provider = _raw(env, "PROVIDER")
    if provider not in SUPPORTED_PROVIDERS:
        raise _reject(
            "PROVIDER",
            provider,
            "one of " + ", ".join(repr(p) for p in SUPPORTED_PROVIDERS),
        )

    # Req 1.8: the credential has no default and no fallback. Absent, empty, or
    # whitespace-only are the same thing, and all three stop the run here --
    # before the provider adapter is built, so no request is issued, and before
    # the trace logger is constructed, so no line is appended.
    api_key_var = api_key_var_for(provider)
    api_key = (env.get(api_key_var) or "").strip()
    if not api_key:
        raise ConfigError(
            f"{api_key_var} is not set (or is empty). Agent Lab cannot call the "
            f"{provider} provider without it. Copy {EXAMPLE_FILE} to .env and "
            f"set {api_key_var}=<your key>, or export {api_key_var} in your "
            f"shell. No provider request was made and no trace line was written."
        )

    chunk_size = _parse_int(env, "CHUNK_SIZE")
    config = Config(
        provider=provider,
        model_id=_raw(env, "MODEL_ID"),
        api_key=api_key,
        api_key_var=api_key_var,
        docs_dir=Path(_raw(env, "DOCS_DIR")),
        notes_file=Path(_raw(env, "NOTES_FILE")),
        trace_file=Path(_raw(env, "TRACE_FILE")),
        chroma_dir=Path(_raw(env, "CHROMA_DIR")),
        max_steps=_parse_int(env, "MAX_STEPS"),
        cost_cap=_parse_decimal(env, "COST_CAP"),
        tool_timeout_seconds=_parse_decimal(env, "TOOL_TIMEOUT_SECONDS"),
        tool_result_max_chars=_parse_int(env, "TOOL_RESULT_MAX_CHARS"),
        notes_read_budget=_parse_int(env, "NOTES_READ_BUDGET"),
        display_max_chars=_parse_int(env, "DISPLAY_MAX_CHARS"),
        chunk_size=chunk_size,
        chunk_overlap=_parse_chunk_overlap(env, chunk_size),
        eval_cost_ceiling=_parse_decimal(env, "EVAL_COST_CEILING"),
        # Req 18.6: absent or empty leaves every one of these at its disabled
        # default. Nothing below enables anything on its own, and a configured
        # memory file is not opened here (Req 18.8 warns at run start instead).
        stream_output=_parse_enum(env, "STREAM_OUTPUT"),
        web_search=_parse_enum(env, "WEB_SEARCH"),
        memory_file=_raw(env, "MEMORY_FILE"),
        compare_model_id=_raw(env, "COMPARE_MODEL_ID"),
        interactive=interactive,
    )

    if not probe:
        return config
    return replace(config, access_warnings=probe_access(config, stream))


__all__ = [
    "DEFAULTS",
    "SUPPORTED_PROVIDERS",
    "INT_RANGES",
    "DECIMAL_RANGES",
    "ENUM_RANGES",
    "GATE_VALUES",
    "GATE_ON",
    "GATE_OFF",
    "EXAMPLE_FILE",
    "Config",
    "ConfigError",
    "api_key_var_for",
    "load_config",
    "probe_access",
]
