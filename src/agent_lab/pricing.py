"""The price table loader (Req 3.8).

`pricing/prices.json` lives outside the program code so an operator can correct
a price without editing or rebuilding Python, which is what Req 3.8 asks for.
This module reads that file **once** per path, validates it, and hands
`cost.py` a small read-only object.

Three decisions shape it.

**Validation is strict at load time, permissive at lookup time.** A malformed
table is a configuration fault: an entry missing a price, or missing the
`sourced_on` date that tells a reader how stale the number is, raises
:class:`PriceTableError` before the first step runs, in the same breath as a
missing API key (Req 1.8's "before any provider request" ordering). A *model id
that is simply absent* from a well-formed table is a different thing entirely:
Req 3.3 says an unpriced id costs 0.0, still accumulates tokens, and earns one
warning. So :meth:`PriceTable.lookup` returns ``None`` for an unknown id rather
than raising, and `cost.py` turns that ``None`` into the warning.

**:class:`PriceTableError` subclasses :class:`~agent_lab.config.ConfigError`.**
The CLI already catches `ConfigError` in ``main`` and exits 6, so a broken price
table reports like every other configuration fault with no new wiring.

**The load is cached by resolved path.** Req 3.8 wants the table read at
startup, not once per model call; :func:`price_table` memoises on the resolved
path, and tests call ``price_table.cache_clear()`` to drop that memo.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from agent_lab.config import ConfigError

# The repository-relative location the design fixes for the table (design:
# project layout). `parents[2]` walks src/agent_lab/pricing.py -> src/agent_lab
# -> src -> the repository root, which is correct for a source checkout and for
# the editable install the README documents.
PRICES_FILENAME = "prices.json"
PRICES_DIRNAME = "pricing"
DEFAULT_PRICES_PATH = (
    Path(__file__).resolve().parents[2] / PRICES_DIRNAME / PRICES_FILENAME
)

# The three fields Req 3.8 requires of every entry.
INPUT_KEY = "input_per_mtok"
OUTPUT_KEY = "output_per_mtok"
SOURCED_ON_KEY = "sourced_on"
REQUIRED_KEYS: tuple[str, ...] = (INPUT_KEY, OUTPUT_KEY, SOURCED_ON_KEY)

MODELS_KEY = "models"

# `sourced_on` is a plain ISO 8601 calendar date. The regex pins the shape and
# `date.fromisoformat` pins the calendar, so "2026-02-30" is rejected too.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# A price must be a sane amount of money per million tokens. The upper bound is
# deliberately generous: it is there to catch a cents-per-token figure typed
# into a per-million-token field, not to second-guess the provider.
_MAX_PRICE_PER_MTOK = 10_000.0


class PriceTableError(ConfigError):
    """The price table is absent, unparseable, or incomplete.

    A subclass of :class:`~agent_lab.config.ConfigError` so the CLI's single
    configuration-error path prints it and exits 6 (Req 1.8's exit contract).
    """


@dataclass(frozen=True)
class ModelPrice:
    """The validated price row for one model id (Req 3.8)."""

    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    sourced_on: str

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """The Req 3.2 formula for one call, in dollars.

        ``in/1e6 * in_price + out/1e6 * out_price``. Kept here so the one
        division by 1e6 that turns a per-million price into a per-token price
        lives next to the prices themselves; `cost.py` owns the accumulation and
        the rounding for display.
        """
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


class PriceTable:
    """A read-only model id -> :class:`ModelPrice` mapping.

    Construct from a parsed file with :func:`load_price_table` or
    :func:`price_table`; construct directly with an already-validated mapping
    when a test wants a literal table (task 5.4 does exactly that).
    """

    __slots__ = ("_prices", "_source_path")

    def __init__(
        self,
        prices: Mapping[str, ModelPrice],
        *,
        source_path: Path | None = None,
    ) -> None:
        self._prices = MappingProxyType(dict(prices))
        self._source_path = source_path

    @property
    def source_path(self) -> Path | None:
        """The file this table was read from, or None if built in memory."""
        return self._source_path

    def lookup(self, model_id: str) -> ModelPrice | None:
        """The price row for ``model_id``, or ``None`` when it is not priced.

        Returning ``None`` rather than raising is the whole contract Req 3.3
        rests on: an unpriced model id is a tolerated condition that costs 0.0
        and produces one warning, not a run-ending error. Matching is
        case-sensitive and exact, like every other identifier in the project, so
        a table entry can never be claimed by a near-miss id.
        """
        return self._prices.get(model_id)

    def is_priced(self, model_id: str) -> bool:
        """Whether ``model_id`` has a price row."""
        return model_id in self._prices

    def model_ids(self) -> tuple[str, ...]:
        """Every priced model id, in file order."""
        return tuple(self._prices)

    def __len__(self) -> int:
        return len(self._prices)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._prices

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        where = self._source_path if self._source_path is not None else "<in memory>"
        return f"PriceTable({len(self._prices)} models, from {where})"


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #


def _fault(path: Path, detail: str) -> PriceTableError:
    """Build the one error shape: the file, what is wrong, how to fix it."""
    return PriceTableError(
        f"The price table at {path} is not usable: {detail} "
        f"Each entry under {MODELS_KEY!r} must carry {INPUT_KEY!r}, "
        f"{OUTPUT_KEY!r} and a {SOURCED_ON_KEY!r} date in YYYY-MM-DD form, "
        f"copied from the provider's current pricing page."
    )


def _check_price(path: Path, model_id: str, key: str, value: Any) -> float:
    """Validate one price field and return it as a float."""
    # `isinstance(True, int)` is True in Python, so booleans are excluded by
    # hand before the numeric check, exactly as the calculator does.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fault(
            path,
            f"model {model_id!r} has a non-numeric {key!r} ({value!r}).",
        )
    number = float(value)
    if not math.isfinite(number):
        raise _fault(path, f"model {model_id!r} has a non-finite {key!r} ({value!r}).")
    if number < 0:
        raise _fault(path, f"model {model_id!r} has a negative {key!r} ({value!r}).")
    if number > _MAX_PRICE_PER_MTOK:
        raise _fault(
            path,
            f"model {model_id!r} has a {key!r} of {value!r}, which exceeds "
            f"{_MAX_PRICE_PER_MTOK:,.0f} dollars per 1,000,000 tokens -- this is "
            f"usually a per-token price written into a per-million-token field.",
        )
    return number


def _check_sourced_on(path: Path, model_id: str, value: Any) -> str:
    """Validate the Req 3.8 sourcing date and return it unchanged."""
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        raise _fault(
            path,
            f"model {model_id!r} has a {SOURCED_ON_KEY!r} of {value!r}, which is "
            f"not a YYYY-MM-DD date.",
        )
    try:
        date.fromisoformat(value)
    except ValueError:
        raise _fault(
            path,
            f"model {model_id!r} has a {SOURCED_ON_KEY!r} of {value!r}, which is "
            f"not a real calendar date.",
        ) from None
    return value


def _check_entry(path: Path, model_id: str, entry: Any) -> ModelPrice:
    if not isinstance(entry, dict):
        raise _fault(
            path,
            f"the value for model {model_id!r} is {type(entry).__name__}, "
            f"not an object.",
        )
    missing = [key for key in REQUIRED_KEYS if key not in entry]
    if missing:
        raise _fault(
            path,
            f"model {model_id!r} is missing {', '.join(repr(k) for k in missing)}.",
        )
    # Underscore-prefixed keys are comments at every level of this file; any
    # other unexpected key is far more likely to be a misspelling of a required
    # one than a deliberate extension, and a misspelling that loaded silently
    # would mean a price read from the wrong field.
    unexpected = sorted(
        key for key in entry if key not in REQUIRED_KEYS and not key.startswith("_")
    )
    if unexpected:
        raise _fault(
            path,
            f"model {model_id!r} carries unrecognised "
            f"{'keys' if len(unexpected) > 1 else 'key'} "
            f"{', '.join(repr(k) for k in unexpected)}.",
        )
    return ModelPrice(
        model_id=model_id,
        input_per_mtok=_check_price(path, model_id, INPUT_KEY, entry[INPUT_KEY]),
        output_per_mtok=_check_price(path, model_id, OUTPUT_KEY, entry[OUTPUT_KEY]),
        sourced_on=_check_sourced_on(path, model_id, entry[SOURCED_ON_KEY]),
    )


def parse_price_table(document: Any, *, path: Path) -> PriceTable:
    """Validate an already-parsed price document (Req 3.8).

    Split out from :func:`load_price_table` so the validation rules can be
    tested against a literal dict without writing a file, and so a future
    second source for the table reuses the same checks. ``path`` is only used to
    name the origin in error messages.

    :raises PriceTableError: on any structural fault, a missing or non-numeric
        price, or a missing or malformed ``sourced_on`` date.
    """
    if not isinstance(document, dict):
        raise _fault(
            path,
            f"the top level is {type(document).__name__}, not an object.",
        )
    unexpected = sorted(
        key for key in document if key != MODELS_KEY and not key.startswith("_")
    )
    if unexpected:
        raise _fault(
            path,
            f"the top level carries unrecognised "
            f"{'keys' if len(unexpected) > 1 else 'key'} "
            f"{', '.join(repr(k) for k in unexpected)}.",
        )
    if MODELS_KEY not in document:
        raise _fault(path, f"the top-level {MODELS_KEY!r} object is missing.")
    models = document[MODELS_KEY]
    if not isinstance(models, dict):
        raise _fault(
            path,
            f"{MODELS_KEY!r} is {type(models).__name__}, not an object keyed by "
            f"model id.",
        )
    if not models:
        # An empty table would price nothing, so every run would report a cost
        # of 0.0 with a warning -- survivable, but far more likely to be a
        # truncated file than an operator's intent. Better to say so at startup.
        raise _fault(path, f"{MODELS_KEY!r} contains no entries.")

    prices: dict[str, ModelPrice] = {}
    for model_id, entry in models.items():
        if model_id.startswith("_"):
            continue  # a commented-out row
        if not model_id.strip():
            raise _fault(path, "a model id is empty or whitespace-only.")
        prices[model_id] = _check_entry(path, model_id, entry)
    if not prices:
        raise _fault(path, f"{MODELS_KEY!r} contains no model entries.")
    return PriceTable(prices, source_path=path)


def load_price_table(path: Path | str | None = None) -> PriceTable:
    """Read and validate the price table from disk, uncached.

    :param path: the table to read; defaults to :data:`DEFAULT_PRICES_PATH`.
    :raises PriceTableError: if the file is absent, unreadable, not JSON, or
        fails any check in :func:`parse_price_table`.
    """
    resolved = Path(path) if path is not None else DEFAULT_PRICES_PATH
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _fault(
            resolved,
            "the file does not exist. Agent Lab ships this table in the "
            f"{PRICES_DIRNAME}/ directory of the repository; restore it or point "
            "to a copy.",
        ) from None
    except OSError as exc:
        raise _fault(resolved, f"reading it failed ({exc.strerror or exc}).") from None
    except UnicodeDecodeError:
        raise _fault(resolved, "it is not valid UTF-8 text.") from None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _fault(
            resolved,
            f"it is not valid JSON (line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}).",
        ) from None
    return parse_price_table(document, path=resolved)


@lru_cache(maxsize=None)
def _cached(resolved: Path) -> PriceTable:
    return load_price_table(resolved)


def price_table(path: Path | str | None = None) -> PriceTable:
    """The price table for this process, read from disk at most once per path.

    Req 3.8 wants the table read at startup rather than on every model call, and
    the table is immutable once validated, so one read per path is enough. Call
    ``price_table.cache_clear()`` to force a re-read, which is what a test that
    writes a new table needs.
    """
    resolved = Path(path) if path is not None else DEFAULT_PRICES_PATH
    return _cached(resolved)


# Exposed on the public function so callers do not reach for the private cache.
price_table.cache_clear = _cached.cache_clear  # type: ignore[attr-defined]
price_table.cache_info = _cached.cache_info  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_PRICES_PATH",
    "INPUT_KEY",
    "MODELS_KEY",
    "OUTPUT_KEY",
    "PRICES_DIRNAME",
    "PRICES_FILENAME",
    "REQUIRED_KEYS",
    "SOURCED_ON_KEY",
    "ModelPrice",
    "PriceTable",
    "PriceTableError",
    "load_price_table",
    "parse_price_table",
    "price_table",
]
