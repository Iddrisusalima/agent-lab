"""Token and dollar accounting for one run (Req 3).

The loop calls :meth:`CostTracker.record` once per completed model call, before
the next step begins (Req 3.1). The tracker adds the call's token counts to the
run totals, prices the call through the :mod:`agent_lab.pricing` table
(Req 3.2), and returns that one call's cost so the trace record can carry it.
At the end of the run the CLI reads :meth:`CostTracker.totals` and
:meth:`CostTracker.warnings` to print the run summary (Req 3.9).

Four decisions shape this module.

**Everything accumulates; nothing is ever recomputed.** The totals are plain
running sums updated inside `record`, so a read is a snapshot of completed
calls and nothing else (Req 3.5). Because each sum only ever has a non-negative
value added -- `Usage` refuses negative counts at construction, and a price is
validated non-negative at load -- every field is monotonically increasing by
construction rather than by a check at read time (Req 3.6).

**A read returns a frozen snapshot.** :meth:`totals` builds a new
:class:`Totals`, so a caller holding one from before a call and one from after
can compare them. If it returned a live view, the monotonicity Req 3.6 asks
about could not even be observed.

**A tolerated condition warns; it never raises.** An unpriced model id and a
call that arrived without token counts are both conditions Req 3.3 and Req 3.4
require the run to survive and the summary to report. So the only exceptions
raised here are for developer faults (a non-`Usage` argument, a non-string
model id), and a broken price *table* is caught earlier, at construction, where
it reports as the configuration error it is.

**The per-call cost that is returned is the per-call cost that is added.**
`record` rounds the raw product to :data:`COST_DECIMALS` places and accumulates
exactly the value it returns, so the total the tracker reports and the sum of
the per-call costs written to the trace agree to far inside the 0.000001-dollar
tolerance of Req 3.7, however many calls a run makes.

**Signalling "the provider reported no count" (Req 3.4).** ``Usage`` cannot
express a missing count: it enforces non-negative integers, deliberately, so
that a negative number can never reach the totals. A missing count therefore
arrives here as an explicit signal, in one of two equivalent forms:

* ``record(None, model_id)`` -- the call completed but carried no usage at all;
* ``record_counts(None, 7, model_id)`` -- one count arrived and the other did
  not, which is what an adapter has when a provider's usage block is partial.

Either form takes the Req 3.4 path: the missing count is recorded as 0, the
call's cost is 0.0, the previously accumulated token and dollar totals are left
untouched, and the "unavailable token counts" counter rises by one so the
summary can report how many calls were affected. Note that `estimate_usage` in
:mod:`agent_lab.model_client` means the shipped adapter rarely needs this path
-- it estimates counts and flags them (Req 2.3) rather than reporting none --
but a future adapter that cannot estimate has a way to say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_lab.model_client import Usage
from agent_lab.pricing import ModelPrice, PriceTable, price_table

# Req 3.2 requires at least 6 decimal places of a dollar to be retained. Ten is
# used instead of six because a single cheap call can genuinely cost less than
# one microdollar -- 100 input tokens at $0.15 per million is $0.000015, and a
# 10-token call is $0.0000015 -- and rounding such a call to 6 places would
# throw away most of its cost. Ten places keeps four more digits than required
# while still being a fixed quantum, which is what makes the per-call costs in
# the trace sum to the reported total (Req 3.7).
COST_DECIMALS = 10

# What the summary rounds to for display (Req 3.9). Defined here, next to the
# accounting it describes, and imported by `render.py`; accounting keeps
# COST_DECIMALS internally and only the printed figure is rounded to 4.
DISPLAY_DECIMALS = 4


@dataclass(frozen=True)
class Totals:
    """A snapshot of the run totals at one moment (Req 3.5, Req 3.9).

    ``steps``
        Completed model calls. One per step, which is what makes this field
        comparable to the trace line count (Req 3.7).
    ``input_tokens`` / ``output_tokens``
        The sums of the per-call counts.
    ``usd``
        The sum of the per-call costs, each rounded to :data:`COST_DECIMALS`.

    ``total_tokens`` is a property rather than a field so a snapshot cannot be
    constructed in which it disagrees with its two parts; the run summary prints
    it alongside them (Req 3.9).

    Frozen, because its purpose is to be held and compared against a later
    snapshot.
    """

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Input plus output, the fourth token figure in the summary (Req 3.9)."""
        return self.input_tokens + self.output_tokens


class CostTracker:
    """Accumulates tokens and dollars across every model call in one run (Req 3).

    One instance per run. The loop constructs it with the model id's price table
    and calls :meth:`record` after each completed call::

        tracker = CostTracker()
        call_cost = tracker.record(reply.usage, client.model_id)
        ...
        if tracker.total_usd >= COST_CAP:   # checked before the next call
            ...

    :param prices: the price table to look model ids up in. Defaults to the
        process-wide table from :func:`agent_lab.pricing.price_table`, read at
        construction so a malformed table fails before the first step rather
        than mid-run; pass a literal :class:`~agent_lab.pricing.PriceTable` in
        tests.
    :raises agent_lab.pricing.PriceTableError: if the default table has to be
        read and is absent or invalid.
    """

    __slots__ = (
        "_prices",
        "_steps",
        "_input_tokens",
        "_output_tokens",
        "_usd",
        "_unpriced",
        "_unavailable_calls",
        "_estimated_calls",
    )

    def __init__(self, prices: PriceTable | None = None) -> None:
        self._prices: PriceTable = prices if prices is not None else price_table()
        self._steps = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._usd = 0.0
        # Req 3.3 asks for one warning per unpriced id "no matter how many calls
        # used that identifier", which is exactly set membership. Rendering
        # sorts it, because set iteration order for strings varies between
        # processes and a run summary should not.
        self._unpriced: set[str] = set()
        self._unavailable_calls = 0
        self._estimated_calls = 0

    # ----------------------------------------------------------------- #
    # Recording (Req 3.1 - Req 3.4)
    # ----------------------------------------------------------------- #

    def record(self, usage: Usage | None, model_id: str) -> float:
        """Record one completed model call and return its cost in dollars.

        Adds the call's input and output token counts to the run totals before
        the next step begins, and prices the call with the input and output
        prices listed for ``model_id`` (Req 3.1, Req 3.2). The returned value is
        the exact figure added to the running total, so a caller that writes it
        to the trace records a per-call cost that sums back to the reported
        total (Req 3.7).

        :param usage: the counts for this call, or ``None`` when the call
            completed without reporting any -- the Req 3.4 path, which records
            0 tokens and 0.0 dollars and leaves the accumulated totals
            untouched.
        :param model_id: the model that served the call. An id absent from the
            price table costs 0.0, still accumulates its tokens, and earns one
            warning (Req 3.3).
        :returns: this call's cost in dollars, 0.0 on either tolerated path.
        :raises ValueError: if ``usage`` is neither a ``Usage`` nor ``None``, or
            ``model_id`` is not a string. Both are caller defects: the loop
            passes a reply's own ``usage`` and its client's own ``model_id``.
        """
        if usage is None:
            return self.record_counts(None, None, model_id)
        if not isinstance(usage, Usage):
            raise ValueError(
                f"usage must be a Usage or None, received {type(usage).__name__}; "
                f"pass None to record a call whose token counts were unavailable"
            )
        return self.record_counts(
            usage.input_tokens,
            usage.output_tokens,
            model_id,
            estimated=usage.estimated,
        )

    def record_counts(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        model_id: str,
        *,
        estimated: bool = False,
    ) -> float:
        """Record one completed call from counts that may individually be absent.

        The optional-bearing form of :meth:`record`, for an adapter holding a
        partial usage block: either count may be ``None`` to mean the provider
        reported no such count. If *either* is ``None`` the whole call takes the
        Req 3.4 path -- a call priced from half its tokens would understate its
        cost, and Req 3.4 asks for 0.0 rather than a partial figure.

        :param estimated: whether these counts came from the estimator rather
            than the provider (Req 2.3). Only used to count the affected calls
            for a summary warning, so a reader knows the cost figure rests on
            an estimate; it does not change the arithmetic.
        """
        self._require_model_id(model_id)

        if input_tokens is None or output_tokens is None:
            # Req 3.4: both counts record as 0, the cost is 0.0, and every
            # previously accumulated token and dollar total is left exactly as
            # it was. The step still counts: the call completed and the loop
            # writes a trace line for it, so omitting it here would break the
            # step-count equality of Req 3.7.
            self._steps += 1
            self._unavailable_calls += 1
            return 0.0

        _require_non_negative_int(input_tokens, "input_tokens")
        _require_non_negative_int(output_tokens, "output_tokens")

        # Req 3.1: both counts enter the totals, whether or not the model is
        # priced. The tokens were spent either way.
        self._steps += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if estimated:
            self._estimated_calls += 1

        price: ModelPrice | None = self._prices.lookup(model_id)
        if price is None:
            # Req 3.3: cost 0.0 for this call, tokens already accumulated above,
            # and one warning for this id however many calls use it.
            self._unpriced.add(model_id)
            return 0.0

        # Req 3.2: in/1e6 * in_price + out/1e6 * out_price, quantised to
        # COST_DECIMALS so the value returned is the value accumulated.
        cost = round(price.cost_usd(input_tokens, output_tokens), COST_DECIMALS)
        self._usd = round(self._usd + cost, COST_DECIMALS)
        return cost

    # ----------------------------------------------------------------- #
    # Reading (Req 3.5, Req 3.6)
    # ----------------------------------------------------------------- #

    def totals(self) -> Totals:
        """A snapshot of the totals for every call completed before this read.

        Reflects no call still in flight, because the fields are only ever
        written inside :meth:`record` after that call's arithmetic is done
        (Req 3.5). Every field of a later snapshot is greater than or equal to
        the same field of an earlier one, since each is a sum of non-negative
        addends (Req 3.6).
        """
        return Totals(
            steps=self._steps,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            usd=self._usd,
        )

    @property
    def total_usd(self) -> float:
        """The run's dollar total, which the loop compares against COST_CAP.

        The loop checks this before issuing each call (Req 4.8), and asking for
        one float is clearer at that site than building a whole snapshot.
        """
        return self._usd

    @property
    def steps(self) -> int:
        """Completed model calls so far."""
        return self._steps

    @property
    def input_tokens(self) -> int:
        """The run's input token total."""
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        """The run's output token total."""
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        """The run's input plus output token total."""
        return self._input_tokens + self._output_tokens

    # ----------------------------------------------------------------- #
    # Warnings (Req 3.3, Req 3.4, Req 3.9)
    # ----------------------------------------------------------------- #

    def warnings(self) -> list[str]:
        """The warning lines to print under the run summary (Req 3.9).

        One line per unpriced model id, naming it, however many calls used it
        (Req 3.3); one line for unavailable token counts stating the number of
        calls affected (Req 3.4); and one line when any call's counts were
        estimated rather than provider-reported, so a reader is not told a
        cost figure is measured when it is not (Req 2.3).

        Order is fixed -- unpriced ids first, sorted, then the two counters --
        so the summary of an identical run reads identically.
        """
        lines: list[str] = []
        for model_id in sorted(self._unpriced):
            lines.append(
                f"Warning: model {model_id!r} is absent from the price table, so its "
                f"calls were costed at $0.000000. Tokens for those calls are included "
                f"in the totals. Add an entry for it to the price table to see its "
                f"real cost."
            )
        if self._unavailable_calls:
            lines.append(
                f"Warning: token counts were unavailable for "
                f"{self._unavailable_calls} model "
                f"{self._plural(self._unavailable_calls, 'call')}, which recorded 0 "
                f"tokens and $0.000000; the totals below understate this run."
            )
        if self._estimated_calls:
            lines.append(
                f"Warning: token counts for {self._estimated_calls} model "
                f"{self._plural(self._estimated_calls, 'call')} were estimated from "
                f"message length because the provider reported none, so the cost "
                f"below is an estimate."
            )
        return lines

    def unpriced_model_ids(self) -> tuple[str, ...]:
        """Every model id this run used that the price table does not list.

        The same set the first group of :meth:`warnings` lines is rendered from,
        exposed so a caller can react to the condition without parsing prose.
        """
        return tuple(sorted(self._unpriced))

    @property
    def unavailable_count(self) -> int:
        """Calls that completed without token counts (Req 3.4)."""
        return self._unavailable_calls

    @property
    def estimated_count(self) -> int:
        """Calls whose token counts were estimated rather than reported."""
        return self._estimated_calls

    @property
    def prices(self) -> PriceTable:
        """The price table this tracker costs calls against."""
        return self._prices

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CostTracker(steps={self._steps}, input_tokens={self._input_tokens}, "
            f"output_tokens={self._output_tokens}, usd={self._usd:.6f})"
        )

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    @staticmethod
    def _require_model_id(model_id: Any) -> None:
        """A model id must be a string; an *unknown* string is fine (Req 3.3).

        The distinction matters: an id absent from the table is a tolerated
        condition with a defined outcome, while a non-string id means the
        caller wired something other than a model identifier into the accounting
        and every cost figure after it would be meaningless.
        """
        if not isinstance(model_id, str):
            raise ValueError(
                f"model_id must be a string, received {type(model_id).__name__}"
            )

    @staticmethod
    def _plural(count: int, noun: str) -> str:
        return noun if count == 1 else f"{noun}s"


def _require_non_negative_int(value: Any, name: str) -> None:
    """Req 3.1: a token count is a non-negative integer.

    ``Usage`` already enforces this, so a count arriving through
    :meth:`CostTracker.record` has been checked once. The check is repeated
    because :meth:`CostTracker.record_counts` takes bare integers from a caller
    that may never have built a ``Usage``, and a negative one is the single
    input that could break the monotonicity of Req 3.6. ``bool`` is refused
    explicitly: ``True`` is an ``int`` in Python and would silently count as one
    token.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int or None, received {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, received {value}")


__all__ = [
    "COST_DECIMALS",
    "DISPLAY_DECIMALS",
    "CostTracker",
    "Totals",
]
