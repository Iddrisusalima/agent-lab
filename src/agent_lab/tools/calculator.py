"""The ``calculator`` tool: one arithmetic operation on two numbers (Req 6).

The tool exists so that a numeric answer never depends on the model's mental
arithmetic (Req 6.2). That purpose sets the two unusual decisions in this
module.

**Arithmetic is decimal, not binary.** Every operand arrives as a decimal
number — a JSON number the model wrote, or a string holding a decimal literal
— and the answer is read back as a decimal number. ``decimal.Decimal`` computes
in that same base, so ``0.1 + 0.2`` is ``0.3`` here rather than
``0.30000000000000004``, and the 1e15 magnitude bound of Req 6.2 and Req 6.9 is
checked against the exact value instead of against a binary approximation of
it. A float operand is converted through its ``repr`` for the same reason: the
shortest string that round-trips the float is the number the model meant.

**Rendering is explicit.** Req 6.8 asks for half-up rounding to 10 fractional
digits, and Python's built-in ``round`` is half-even — ``round(0.5)`` is ``0``
— so rounding goes through ``Decimal.quantize`` with ``ROUND_HALF_UP``, named at
the call site. Req 6.7 asks for exact integers in plain decimal with no
fractional and no exponent part, which rules out ``str(Decimal("1E+15"))``;
``format(value, "f")`` is the spelling that never produces an exponent.

A successful result's content is the rendered number and nothing else — no
prose, no units, no restatement of the operands. The agent loop feeds that text
straight back to the model as an observation, and an eval check reads it
literally (Req 17), so a bare numeral is both the easiest thing for the model to
reuse and the easiest thing to assert on.

Nothing here raises (Req 6.6). Every refusal — an unsupported operation, an
operand that is not a number, a division by zero, a result outside the supported
range — is a ``ToolResult`` with ``status="error"`` carrying a message written
for the model, naming the failing parameter and the value received for it
(Req 6.3 - 6.5, Req 6.9). The dispatcher would convert a raised exception into
an error result anyway (Req 5.7), but its message is written for a developer
reading a trace, and the model needs to know which parameter to resend.
"""

from __future__ import annotations

import decimal
import math
import re
from decimal import Decimal
from typing import Any, Sequence

from agent_lab.tools.base import ToolResult, ToolSpec

# --------------------------------------------------------------------------- #
# The contract, as constants
# --------------------------------------------------------------------------- #

# Req 6.1: the four operations, compared case-sensitively, listed in this order
# by the error of Req 6.4 and by the parameter schema the model is shown.
OPERATIONS: tuple[str, ...] = ("add", "subtract", "multiply", "divide")

# Req 6.1: the longest string literal accepted for an operand. A decimal number
# that matters to an operator fits easily; a longer string is far more likely to
# be prose, a pasted table, or an attempt to make the parser work hard.
MAX_LITERAL_CHARS = 64

# Req 6.2, Req 6.9: the magnitude bound on both operands and results, as an
# exact decimal. ``Decimal(10) ** 15`` rather than ``1e15`` so the comparison
# never involves a binary float.
MAX_MAGNITUDE = Decimal(10) ** 15

# How the bound is written in a message. Grouped, because a model and an
# operator both read "1,000,000,000,000,000" faster than sixteen bare digits.
MAX_MAGNITUDE_TEXT = "1,000,000,000,000,000"

# Req 6.8: fractional digits kept for a result that is not an exact integer.
FRACTIONAL_DIGITS = 10

# Req 5.8: the dispatcher's per-tool limit. The registry factory passes
# ``config.tool_timeout_seconds``; this default exists so that a direct
# construction — in a test, or in a script — needs no Config. Arithmetic on
# bounded decimals finishes in microseconds, far inside the 100 milliseconds of
# Req 6.2, so this value never actually governs anything.
DEFAULT_TIMEOUT_SECONDS = 10.0

# Req 5.2: one sentence, at most 200 characters. It names the four operations,
# because the model chooses one by exact spelling and the schema's enum is not
# always surfaced to it as prominently as the description.
DESCRIPTION = (
    "Apply one arithmetic operation (add, subtract, multiply or divide) to two "
    "numbers and return the exact result."
)

# Req 6.1: a decimal numeric literal — optional sign, digits with at most one
# decimal point, optional exponent. Surrounding whitespace is stripped before
# the match, so it is not part of the pattern; internal whitespace is not.
#
# The pattern is narrower than ``Decimal`` accepts, deliberately. ``Decimal``
# would take "nan", "inf", "Infinity" and "1_000", none of which is a decimal
# numeric literal under Req 6.1, and "nan" in particular would otherwise flow
# into the arithmetic and come back out as a result that is not a number.
_DECIMAL_LITERAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

# Working precision for the arithmetic. Operands are bounded by 1e15, so a sum
# or a product needs at most 31 integer digits; the rest is headroom for a
# division that does not terminate, whose digits beyond the tenth fractional one
# only ever decide a rounding boundary.
_PRECISION = 50

# Req 6.8: the quantum ``quantize`` rounds to, i.e. Decimal("1E-10").
_QUANTUM = Decimal(1).scaleb(-FRACTIONAL_DIGITS)

# Distinguishes "the model sent no value for this parameter" from "the model
# sent None". Req 6.5 treats an absent operand as an error, and ``None`` is an
# error for a different reason, so the two cannot share a representation.
_MISSING: Any = object()

# How much of a received value a message echoes before cutting it short. Long
# enough to show any real number and any plausible typo, short enough that a
# multi-kilobyte string cannot push the rest of the message out of the
# dispatcher's result budget (Req 5.12).
_ECHO_CHARS = 80


class _OperandError(ValueError):
    """An operand could not be converted; the message is for the model.

    Private and never allowed out of this module: ``execute`` catches it and
    turns it into an error result, because Req 6.6 requires that no exception
    reach the dispatcher. It exists so that :func:`_to_number` can report *why*
    a conversion failed from wherever it noticed, instead of threading a
    second return value through every branch.
    """


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #


class CalculatorTool:
    """Apply one of four arithmetic operations to two numbers (Req 6).

    Holds no state beyond its spec: no configuration, no filesystem, no history.
    One instance can serve every step of a run, and two concurrent calls cannot
    interfere, because each call builds its own decimal context.
    """

    def __init__(self, *, timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.spec = ToolSpec(
            name="calculator",
            description=DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    # Req 6.1: a number or a string holding a decimal literal.
                    # Both are declared, so the dispatcher's type check accepts
                    # either without the tool having to be lenient about a type
                    # it never declared (Req 5.5).
                    "a": {
                        "type": ["number", "string"],
                        "description": (
                            "First operand, as a number or a decimal numeric literal string "
                            f"of at most {MAX_LITERAL_CHARS} characters."
                        ),
                    },
                    "b": {
                        "type": ["number", "string"],
                        "description": (
                            "Second operand, as a number or a decimal numeric literal string "
                            f"of at most {MAX_LITERAL_CHARS} characters."
                        ),
                    },
                    "operation": {
                        "type": "string",
                        # The enum is advisory: it tells the model the exact
                        # spellings, and a provider may show it. Req 6.4 is
                        # enforced here in the tool regardless, because the
                        # dispatcher validates types and names only.
                        "enum": list(OPERATIONS),
                        "description": (
                            "Operation to apply, exactly one of: " + ", ".join(OPERATIONS) + "."
                        ),
                    },
                },
                "required": ["a", "b", "operation"],
            },
            timeout_seconds=timeout_seconds,
        )

    # -- execution ------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Compute ``a operation b`` and return the rendered value (Req 6.2).

        Returns an error result instead for every condition Req 6.3 - 6.5 and
        Req 6.9 name, and raises nothing in any case (Req 6.6).

        Undeclared keyword arguments are ignored. The dispatcher already refuses
        a request carrying one (Req 5.6), so by the time this runs an extra key
        can only come from a direct call, and arithmetic on ``a`` and ``b`` is
        the same answer either way.
        """
        try:
            return self._execute(kwargs)
        except Exception as exc:  # noqa: BLE001 - Req 6.6: nothing may escape.
            # Unreachable by design: every path below returns a result, and the
            # arithmetic is already guarded. Kept because "the tool never
            # raises" is a property a reviewer should be able to confirm by
            # reading this method alone, not by auditing every branch under it.
            return _error_result(
                [
                    f"the operation could not be completed ({type(exc).__name__}); "
                    f"check the operands and the operation and try again"
                ]
            )

    def _execute(self, arguments: dict[str, Any]) -> ToolResult:
        context = _working_context()

        # Every input problem is collected before any is reported. An operation
        # of "Add" together with an operand of "two" is two mistakes, and a
        # model told about one of them will spend a step discovering the other
        # (Req 6.4, Req 6.5).
        operation = arguments.get("operation", _MISSING)
        problems: list[str] = []
        operation_problem = _operation_problem(operation)
        if operation_problem is not None:
            problems.append(operation_problem)

        operands: dict[str, Decimal] = {}
        for name in ("a", "b"):
            try:
                operands[name] = _to_number(name, arguments.get(name, _MISSING), context)
            except _OperandError as exc:
                problems.append(str(exc))

        if problems:
            return _error_result(problems)

        a, b = operands["a"], operands["b"]

        # Req 6.3: every spelling of zero — 0, 0.0, -0, "0e9", "-0.000" — is
        # zero to ``is_zero``, which is why the check is on the converted value
        # and not on the text that produced it.
        if operation == "divide" and b.is_zero():
            return _error_result(
                [
                    "division by zero is undefined: parameter b converts to zero, "
                    "so no numeric result exists"
                ]
            )

        try:
            value = _apply(str(operation), a, b, context)
            if not value.is_finite() or value.copy_abs() > MAX_MAGNITUDE:
                return _error_result([_result_out_of_range()])
            content = _render(value, context)
        except (decimal.DecimalException, ArithmeticError, ValueError):
            # A decimal signal here means the exact result cannot be represented
            # at all — an exponent far outside the working context, reachable
            # only from an operand like "1e-9999999999". Req 6.9's answer covers
            # it: the result lies outside the supported numeric range.
            return _error_result([_result_out_of_range()])

        return ToolResult(status="success", content=content)


# --------------------------------------------------------------------------- #
# Conversion — the single gate every operand passes through
# --------------------------------------------------------------------------- #


def _to_number(name: str, value: Any, context: decimal.Context) -> Decimal:
    """Convert one operand to an exact decimal, or raise ``_OperandError``.

    The only place that enforces any rule about an operand, which is what makes
    the rules checkable: the :data:`MAX_LITERAL_CHARS` limit, the refusal of
    booleans, the finiteness requirement and the :data:`MAX_MAGNITUDE` bound are
    each written once, here (Req 6.1, Req 6.5).

    Every raised message names ``name`` and echoes the value received for it, as
    Req 6.5 requires, so the model can see which of the two operands to resend
    and what the tool actually received — which is often not what the model
    believes it sent.

    Accepted: ``int``, ``float``, ``Decimal``, and ``str`` holding a decimal
    literal with optional surrounding whitespace. Refused, with a reason:
    everything else.
    """
    if value is _MISSING:
        raise _OperandError(f"parameter {name} is required but no value was supplied")

    # Before the int branch, because ``True`` is an ``int`` in Python and
    # ``Decimal(True)`` is a perfectly good ``Decimal(1)``. Silently computing
    # with a boolean would turn a model's type confusion into a plausible wrong
    # answer, which is the worst outcome available here (Req 6.5).
    if isinstance(value, bool):
        raise _OperandError(
            f"parameter {name} must be a number, not a boolean, received {_echo(value)}"
        )

    if isinstance(value, str):
        if len(value) > MAX_LITERAL_CHARS:
            # Measured on the string as received, whitespace included: the
            # length limit is a limit on what was sent, not on what survives
            # trimming.
            raise _OperandError(
                f"parameter {name} must be at most {MAX_LITERAL_CHARS} characters, "
                f"received {len(value)} characters: {_echo(value)}"
            )
        text = value.strip()
        if not text:
            raise _OperandError(
                f"parameter {name} must not be empty or whitespace only, "
                f"received {_echo(value)}"
            )
        if _DECIMAL_LITERAL.fullmatch(text) is None:
            raise _OperandError(
                f"parameter {name} must be a decimal number such as 42, -3.5 or 2e3, "
                f"received {_echo(value)}"
            )
        number = _construct(name, value, text)
    elif isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = _construct(name, value, value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise _OperandError(
                f"parameter {name} must be a finite number, received {_echo(value)}"
            )
        # Through ``repr``: the shortest decimal string that round-trips the
        # float, i.e. the number the model wrote, rather than the binary
        # expansion ``Decimal(0.1)`` would produce.
        number = _construct(name, value, repr(value))
    else:
        raise _OperandError(
            f"parameter {name} must be a number or a string holding a decimal number, "
            f"received {_echo(value)}"
        )

    # Only reachable for a ``Decimal`` the caller built, since every other
    # branch above has already excluded a non-finite value.
    if not number.is_finite():
        raise _OperandError(f"parameter {name} must be a finite number, received {_echo(value)}")

    # ``copy_abs`` rather than ``abs``: it is exact and ignores the ambient
    # context, so the bound is checked against the value itself (Req 6.2).
    if number.copy_abs() > MAX_MAGNITUDE:
        raise _OperandError(
            f"parameter {name} must have an absolute value of at most {MAX_MAGNITUDE_TEXT}, "
            f"received {_echo(value)}"
        )

    # A value can satisfy every rule above and still be unusable: an operand
    # like "1e-9999999999" is finite and tiny, but its exponent lies outside
    # what the working context can compute with. Finding that out here names the
    # parameter at fault (Req 6.5); finding it out during the arithmetic could
    # only blame the result.
    try:
        context.plus(number)
    except (decimal.DecimalException, ArithmeticError, ValueError):
        raise _OperandError(
            f"parameter {name} has an exponent outside the supported numeric range, "
            f"received {_echo(value)}"
        ) from None

    return number


def _construct(name: str, received: Any, source: str | int) -> Decimal:
    """Build a ``Decimal`` from an already-validated source, or raise.

    ``Decimal`` construction is exact and context-free for every value this
    module admits, but it is not total: an exponent with more digits than the
    decimal module can hold raises, and a sufficiently enormous ``int`` can
    exhaust memory. Both are reported as a conversion failure naming the
    parameter rather than allowed to escape (Req 6.5, Req 6.6).
    """
    try:
        return Decimal(source)
    except (decimal.DecimalException, ValueError, OverflowError, MemoryError):
        raise _OperandError(
            f"parameter {name} could not be converted to a number, received {_echo(received)}"
        ) from None


def _operation_problem(operation: Any) -> str | None:
    """Return why ``operation`` is unusable, or ``None`` when it is one of four.

    Req 6.4 wants one answer for every way the parameter can be wrong — absent,
    empty, misspelled, differently cased, not even a string — and that answer
    must list the four supported values. Case sensitivity is the point of the
    plain ``in`` test: ``"Add"`` is refused rather than normalised, so the model
    learns the exact spelling it has to send instead of relying on the tool to
    guess (Req 6.1).
    """
    supported = ", ".join(OPERATIONS)
    if operation is _MISSING:
        return (
            f"operation is required and must be exactly one of {supported}, "
            f"but no operation was supplied"
        )
    if isinstance(operation, str) and operation in OPERATIONS:
        return None
    return (
        f"operation must be exactly one of {supported}, in lower case, "
        f"received {_echo(operation)}"
    )


# --------------------------------------------------------------------------- #
# Arithmetic and rendering
# --------------------------------------------------------------------------- #


def _working_context() -> decimal.Context:
    """A fresh decimal context for one call.

    Fresh, rather than the thread-local context, for two reasons: nothing this
    module does can change the arithmetic of any other caller, and nothing any
    other caller does — a provider SDK, a test, the cost tracker's money
    arithmetic — can change the arithmetic here. Rounding is ``ROUND_HALF_UP``
    so that an intermediate rounding, if one ever happens at 50 digits, leans
    the same way as the rendering of Req 6.8.

    The traps are the decimal module's defaults, named explicitly: an invalid
    operation, a division by zero and an overflow raise, and are caught at the
    call site. ``Inexact``, ``Rounded`` and ``Underflow`` stay untrapped, which
    is what lets a non-terminating division produce a usable answer.
    """
    return decimal.Context(
        prec=_PRECISION,
        rounding=decimal.ROUND_HALF_UP,
        traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
    )


def _apply(operation: str, a: Decimal, b: Decimal, context: decimal.Context) -> Decimal:
    """Apply one supported operation (Req 6.2).

    Called only with an operation already checked against :data:`OPERATIONS`
    and, for ``divide``, with a non-zero ``b`` (Req 6.3). The context's methods
    are used rather than the operators, so the precision and rounding above
    govern the result no matter what the ambient thread context says.
    """
    if operation == "add":
        return context.add(a, b)
    if operation == "subtract":
        return context.subtract(a, b)
    if operation == "multiply":
        return context.multiply(a, b)
    return context.divide(a, b)


def _render(value: Decimal, context: decimal.Context) -> str:
    """Render ``value`` in plain decimal notation (Req 6.7, Req 6.8).

    An exact integer comes back with no fractional part and no exponent part;
    anything else is rounded half-up to 10 fractional digits with trailing zeros
    removed. ``format(..., "f")`` is what guarantees the "no exponent part"
    half: ``str(Decimal("1E+15"))`` keeps the exponent, while ``format`` writes
    the digits out.

    Rounding comes first, so "is this an exact integer" is asked of the number
    that will actually be shown. A value of ``0.00000000001`` rounds to zero and
    is rendered ``0`` — not ``0.0000000000``, which would claim a precision the
    answer does not have.

    Zero is special-cased for its sign. Decimal keeps a signed zero, so
    ``multiply(-1, 0)`` is ``-0`` and would render as ``-0``; a calculator that
    answers ``-0`` invites the reader to wonder what it means. It also costs
    nothing elsewhere: ``-0`` and ``0`` compare equal, so the two orders of a
    multiplication still render identically (Req 6.6).
    """
    rounded = value.quantize(_QUANTUM, rounding=decimal.ROUND_HALF_UP, context=context)
    if rounded.is_zero():
        return "0"

    integral = rounded.to_integral_value()
    if rounded == integral:
        return f"{integral:f}"

    # A non-integral value quantized to 10 places always has a decimal point and
    # at least one non-zero fractional digit, so stripping trailing zeros can
    # neither empty the string nor leave a dangling point.
    return f"{rounded:f}".rstrip("0")


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


def _result_out_of_range() -> str:
    """The Req 6.9 message: the result, not an operand, is the problem."""
    return (
        f"the result lies outside the supported numeric range: a result must be finite "
        f"with an absolute value of at most {MAX_MAGNITUDE_TEXT}"
    )


def _error_result(problems: Sequence[str]) -> ToolResult:
    """One error result carrying every problem found, and no numeric value.

    A single problem reads as a sentence; several read as a list, because a
    model parsing "and" out of a run-on sentence is a model that fixes one of
    two mistakes. No partial answer is ever included: Req 6.3, Req 6.5 and
    Req 6.9 each require that no numeric result accompany the error, and the
    content here is the only thing the loop sees.
    """
    if len(problems) == 1:
        content = f"calculator error: {problems[0]}."
    else:
        listed = "\n".join(f"- {problem}." for problem in problems)
        content = f"calculator error:\n{listed}"
    return ToolResult(status="error", content=content)


def _echo(value: Any) -> str:
    """How a received value is quoted back in a message (Req 6.5).

    ``repr``, so that an empty string, a whitespace-only string and the string
    ``"0"`` are each distinguishable from one another and from the number zero —
    which is the whole point of quoting the value back. Long values are cut and
    their true length stated, so a pasted document cannot crowd out the rest of
    the message.
    """
    text = repr(value)
    if len(text) <= _ECHO_CHARS:
        return text
    return f"{text[:_ECHO_CHARS]}... ({len(text)} characters)"


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DESCRIPTION",
    "FRACTIONAL_DIGITS",
    "MAX_LITERAL_CHARS",
    "MAX_MAGNITUDE",
    "MAX_MAGNITUDE_TEXT",
    "OPERATIONS",
    "CalculatorTool",
]
