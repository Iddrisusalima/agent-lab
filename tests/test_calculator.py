"""Unit tests for the calculator tool (Req 16.2, Req 6.3 - 6.5, Req 6.9).

Req 16.2 names the cases this file must hold: one asserted arithmetic result per
operation, division by zero, an unsupported operation answered with the four
supported values, and a non-numeric parameter named together with the value
received for it. Req 6.7 and Req 6.8 add the two rendering rules, and Req 6.9
adds the out-of-range result.

Every assertion reads the result the way the agent loop does: a successful
``content`` is the bare rendered numeral and nothing else, so comparing it to a
string literal is both the strictest and the simplest check available.
"""

from decimal import Decimal

import pytest

from agent_lab.tools.base import ToolResult
from agent_lab.tools.calculator import (
    FRACTIONAL_DIGITS,
    MAX_LITERAL_CHARS,
    MAX_MAGNITUDE,
    MAX_MAGNITUDE_TEXT,
    OPERATIONS,
    CalculatorTool,
)


@pytest.fixture
def calculator() -> CalculatorTool:
    return CalculatorTool()


def compute(calculator: CalculatorTool, **kwargs) -> ToolResult:
    """Invoke the tool and assert nothing escaped it (Req 6.6)."""
    result = calculator.execute(**kwargs)
    assert isinstance(result, ToolResult)
    return result


def succeeds(calculator: CalculatorTool, **kwargs) -> str:
    result = compute(calculator, **kwargs)
    assert result.status == "success", result.content
    return result.content


def fails(calculator: CalculatorTool, **kwargs) -> str:
    result = compute(calculator, **kwargs)
    assert result.status == "error", result.content
    return result.content


# -- the spec the model is shown (Req 5.2, Req 6.1) ---------------------------


def test_spec_declares_both_operands_and_the_four_operations():
    spec = CalculatorTool().spec
    assert spec.name == "calculator"
    assert spec.required_parameters == ("a", "b", "operation")
    assert spec.declared_type("a") == ["number", "string"]
    assert spec.declared_type("b") == ["number", "string"]
    assert spec.parameters["properties"]["operation"]["enum"] == list(OPERATIONS)


# -- one asserted result per operation (Req 16.2, Req 6.2) --------------------


def test_add_returns_the_sum(calculator):
    assert succeeds(calculator, a=1843, b=27, operation="add") == "1870"


def test_subtract_returns_the_difference(calculator):
    assert succeeds(calculator, a=1843, b=27, operation="subtract") == "1816"


def test_multiply_returns_the_product(calculator):
    assert succeeds(calculator, a=1843, b=27, operation="multiply") == "49761"


def test_divide_returns_the_quotient(calculator):
    # 1843 / 27 is 68.259259..., half-up at the tenth fractional digit.
    assert succeeds(calculator, a=1843, b=27, operation="divide") == "68.2592592593"


def test_decimal_literal_strings_are_accepted_as_operands(calculator):
    # Req 6.1: a string holding a decimal literal, surrounding whitespace and
    # an exponent part included.
    assert succeeds(calculator, a=" 2.5 ", b="4e2", operation="multiply") == "1000"


def test_arithmetic_is_decimal_rather_than_binary(calculator):
    # The operands are the decimal numbers the model wrote, so the answer is the
    # decimal one: 0.30000000000000004 would be a float artefact, not a result.
    assert succeeds(calculator, a=0.1, b=0.2, operation="add") == "0.3"


# -- division by zero (Req 16.2, Req 6.3) -------------------------------------


@pytest.mark.parametrize("zero", [0, 0.0, -0.0, "0", "0.0", "-0", "-0.000", "0e9"])
def test_divide_by_every_spelling_of_zero_is_an_error(calculator, zero):
    message = fails(calculator, a=7, b=zero, operation="divide")
    assert "division by zero is undefined" in message
    # Req 6.3: no numeric result accompanies the error.
    assert "7" not in message.replace("zero", "")


def test_zero_divided_by_zero_is_the_same_error(calculator):
    assert "division by zero is undefined" in fails(calculator, a=0, b=0, operation="divide")


def test_zero_is_a_permitted_operand_elsewhere(calculator):
    # Req 6.2 permits zero; only `divide` refuses it as the second operand.
    assert succeeds(calculator, a=0, b=0, operation="add") == "0"
    assert succeeds(calculator, a=0, b=5, operation="divide") == "0"


# -- unsupported operation (Req 16.2, Req 6.4) --------------------------------


@pytest.mark.parametrize(
    "operation",
    ["Add", "ADD", "modulo", "", " add", "plus", None, 7],
    ids=["cased", "upper", "unknown", "empty", "padded", "prose", "none", "nonstring"],
)
def test_unsupported_operation_lists_the_four_supported_values(calculator, operation):
    message = fails(calculator, a=2, b=3, operation=operation)
    for supported in OPERATIONS:
        assert supported in message


def test_absent_operation_lists_the_four_supported_values(calculator):
    message = fails(calculator, a=2, b=3)
    for supported in OPERATIONS:
        assert supported in message
    assert "no operation was supplied" in message


def test_unsupported_operation_returns_no_numeric_result(calculator):
    # Req 6.4: the refusal carries the supported values, never an answer the
    # model could mistake for the product of 6 and 7.
    assert "42" not in fails(calculator, a=6, b=7, operation="Multiply")


# -- a non-numeric parameter (Req 16.2, Req 6.5) ------------------------------


@pytest.mark.parametrize("name", ["a", "b"])
def test_non_numeric_parameter_names_the_parameter_and_the_value(calculator, name):
    arguments = {"a": 1, "b": 2, "operation": "add", name: "two"}
    message = fails(calculator, **arguments)
    assert f"parameter {name}" in message
    assert "'two'" in message
    # Req 6.5: no numeric result accompanies the error.
    assert "calculator error" in message


def test_boolean_operand_is_refused_rather_than_treated_as_one(calculator):
    # `True` is an `int` in Python, so a silent conversion here would turn the
    # model's type confusion into a plausible wrong answer.
    message = fails(calculator, a=True, b=2, operation="add")
    assert "parameter a" in message
    assert "boolean" in message
    assert "True" in message


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_empty_or_whitespace_only_operand_is_refused(calculator, value):
    message = fails(calculator, a=value, b=2, operation="add")
    assert "parameter a" in message
    assert repr(value) in message


def test_operand_longer_than_the_literal_limit_is_refused(calculator):
    literal = "1" * (MAX_LITERAL_CHARS + 1)
    message = fails(calculator, a=literal, b=2, operation="add")
    assert "parameter a" in message
    assert str(MAX_LITERAL_CHARS) in message
    assert str(len(literal)) in message


def test_absent_operand_is_refused_and_named(calculator):
    message = fails(calculator, a=1, operation="add")
    assert "parameter b" in message
    assert "required" in message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_operand_is_refused(calculator, value):
    message = fails(calculator, a=value, b=2, operation="add")
    assert "parameter a" in message
    assert "finite" in message


@pytest.mark.parametrize("value", [None, [1], {"n": 1}, object()])
def test_operand_of_an_unsupported_type_is_refused(calculator, value):
    message = fails(calculator, a=value, b=2, operation="add")
    assert "parameter a" in message


def test_operand_above_the_magnitude_bound_is_refused(calculator):
    # Req 6.2: the bound is on the absolute value, so a negative operand past it
    # is refused the same way.
    message = fails(calculator, a="-1000000000000001", b=1, operation="add")
    assert "parameter a" in message
    assert MAX_MAGNITUDE_TEXT in message


def test_operand_at_the_magnitude_bound_is_accepted(calculator):
    assert succeeds(calculator, a=str(MAX_MAGNITUDE), b=0, operation="add") == "1000000000000000"


def test_every_problem_is_reported_together(calculator):
    # Req 6.4 and Req 6.5: one error result carrying both mistakes, so the model
    # does not spend a step discovering the second one.
    message = fails(calculator, a="two", b=3, operation="Add")
    assert "parameter a" in message
    assert "'two'" in message
    assert "add, subtract, multiply, divide" in message


# -- rendering (Req 6.7, Req 6.8) ---------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "operation", "expected"),
    [
        (10, 4, "divide", "2.5"),
        (8, 2, "divide", "4"),
        (-6, 2, "divide", "-3"),
        ("1e15", 0, "add", "1000000000000000"),
        ("1e15", "-1e15", "add", "0"),
        (-1, 0, "multiply", "0"),
        (0, -1, "multiply", "0"),
        ("2.50", "2.50", "add", "5"),
        ("1.5", "0.25", "add", "1.75"),
    ],
    ids=[
        "fractional",
        "integral-quotient",
        "negative-integral",
        "large-integral-no-exponent",
        "zero-from-cancellation",
        "negative-zero-left",
        "negative-zero-right",
        "trailing-zeros-stripped",
        "exact-fraction",
    ],
)
def test_rendering_is_plain_decimal(calculator, a, b, operation, expected):
    assert succeeds(calculator, a=a, b=b, operation=operation) == expected


def test_integral_result_carries_no_exponent_or_fractional_part(calculator):
    content = succeeds(calculator, a="1e15", b=1, operation="divide")
    assert content == "1000000000000000"
    assert "e" not in content.lower()
    assert "." not in content


def test_non_integral_result_keeps_at_most_ten_fractional_digits(calculator):
    content = succeeds(calculator, a=1, b=3, operation="divide")
    assert content == "0.3333333333"
    assert len(content.split(".")[1]) == FRACTIONAL_DIGITS


def test_rounding_is_half_up_rather_than_half_even(calculator):
    # Req 6.8 asks for half-up. Python's built-in `round` is half-even and would
    # render this as 0, so this case is the one that distinguishes the two.
    assert succeeds(calculator, a="0.00000000005", b=0, operation="add") == "0.0000000001"
    assert succeeds(calculator, a="0.00000000015", b=0, operation="add") == "0.0000000002"


def test_a_value_below_the_last_kept_digit_renders_as_zero(calculator):
    # Not "0.0000000000", which would claim a precision the answer lacks.
    assert succeeds(calculator, a="0.00000000001", b=0, operation="add") == "0"


def test_commutative_operations_render_identically(calculator):
    for operation in ("add", "multiply"):
        left = succeeds(calculator, a="-3.5", b=7, operation=operation)
        right = succeeds(calculator, a=7, b="-3.5", operation=operation)
        assert left == right


# -- an out-of-range result (Req 6.9) -----------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "operation"),
    [
        ("1000000000000000", "10", "multiply"),
        ("1000000000000000", "1000000000000000", "add"),
        ("-1000000000000000", "1000000000000000", "subtract"),
        ("1000000000000000", "0.0000000001", "divide"),
    ],
    ids=["multiply", "add", "subtract", "divide"],
)
def test_result_beyond_the_magnitude_bound_is_an_error(calculator, a, b, operation):
    message = fails(calculator, a=a, b=b, operation=operation)
    assert "result lies outside the supported numeric range" in message
    assert MAX_MAGNITUDE_TEXT in message


def test_result_at_the_magnitude_bound_is_still_a_result(calculator):
    # The bound is inclusive: 1e15 is permitted, and renders without an exponent.
    assert (
        succeeds(calculator, a="500000000000000", b=2, operation="multiply")
        == "1000000000000000"
    )


def test_out_of_range_result_blames_the_result_not_an_operand(calculator):
    # Both operands are inside the bound, so the message must not name one of
    # them as the failing parameter (Req 6.5 against Req 6.9).
    message = fails(calculator, a="999999999999999", b="999999999999999", operation="multiply")
    assert "parameter a" not in message
    assert "parameter b" not in message


# -- nothing escapes (Req 6.6) ------------------------------------------------


def test_no_argument_combination_raises(calculator):
    hostile = [
        {},
        {"a": 1},
        {"operation": "add"},
        {"a": object(), "b": object(), "operation": object()},
        {"a": "1e-9999999999", "b": 1, "operation": "add"},
        {"a": Decimal("1e-9999999999"), "b": 1, "operation": "multiply"},
        {"a": "nan", "b": 1, "operation": "add"},
        {"a": "inf", "b": 1, "operation": "add"},
        {"a": 1, "b": 2, "operation": "add", "unexpected": "ignored"},
    ]
    for arguments in hostile:
        result = compute(calculator, **arguments)
        assert result.status in ("success", "error")


def test_decimal_operands_are_accepted_from_a_direct_call(calculator):
    # Not a shape the model can send, but the conversion gate admits it, so the
    # bound still has to be enforced for it.
    assert succeeds(calculator, a=Decimal("1.5"), b=Decimal("1.5"), operation="add") == "3"
    assert "parameter a" in fails(calculator, a=MAX_MAGNITUDE * 2, b=1, operation="add")
