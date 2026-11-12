"""Unit tests for the tool contract in `tools/base.py` (Req 5.2, Req 5.12)."""

import pytest

from agent_lab.tools.base import (
    MAX_DESCRIPTION_CHARS,
    Tool,
    ToolResult,
    ToolSpec,
    ToolSpecError,
)


def make_spec(**overrides) -> ToolSpec:
    kwargs = {
        "name": "calculator",
        "description": "Evaluate one arithmetic operation on two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": ["number", "string"]},
                "b": {"type": ["number", "string"]},
                "operation": {"type": "string"},
            },
            "required": ["a", "b", "operation"],
        },
        "timeout_seconds": 10.0,
    }
    kwargs.update(overrides)
    return ToolSpec(**kwargs)


# -- ToolSpec: name, description, timeout (Req 5.2) ---------------------------


def test_spec_exposes_name_description_schema_and_timeout():
    spec = make_spec()
    assert spec.name == "calculator"
    assert len(spec.description) <= MAX_DESCRIPTION_CHARS
    assert spec.timeout_seconds == 10.0
    assert spec.declared_parameters == ("a", "b", "operation")
    assert spec.required_parameters == ("a", "b", "operation")


def test_spec_reports_declared_type_and_required_status_per_parameter():
    # Req 5.5: the dispatcher quotes both back when it refuses a call.
    spec = make_spec(
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        }
    )
    assert spec.declared_type("query") == "string"
    assert spec.declared_type("k") == "integer"
    assert spec.declared_type("absent") is None
    assert spec.is_required("query") is True
    assert spec.is_required("k") is False


def test_description_longer_than_the_limit_is_rejected():
    with pytest.raises(ToolSpecError, match="at most 200"):
        make_spec(description="x" * (MAX_DESCRIPTION_CHARS + 1))


def test_multi_sentence_description_is_rejected():
    with pytest.raises(ToolSpecError, match="one sentence"):
        make_spec(description="Adds two numbers. Also subtracts them.")


def test_single_sentence_with_or_without_a_final_period_is_accepted():
    assert make_spec(description="Adds two numbers").description.endswith("numbers")
    assert make_spec(description="Adds two numbers.").description.endswith(".")


def test_empty_description_or_name_is_rejected():
    with pytest.raises(ToolSpecError, match="description"):
        make_spec(description="   ")
    with pytest.raises(ToolSpecError, match="name"):
        make_spec(name="")


def test_timeout_none_is_accepted_and_a_non_positive_timeout_is_not():
    # Req 9.2: interactive save_note waits on the operator with no limit, which
    # the dispatcher expresses as ``timeout_seconds is None``.
    assert make_spec(timeout_seconds=None).timeout_seconds is None
    with pytest.raises(ToolSpecError, match="positive"):
        make_spec(timeout_seconds=0)
    with pytest.raises(ToolSpecError, match="positive"):
        make_spec(timeout_seconds=-1.0)


# -- ToolSpec: parameter schema shape ----------------------------------------


def test_zero_parameter_schema_is_accepted():
    # read_notes declares no parameters at all (Req 10.5).
    spec = ToolSpec(name="read_notes", description="Read the notes file", parameters={})
    assert spec.declared_parameters == ()
    assert spec.required_parameters == ()


def test_parameter_without_a_declared_type_is_rejected():
    with pytest.raises(ToolSpecError, match="must declare a type"):
        make_spec(parameters={"properties": {"a": {}}})


def test_unsupported_parameter_type_is_rejected():
    with pytest.raises(ToolSpecError, match="unsupported types"):
        make_spec(parameters={"properties": {"a": {"type": "decimal"}}})


def test_required_parameter_that_is_not_declared_is_rejected():
    with pytest.raises(ToolSpecError, match="undeclared parameters"):
        make_spec(parameters={"properties": {"a": {"type": "number"}}, "required": ["a", "b"]})


def test_schema_is_normalised_so_the_dispatcher_never_probes_for_keys():
    spec = make_spec(parameters={"properties": {"a": {"type": "number"}}})
    assert spec.parameters["type"] == "object"
    assert spec.parameters["required"] == []
    # Req 5.6 restated in the schema the provider shows the model.
    assert spec.parameters["additionalProperties"] is False


def test_mutating_the_caller_dict_after_construction_does_not_change_the_spec():
    # Req 12.9: what the registry publishes is fixed at construction.
    params = {"properties": {"a": {"type": "number"}}, "required": ["a"]}
    spec = ToolSpec(name="t", description="A tool", parameters=params)
    params["properties"]["b"] = {"type": "string"}
    params["required"].append("b")
    assert spec.declared_parameters == ("a",)
    assert spec.required_parameters == ("a",)


# -- ToolResult (Req 5.9, Req 5.12) ------------------------------------------


def test_result_defaults_are_untruncated_and_unapproved():
    result = ToolResult(status="success", content="49761")
    assert result.truncated is False
    assert result.original_length is None
    assert result.approval is None
    assert result.is_error is False


def test_error_result_is_reported_as_an_error():
    assert ToolResult(status="error", content="unknown tool 'calc'").is_error is True


def test_truncated_result_carries_its_pre_truncation_length():
    # Req 5.12: the marker states the character count before truncation, so the
    # count has to travel with the result.
    result = ToolResult(status="success", content="x" * 4000, truncated=True, original_length=9123)
    assert result.truncated is True
    assert result.original_length == 9123


def test_truncated_result_without_a_length_is_rejected():
    with pytest.raises(ToolSpecError, match="original_length"):
        ToolResult(status="success", content="x", truncated=True)


def test_invalid_status_and_approval_values_are_rejected():
    with pytest.raises(ToolSpecError, match="status"):
        ToolResult(status="ok", content="x")
    with pytest.raises(ToolSpecError, match="approval"):
        ToolResult(status="success", content="x", approval="yes")


def test_each_approval_decision_round_trips():
    # Req 9.3, Req 9.4, Req 9.6: the decision reaches the trace record.
    for decision in ("granted", "declined", "auto"):
        assert ToolResult(status="success", content="x", approval=decision).approval == decision


def test_result_is_frozen():
    result = ToolResult(status="success", content="x")
    with pytest.raises(Exception):
        result.content = "y"  # type: ignore[misc]


# -- Tool protocol ------------------------------------------------------------


def test_an_object_with_a_spec_and_execute_satisfies_the_tool_protocol():
    class Echo:
        spec = ToolSpec(
            name="echo",
            description="Return the text it was given",
            parameters={"properties": {"text": {"type": "string"}}, "required": ["text"]},
            timeout_seconds=1.0,
        )

        def execute(self, **kwargs) -> ToolResult:
            return ToolResult(status="success", content=kwargs["text"])

    tool: Tool = Echo()
    assert isinstance(tool, Tool)
    assert tool.execute(text="hello").content == "hello"
