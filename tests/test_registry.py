"""Unit tests for the tool registry (Req 5.1, Req 5.2)."""

import pytest

from agent_lab.tools.base import ToolResult, ToolSpec, ToolSpecError
from agent_lab.tools.registry import DuplicateToolError, ToolRegistry


class StubTool:
    """A minimal `Tool`: a spec and an `execute` that returns a result."""

    def __init__(self, name: str, **spec_overrides) -> None:
        kwargs = {
            "name": name,
            "description": f"Stand in for the {name} tool during a test.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "timeout_seconds": 10.0,
        }
        kwargs.update(spec_overrides)
        self.spec = ToolSpec(**kwargs)

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(status="success", content=repr(kwargs))


# -- registration and names (Req 5.1) ----------------------------------------


def test_register_keeps_registration_order_in_names():
    registry = ToolRegistry()
    for name in ("calculator", "search_docs", "save_note", "read_notes"):
        registry.register(StubTool(name))
    assert registry.names() == ("calculator", "search_docs", "save_note", "read_notes")
    assert len(registry) == 4


def test_constructor_registers_an_iterable_of_tools():
    registry = ToolRegistry([StubTool("calculator"), StubTool("read_notes")])
    assert registry.names() == ("calculator", "read_notes")


def test_duplicate_name_is_rejected_and_message_lists_registered_names():
    registry = ToolRegistry([StubTool("calculator")])
    with pytest.raises(DuplicateToolError) as excinfo:
        registry.register(StubTool("calculator"))
    message = str(excinfo.value)
    assert "calculator" in message
    # The first registration stands; the rejected one did not replace it.
    assert registry.names() == ("calculator",)


def test_duplicate_comparison_is_case_sensitive_and_exact():
    # Req 5.1 compares names case-sensitively, and Req 5.3 resolves them the
    # same way, so a case variant is a different tool rather than a collision.
    registry = ToolRegistry([StubTool("calculator")])
    registry.register(StubTool("Calculator"))
    assert registry.names() == ("calculator", "Calculator")


def test_registering_a_non_tool_is_refused():
    registry = ToolRegistry()
    with pytest.raises(ToolSpecError):
        registry.register(object())


def test_registering_a_tool_without_execute_is_refused():
    class SpecOnly:
        spec = ToolSpec(name="calculator", description="No execute here.")

    with pytest.raises(ToolSpecError):
        ToolRegistry().register(SpecOnly())


# -- resolve (Req 5.3, Req 5.4) ----------------------------------------------


def test_resolve_returns_the_tool_on_an_exact_match():
    tool = StubTool("save_note")
    registry = ToolRegistry([tool])
    assert registry.resolve("save_note") is tool
    assert "save_note" in registry


def test_resolve_returns_none_for_an_unknown_or_differently_cased_name():
    registry = ToolRegistry([StubTool("save_note")])
    assert registry.resolve("Save_Note") is None
    assert registry.resolve("save_notes") is None
    assert registry.resolve(" save_note") is None
    assert registry.resolve("") is None


def test_resolve_returns_none_for_a_non_string_name():
    # The name arrives from the model, so it is not guaranteed to be a string.
    registry = ToolRegistry([StubTool("save_note")])
    assert registry.resolve(None) is None
    assert registry.resolve(17) is None


# -- schemas (Req 5.2) --------------------------------------------------------


def test_schemas_declare_name_description_and_parameter_types():
    registry = ToolRegistry(
        [
            StubTool(
                "search_docs",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"],
                },
            )
        ]
    )
    (schema,) = registry.schemas()
    assert schema["name"] == "search_docs"
    assert len(schema["description"]) <= 200
    assert schema["parameters"]["properties"]["query"] == {"type": "string"}
    assert schema["parameters"]["required"] == ["query"]
    # An execution concern, not something the model decides.
    assert "timeout_seconds" not in schema


def test_schema_name_set_equals_the_registered_names():
    names = ("calculator", "search_docs", "save_note", "read_notes")
    registry = ToolRegistry([StubTool(name) for name in names])
    assert [schema["name"] for schema in registry.schemas()] == list(names)
    assert {schema["name"] for schema in registry.schemas()} == set(registry.names())


def test_mutating_a_returned_schema_does_not_change_the_registry():
    # Req 12.9: nothing downstream can edit what the next call offers.
    registry = ToolRegistry([StubTool("calculator")])
    schemas = registry.schemas()
    schemas[0]["name"] = "hijacked"
    schemas[0]["parameters"]["properties"]["text"]["type"] = "object"
    schemas.append({"name": "extra"})

    fresh = registry.schemas()
    assert len(fresh) == 1
    assert fresh[0]["name"] == "calculator"
    assert fresh[0]["parameters"]["properties"]["text"] == {"type": "string"}


def test_specs_expose_the_declarations_including_the_timeout():
    registry = ToolRegistry([StubTool("save_note", timeout_seconds=None)])
    (spec,) = registry.specs()
    assert spec.name == "save_note"
    assert spec.timeout_seconds is None


def test_an_empty_registry_reports_nothing():
    registry = ToolRegistry()
    assert registry.names() == ()
    assert registry.schemas() == []
    assert registry.resolve("calculator") is None
    assert len(registry) == 0
