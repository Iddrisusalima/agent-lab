"""Unit tests for the agent loop's termination conditions (Req 16.7).

Req 16.7 names three tests, and they are the first three sections below: a
final answer ends the run with the assistant text of the reply that asked for
no tool (Req 4.3); reaching ``MAX_STEPS`` performs exactly that many steps and
prints the step-limit message (Req 4.7); reaching ``COST_CAP`` ends the run
*before* the next step's model call, leaving the total at no more than the cap
plus one call (Req 4.8, Req 4.9). Two more sections cover the neighbouring
claims the task names: a dispatcher error result reaches the model inside the
step that produced it (Req 5.11), and each outcome carries its own documented
exit code (Req 4.10, Req 4.12).

Everything here runs offline against ``FakeModelClient`` from ``conftest``
(Req 16.8): the model's side of each run is a script, and the only live
collaborators are the real dispatcher and the real cost tracker, so the step
order and the arithmetic under test are the shipped ones.

The prices are a literal table rather than the committed ``pricing/prices.json``
for the reason ``test_cost.py`` gives: a cost-cap assertion computed from
published prices would start failing when a provider changes them. One dollar
per million input tokens and two per million output tokens makes each scripted
call's cost readable off the usage counts.

Abnormal-termination messages are asserted through a ``PlainRenderer`` writing
into ``io.StringIO`` buffers, which is the same path the CLI wires to the real
streams (Req 13.8). The loop prints neither the final answer nor the run
summary -- those are the CLI's -- so nothing here asserts on them.
"""

import io

import pytest

from agent_lab import exit_codes
from agent_lab.agent_loop import (
    COST_CAP,
    MAX_STEPS,
    AgentLoop,
    RunOutcome,
    exit_code_for,
)
from agent_lab.cost import CostTracker
from agent_lab.model_client import ModelClientError
from agent_lab.pricing import ModelPrice, PriceTable
from agent_lab.render import PlainRenderer
from agent_lab.tools.base import ToolResult, ToolSpec
from agent_lab.tools.dispatcher import ToolDispatcher
from agent_lab.tools.registry import ToolRegistry

from conftest import FAKE_API_KEY, FAKE_MODEL_ID, make_reply, make_tool_call, make_usage

# --------------------------------------------------------------------------- #
# Prices: $1 per million input tokens, $2 per million output tokens
# --------------------------------------------------------------------------- #

LITERAL_PRICES = PriceTable({FAKE_MODEL_ID: ModelPrice(FAKE_MODEL_ID, 1.0, 2.0, "2026-10-31")})

#: A call priced at exactly $0.10, so three of them cross the $0.25 cap and the
#: crossing step can be named without arithmetic in the assertion.
DIME_INPUT_TOKENS = 100_000
DIME_OUTPUT_TOKENS = 0
DIME = LITERAL_PRICES.lookup(FAKE_MODEL_ID).cost_usd(DIME_INPUT_TOKENS, DIME_OUTPUT_TOKENS)


# --------------------------------------------------------------------------- #
# A stand-in tool, so no test depends on the four shipped ones
# --------------------------------------------------------------------------- #


class EchoTool:
    """Declares one required string parameter and succeeds."""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="echo",
            description="Echo the text it was given back to the caller.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            timeout_seconds=10.0,
        )
        self.calls: list[dict] = []

    def execute(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(status="success", content=f"echo: {kwargs.get('text', '')}")


class Harness:
    """One wired loop plus the pieces a test asserts on."""

    def __init__(self, client, **loop_kwargs) -> None:
        self.client = client
        self.tool = EchoTool()
        self.registry = ToolRegistry([self.tool])
        self.dispatcher = ToolDispatcher(self.registry, secret=FAKE_API_KEY)
        self.cost = CostTracker(prices=LITERAL_PRICES)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.renderer = PlainRenderer(
            api_key=FAKE_API_KEY, stdout=self.stdout, stderr=self.stderr
        )
        self.loop = AgentLoop(
            client,
            self.dispatcher,
            self.registry,
            self.cost,
            trace_sink=None,
            renderer=self.renderer,
            system_prompt="You are a test fixture.",
            **loop_kwargs,
        )

    def run(self, task: str = "What is in my notes?"):
        return self.loop.run(task)

    @property
    def errors(self) -> str:
        """Everything the loop wrote to stderr (Req 13.8)."""
        return self.stderr.getvalue()


def echo_call(text: str = "hello"):
    """A well-formed request for the stand-in tool."""
    return make_tool_call("echo", {"text": text})


# --------------------------------------------------------------------------- #
# 1. A final answer (Req 16.7, Req 4.3)
# --------------------------------------------------------------------------- #


def test_reply_without_tool_call_ends_run_with_its_text(make_model_client):
    # Req 4.3: no tool request plus non-whitespace text is the final answer,
    # and the answer is that reply's text.
    client = make_model_client(make_reply("The answer is 42."))
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.FINAL_ANSWER
    assert result.final_answer == "The answer is 42."
    assert result.steps == 1
    assert client.call_count == 1
    # No tool was requested, so none was dispatched.
    assert result.tools_called == []
    assert harness.tool.calls == []
    # Req 4.14: a final answer is not an abnormal termination.
    assert result.error_message is None
    assert harness.errors == ""


def test_final_answer_text_is_returned_unmodified(make_model_client):
    # Req 4.3 returns "that assistant text": the loop neither trims nor
    # reformats it, so a report quoting the answer quotes what the model wrote.
    text = "  Line one.\nLine two.  "
    client = make_model_client(make_reply(text))

    result = Harness(client).run()

    assert result.outcome is RunOutcome.FINAL_ANSWER
    assert result.final_answer == text


def test_final_answer_after_a_tool_step_counts_both_steps(make_model_client):
    # The ordinary two-step shape: one tool step, then the answer. The run ends
    # on the reply with no tool call, not on the first reply that had text.
    client = make_model_client(
        make_reply("Looking it up.", [echo_call("notes")]),
        make_reply("Your notes mention the budget."),
    )
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.FINAL_ANSWER
    assert result.final_answer == "Your notes mention the budget."
    assert result.steps == 2
    assert result.tools_called == ["echo"]


# --------------------------------------------------------------------------- #
# 2. Reaching MAX_STEPS (Req 16.7, Req 4.6, Req 4.7)
# --------------------------------------------------------------------------- #


def test_endless_tool_requests_end_at_max_steps(make_model_client):
    # Req 4.6, Req 4.7: a model that never stops asking for tools gets exactly
    # MAX_STEPS steps, and the step after the last one is never issued.
    client = make_model_client(make_reply("Still working.", [echo_call()]), repeat_last=True)
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.STEP_LIMIT
    assert result.steps == MAX_STEPS
    # The step count is the number of model calls, and no call was made after
    # the bound was reached (Req 4.4, Req 4.7).
    assert client.call_count == MAX_STEPS
    assert len(result.tools_called) == MAX_STEPS
    # Req 4.7: a step-limited run produced no final answer, and says so rather
    # than offering the last thing the model happened to write.
    assert result.final_answer is None


def test_step_limit_message_is_printed_to_stderr(make_model_client):
    # Req 4.7: the printed message states the limit that was reached and that
    # no final answer was produced. Req 13.8 puts it on stderr.
    client = make_model_client(make_reply("Still working.", [echo_call()]), repeat_last=True)
    harness = Harness(client)

    result = harness.run()

    printed = harness.errors
    assert f"step limit of {MAX_STEPS}" in printed
    assert "no final answer was produced" in printed
    # The same text is carried on the result for a caller that cannot read the
    # terminal, and is not printed twice by the loop.
    assert result.error_message is not None
    assert result.error_message in printed
    assert printed.count("step limit of") == 1


def test_configured_step_bound_is_the_one_enforced(make_model_client):
    # Req 1.6: the bound is whatever was injected, so a lower configured value
    # ends the run earlier by the same path. Keeps the test above honest about
    # which number it is reading.
    client = make_model_client(make_reply("Still working.", [echo_call()]), repeat_last=True)
    harness = Harness(client, max_steps=3)

    result = harness.run()

    assert result.outcome is RunOutcome.STEP_LIMIT
    assert result.steps == 3
    assert client.call_count == 3
    assert "step limit of 3 steps" in harness.errors


# --------------------------------------------------------------------------- #
# 3. Reaching COST_CAP (Req 16.7, Req 4.8, Req 4.9)
# --------------------------------------------------------------------------- #


def test_cost_cap_ends_the_run_before_the_next_step_starts(make_model_client):
    # Three $0.10 calls take the total to $0.30, which is at or above the $0.25
    # cap, so the fourth step's check ends the run before its model call.
    client = make_model_client(
        make_reply(
            "Still working.",
            [echo_call()],
            usage=make_usage(DIME_INPUT_TOKENS, DIME_OUTPUT_TOKENS),
        ),
        repeat_last=True,
    )
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.COST_CAP
    # Req 4.8: the call of the step that found the cap crossed was never
    # issued, so the model was called exactly once per completed step.
    assert result.steps == 3
    assert client.call_count == 3
    # The cap, not the step bound, is what ended this run.
    assert result.steps < MAX_STEPS
    assert result.final_answer is None


def test_total_cost_stays_within_the_cap_plus_one_call(make_model_client):
    # Req 4.9: the worst case is the cap plus one call, because the call that
    # crossed the cap was already in flight when the cap was crossed.
    client = make_model_client(
        make_reply(
            "Still working.",
            [echo_call()],
            usage=make_usage(DIME_INPUT_TOKENS, DIME_OUTPUT_TOKENS),
        ),
        repeat_last=True,
    )
    harness = Harness(client)

    result = harness.run()

    assert result.cost_usd == pytest.approx(3 * DIME)
    assert result.cost_usd >= COST_CAP  # the cap really was crossed
    assert result.cost_usd <= COST_CAP + DIME + 1e-12
    # The reported total is the tracker's, so the figure the CLI prints and the
    # figure the bound was checked against are the same number (Req 3.7).
    assert result.cost_usd == pytest.approx(harness.cost.total_usd)


def test_cost_cap_message_states_the_cap_and_the_total(make_model_client):
    # Req 4.8: the message states the cap and the total reached, both in
    # dollars to at least four decimal places.
    client = make_model_client(
        make_reply(
            "Still working.",
            [echo_call()],
            usage=make_usage(DIME_INPUT_TOKENS, DIME_OUTPUT_TOKENS),
        ),
        repeat_last=True,
    )
    harness = Harness(client)

    result = harness.run()

    printed = harness.errors
    assert f"${COST_CAP:.4f}" in printed
    assert f"${result.cost_usd:.4f}" in printed
    assert "no further model call" in printed
    assert result.error_message in printed


def test_a_run_under_the_cap_is_not_cost_capped(make_model_client):
    # The negative half of Req 4.8: the check is "at or above", so a run whose
    # total stays below the cap reaches its final answer untouched.
    client = make_model_client(make_reply("Cheap answer.", usage=make_usage(10, 5)))
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.FINAL_ANSWER
    assert result.cost_usd < COST_CAP
    assert harness.errors == ""


# --------------------------------------------------------------------------- #
# 4. An error result is appended within the same step (Req 5.11)
# --------------------------------------------------------------------------- #


def test_error_result_reaches_the_model_in_the_same_step(make_model_client):
    # Req 5.11: a tool request missing its required parameter comes back as an
    # error result, and that result is appended inside the step that produced
    # it -- so it is part of the conversation the *next* model call receives.
    client = make_model_client(
        make_reply("Calling the tool.", [make_tool_call("echo", {})]),
        make_reply("That call was malformed; here is the answer instead."),
    )
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.FINAL_ANSWER
    assert result.steps == 2

    # The second call carries the task, the assistant message that asked for the
    # tool, and the tool result -- in that order (Req 2.12, Req 5.11).
    second_call = client.calls[1]
    assert second_call.roles == ("user", "assistant", "tool")

    error_content = second_call.messages[-1].content
    assert "echo" in error_content
    assert "text" in error_content  # Req 5.5: the invalid parameter is named
    assert client.tool_results_submitted() == [error_content]

    # No tool code ran, which is what makes this an error result rather than a
    # tool failure (Req 5.3).
    assert harness.tool.calls == []
    # The attempt is still recorded as a dispatched call (Req 4.2).
    assert result.tools_called == ["echo"]


def test_error_result_is_appended_before_the_next_model_call(make_model_client):
    # The ordering claim of Req 5.11 stated as such: at the moment of the first
    # call the conversation holds only the task, and the error result is
    # already there by the second call. One run, two snapshots.
    client = make_model_client(
        make_reply("", [make_tool_call("echo", {"text": 7})]),
        make_reply("Recovered."),
    )
    harness = Harness(client)

    result = harness.run()

    assert [call.roles for call in client.calls] == [
        ("user",),
        ("user", "assistant", "tool"),
    ]
    assert result.outcome is RunOutcome.FINAL_ANSWER
    # A wrong-typed argument is the other Req 5.5 case, and takes the same path.
    assert "text" in client.calls[1].messages[-1].content
    assert harness.tool.calls == []


# --------------------------------------------------------------------------- #
# 5. The documented exit code per outcome (Req 4.10, Req 4.12, Req 4.13)
# --------------------------------------------------------------------------- #


def test_final_answer_run_exits_zero(make_model_client):
    result = Harness(make_model_client(make_reply("Done."))).run()

    assert result.outcome is RunOutcome.FINAL_ANSWER
    assert result.exit_code == exit_codes.FINAL_ANSWER == 0


def test_step_limited_run_exits_with_the_step_limit_code(make_model_client):
    client = make_model_client(make_reply("More.", [echo_call()]), repeat_last=True)

    result = Harness(client).run()

    assert result.outcome is RunOutcome.STEP_LIMIT
    assert result.exit_code == exit_codes.STEP_LIMIT


def test_cost_capped_run_exits_with_the_cost_cap_code(make_model_client):
    client = make_model_client(
        make_reply(
            "More.",
            [echo_call()],
            usage=make_usage(DIME_INPUT_TOKENS, DIME_OUTPUT_TOKENS),
        ),
        repeat_last=True,
    )

    result = Harness(client).run()

    assert result.outcome is RunOutcome.COST_CAP
    assert result.exit_code == exit_codes.COST_CAP


def test_failed_model_call_exits_with_the_model_error_code(make_model_client):
    # Req 4.13: the typed error after the client's own retries. The failed call
    # performed no step, so nothing was recorded for it.
    client = make_model_client(ModelClientError("the provider refused the request"))
    harness = Harness(client)

    result = harness.run()

    assert result.outcome is RunOutcome.MODEL_ERROR
    assert result.exit_code == exit_codes.MODEL_ERROR
    assert result.steps == 0
    assert result.cost_usd == 0.0
    # Req 4.13: the conversation state of the completed steps is retained.
    assert [message.role for message in result.messages] == ["user"]
    assert "the model call failed" in harness.errors


def test_invalid_task_exits_with_the_invalid_task_code(make_model_client):
    # Req 4.12: a whitespace-only task starts no run and issues no model call.
    client = make_model_client(make_reply("never reached"))
    harness = Harness(client)

    result = harness.run("   \n\t ")

    assert result.outcome is RunOutcome.INVALID_TASK
    assert result.exit_code == exit_codes.INVALID_TASK
    assert client.call_count == 0
    assert result.steps == 0
    assert result.messages == ()
    assert "empty" in harness.errors


def test_every_outcome_has_a_distinct_documented_exit_code():
    # Req 4.10: one distinct, documented code per outcome, so a caller tells the
    # termination paths apart without parsing terminal text.
    codes = {outcome: exit_code_for(outcome) for outcome in RunOutcome}

    assert codes == {
        RunOutcome.FINAL_ANSWER: 0,
        RunOutcome.STEP_LIMIT: 2,
        RunOutcome.COST_CAP: 3,
        RunOutcome.MODEL_ERROR: 4,
        RunOutcome.INVALID_TASK: 5,
    }
    assert len(set(codes.values())) == len(RunOutcome)
    # Only a final answer is success, and 1 is left to the interpreter's own
    # uncaught-exception code.
    assert [code for code in codes.values() if code == 0] == [0]
    assert 1 not in set(codes.values())


def test_unrecognised_outcome_is_not_read_as_success():
    # Whatever produced it, the run did not demonstrably answer the question.
    assert exit_code_for("something-else") == exit_codes.MODEL_ERROR
