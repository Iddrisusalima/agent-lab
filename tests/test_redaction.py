"""Unit tests for `redaction.py` and the exit code table (Req 1.11, Req 4.10)."""

from agent_lab import exit_codes
from agent_lab.redaction import REDACTION_MARKER, redact


def test_none_secret_returns_text_unchanged():
    assert redact("sk-abc appears here", None) == "sk-abc appears here"


def test_empty_secret_returns_text_unchanged():
    # An empty secret is a substring of every position; substituting it would
    # shred the text instead of redacting it.
    text = "nothing to hide"
    assert redact(text, "") == text


def test_every_occurrence_is_replaced():
    out = redact("key=sk-123 and again sk-123.", "sk-123")
    assert out == f"key={REDACTION_MARKER} and again {REDACTION_MARKER}."
    assert "sk-123" not in out


def test_secret_is_matched_literally_not_as_a_regex():
    # A key containing regex metacharacters must still be redacted, and the
    # metacharacters must not match anything else.
    out = redact("a.c and sk-a+b.c* here", "sk-a+b.c*")
    assert "sk-a+b.c*" not in out
    assert out == f"a.c and {REDACTION_MARKER} here"


def test_secret_inside_a_longer_token_is_redacted():
    # Unanchored substitution: a key embedded in a serialised JSON line or a
    # longer identifier is still caught.
    out = redact('{"auth": "Bearer sk-xyz"}', "sk-xyz")
    assert "sk-xyz" not in out
    assert REDACTION_MARKER in out


def test_absent_secret_leaves_text_unchanged():
    assert redact("step 1: calculator", "sk-xyz") == "step 1: calculator"


def test_custom_marker_is_used():
    assert redact("sk-1", "sk-1", marker="[hidden]") == "[hidden]"


def test_empty_text_is_handled():
    assert redact("", "sk-1") == ""


def test_exit_codes_are_distinct_and_documented():
    codes = {
        "final answer": exit_codes.FINAL_ANSWER,
        "step limit": exit_codes.STEP_LIMIT,
        "cost cap": exit_codes.COST_CAP,
        "model error": exit_codes.MODEL_ERROR,
        "invalid task": exit_codes.INVALID_TASK,
        "configuration error": exit_codes.CONFIGURATION_ERROR,
        "eval failure": exit_codes.EVAL_FAILURE,
    }
    # Req 4.10: a distinct code per outcome, success zero and every other
    # outcome non-zero.
    assert len(set(codes.values())) == len(codes)
    assert codes["final answer"] == 0
    assert all(code != 0 for name, code in codes.items() if name != "final answer")
    # Code 1 stays reserved for an uncaught exception.
    assert 1 not in set(codes.values())
