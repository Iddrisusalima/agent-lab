"""Documented process exit codes.

Every run ends with exactly one outcome (Req 4.14), and every outcome maps to a
distinct exit code so a caller can tell the four termination paths apart without
parsing terminal text (Req 4.10). The same table appears in the README, which is
what makes these codes "documented" in the sense Req 4.10, 4.12 and 4.13 require.

Code 1 is deliberately left unused: the interpreter already uses it for an
uncaught exception, so reserving it keeps "the agent decided to stop" distinct
from "the agent crashed".
"""

# Req 4.10: the run produced a final answer. The success path.
FINAL_ANSWER = 0

# Req 4.7, Req 4.10: the run reached MAX_STEPS before producing a final answer.
STEP_LIMIT = 2

# Req 4.8, Req 4.10: the accumulated cost reached COST_CAP before a step began.
COST_CAP = 3

# Req 4.13, Req 4.10: the model client raised its typed error after exhausting
# retries, or returned a reply the loop cannot act on (more than 8 tool
# requests, an unregistered tool name, or neither text nor a tool request).
MODEL_ERROR = 4

# Req 4.12: the task string was empty, whitespace only, or longer than 4000
# characters after trimming. No run started and no model call was issued.
INVALID_TASK = 5

# Req 1.8, Req 1.9, Req 2.9: configuration error — missing or empty API key, an
# out-of-range value, or an unknown provider identifier. Raised before step 1.
CONFIGURATION_ERROR = 6

# Req 14.13, Req 15.4, Req 15.8, Req 15.14, Req 15.15: the eval suite failed —
# a suite defect, an output path collision, an unknown task id, or any task that
# failed or did not run.
EVAL_FAILURE = 7

__all__ = [
    "FINAL_ANSWER",
    "STEP_LIMIT",
    "COST_CAP",
    "MODEL_ERROR",
    "INVALID_TASK",
    "CONFIGURATION_ERROR",
    "EVAL_FAILURE",
]
