"""Loading of the versioned system-prompt resource (Req 12.10).

Req 12.10 asks for the System_Prompt to live in a single text resource carrying
a version identifier, with no second copy of the instruction text held in code,
so that an operator can edit the agent's instructions without touching Python.
This module is the whole of the code side of that arrangement: it reads the
resource, parses the version out of its first-line header, and returns both.

Two consequences are deliberate.

**No instruction text appears here.** This module does not assert that the
resource says any particular thing, and quotes none of its wording — not in the
code, not in these docstrings. A content check would have to restate the
sentences it checks for, which is the second copy Req 12.10 forbids. The
resource lint in ``tests/test_resources.py`` owns that job instead: it reads the
file and asserts that no module under ``src/`` repeats its distinctive
sentences, which only works while this module keeps quiet about the content.

**A broken resource stops the run before step 1.** :class:`PromptError` derives
from :class:`~agent_lab.config.ConfigError`, so an absent, empty, unreadable, or
unversioned prompt file travels the path the CLI already has for a
configuration fault: a message on stderr and exit code 6, before any provider
request is issued. The prompt is configuration that happens to live in a file.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import ConfigError

# The repository root: ``src/agent_lab/prompts.py`` -> parents[2]. The resource
# sits outside the package, next to ``pricing/prices.json``, because both are
# files an operator is expected to open and edit.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Req 12.10: the one resource, named for the version it carries. A revision gets
# a new file (``system_prompt.v2.md``) rather than an edit that leaves traces
# claiming a version the text no longer has.
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.v1.md"

# The version identifier as it may appear in the first-line header: a ``v``
# followed by one or more dot-separated numbers, on a word boundary, so that
# ``# Agent Lab System Prompt v1`` and ``# ... (v1.2)`` both parse and an
# ordinary word beginning with "v" does not. The leading ``v`` is captured so
# the returned identifier reads the way the header and the file name spell it.
_VERSION_PATTERN = re.compile(r"\b(v\d+(?:\.\d+)*)\b", re.IGNORECASE)


class PromptError(ConfigError):
    """The system-prompt resource could not be loaded or carries no version.

    A subclass of :class:`~agent_lab.config.ConfigError` so that the CLI's
    existing handling applies: the message names the path, the process exits
    with ``exit_codes.CONFIGURATION_ERROR``, and no step runs (Req 1.8's
    pattern, applied to the prompt resource of Req 12.10).
    """


def load_system_prompt(
    path: Path | str | None = None,
) -> tuple[str, str]:
    """Return the prompt text and the version identifier from its header.

    Args:
        path: The resource to read. Defaults to :data:`DEFAULT_PROMPT_PATH`.

    Returns:
        A ``(text, version)`` pair. ``text`` is the complete resource with
        newlines normalised to ``\\n`` and surrounding blank space trimmed,
        header line included — the header is part of the single copy, and
        leaving it in place means the model is told which version it is running
        under and the trace of a run can be matched to the text that produced
        it. ``version`` is the identifier parsed from the first line, such as
        ``"v1"``.

    Raises:
        PromptError: The path does not exist, is a directory, cannot be read,
            does not decode as UTF-8, holds nothing but blank space, or has a
            first line with no version identifier in it. Every message names
            the path, so the fix is apparent from the message alone.
    """
    resource = Path(path) if path is not None else DEFAULT_PROMPT_PATH

    try:
        raw = resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptError(
            f"System prompt resource not found: {resource}. "
            "Agent Lab reads its instructions from this file; restore it or "
            "pass a different path."
        ) from None
    except IsADirectoryError:
        raise PromptError(
            f"System prompt resource is a directory, not a file: {resource}."
        ) from None
    except UnicodeDecodeError:
        raise PromptError(
            f"System prompt resource is not valid UTF-8 text: {resource}."
        ) from None
    except OSError as exc:
        # Permissions, a vanished drive, a path too long for the platform.
        raise PromptError(
            f"System prompt resource could not be read: {resource} "
            f"({exc.strerror or exc})."
        ) from None

    # A file written on Windows arrives with CRLF line endings. Normalising
    # here keeps the text identical whatever checkout produced it, which
    # matters because the eval report and the trace quote this text.
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        raise PromptError(
            f"System prompt resource is empty: {resource}. It must hold the "
            "instruction text, with a version identifier in its first line."
        )

    first_line = text.split("\n", 1)[0]
    match = _VERSION_PATTERN.search(first_line)
    if match is None:
        raise PromptError(
            f"System prompt resource carries no version identifier in its "
            f"first line: {resource}. The first line read {first_line!r}; it "
            "must contain a version such as 'v1', as in "
            "'# Agent Lab System Prompt v1'."
        )

    return text, match.group(1).lower()


__all__ = [
    "DEFAULT_PROMPT_PATH",
    "PROJECT_ROOT",
    "PromptError",
    "load_system_prompt",
]
