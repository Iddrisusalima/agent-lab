"""Unit tests for the save_note tool (Req 16.5, Req 9.2 - 9.12).

Req 16.5 names three cases this file must hold — an approved interactive write
asserting the appended text and the reported character count, a declined
interactive write asserting the notes file bytes are unchanged, and a
non-interactive append with no approval prompt — and requires operator input to
arrive through a substitute so that no terminal is involved. The task adds the
end-of-input decline (Req 9.12), the created parent directory (Req 9.8), and the
failed write (Req 9.10).

Every confirmation here runs through a real :class:`InteractiveGate` driven by
``io.StringIO`` on both sides. Substituting the gate itself would test the tool
against a fake decision rather than against the keystroke rule of Req 9.3, and
the gate is injectable precisely so that the real one can be used without a
terminal.

File contents are compared as *bytes*. Req 9.4 and Req 9.5 are stated in bytes,
and on Windows a text-mode comparison would hide exactly the newline
translation that ``save_note`` opens the file with ``newline="\\n"`` to prevent.
"""

from __future__ import annotations

import dataclasses
import io
import sys
from pathlib import Path
from typing import IO

import pytest

from agent_lab.config import Config, load_config
from agent_lab.confirmation import AutoApproveGate, InteractiveGate
from agent_lab.tools.base import ToolResult
from agent_lab.tools.read_notes import ENTRY_BOUNDARY_PATTERN
from agent_lab.tools.save_note import SaveNoteTool

KEY_VAR = "OPENAI_API_KEY"
SECRET = "sk-test"

NOTE = "Quarterly numbers land on the 14th; check them against the ledger."

# Bytes already in the notes file for the tests that assert it is untouched.
# LF-only and with a trailing blank line, i.e. what `save_note` itself writes.
EXISTING = b"## 2026-02-14T09:31:07+00:00\nan earlier note\n\n"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def make_config(notes_file: Path, *, interactive: bool = True) -> Config:
    """A real ``Config`` whose only non-default path is the notes file.

    ``probe=False`` because the access check of Req 1.10 is tested in
    ``test_config.py`` and some cases here deliberately point the path at
    something unwritable.
    """
    config = load_config({KEY_VAR: SECRET}, interactive=interactive, probe=False)
    return dataclasses.replace(config, notes_file=notes_file)


class Conversation:
    """One operator at a keyboard, made of two in-memory streams.

    ``shown`` is everything the gate displayed, which is how the prompt itself
    is asserted — present for an interactive confirmation (Req 9.2), absent for
    a non-interactive one (Req 9.6).
    """

    def __init__(self, reply: str | None) -> None:
        # `None` means end-of-input: `readline` returns "" and the gate declines.
        self._input = io.StringIO("" if reply is None else reply)
        self._output = io.StringIO()

    @property
    def gate(self) -> InteractiveGate:
        return InteractiveGate(
            input_stream=self._input, output_stream=self._output, secret=SECRET
        )

    @property
    def shown(self) -> str:
        return self._output.getvalue()

    def close(self) -> Conversation:
        """Close both streams, so the gate meets an unusable terminal."""
        self._input.close()
        self._output.close()
        return self


class ExplodingStream(io.StringIO):
    """A stdin substitute that fails the test if anything reads it.

    Used by the non-interactive case: "appended without an approval prompt"
    (Req 9.6) is only really asserted if a read attempt would be visible.
    """

    def readline(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        raise AssertionError("save_note read operator input in non-interactive mode")


def invoke(tool: SaveNoteTool, **kwargs: object) -> ToolResult:
    """Execute the tool and assert nothing escaped it (Req 5.7)."""
    result = tool.execute(**kwargs)
    assert isinstance(result, ToolResult)
    return result


def written_bytes(path: Path) -> bytes:
    assert path.is_file(), f"{path} was not created"
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# The declared timeout (the documented Req 9.2 deviation)
# --------------------------------------------------------------------------- #


def test_interactive_declares_no_timeout_and_non_interactive_declares_the_configured_one(
    tmp_path: Path,
):
    # Req 9.2 forbids a time limit while an operator may be at the prompt, so
    # the dispatcher must be told to run this tool inline (`None`). With nobody
    # to wait for, the ordinary configured bound applies.
    notes = tmp_path / "notes.md"
    interactive = make_config(notes)
    assert SaveNoteTool(interactive, Conversation("y\n").gate).spec.timeout_seconds is None

    offline = make_config(notes, interactive=False)
    tool = SaveNoteTool(offline, AutoApproveGate(offline))
    assert tool.spec.timeout_seconds == offline.tool_timeout_seconds


# --------------------------------------------------------------------------- #
# Approved interactive write (Req 16.5, Req 9.2, Req 9.3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reply", ["y\n", "Y\n", "  y  \n", "y"])
def test_approved_write_appends_the_text_and_reports_the_character_count(
    tmp_path: Path, reply: str
):
    notes = tmp_path / "notes.md"
    talk = Conversation(reply)
    tool = SaveNoteTool(make_config(notes), talk.gate)

    result = invoke(tool, text=NOTE)

    assert result.status == "success", result.content
    # Req 9.3: the decision travels out for the trace record.
    assert result.approval == "granted"

    # Req 9.7: one new entry, a timestamp heading then the text then a blank
    # line, in LF bytes on every platform.
    entry = written_bytes(notes)
    assert entry.endswith(f"\n{NOTE}\n\n".encode())
    assert b"\r" not in entry
    heading, _, body = entry.decode("utf-8").partition("\n")
    assert ENTRY_BOUNDARY_PATTERN.match(heading), heading
    assert body == f"{NOTE}\n\n"

    # Req 9.3: the number of characters appended. Both numbers are stated, and
    # both are checked against the file rather than against a literal.
    appended = entry.decode("utf-8")
    assert f"{len(appended)} characters were appended" in result.content
    assert f"{len(NOTE)} characters are the note text" in result.content

    # Req 9.2: the complete text and the destination were displayed first.
    assert NOTE in talk.shown
    assert str(notes) in talk.shown


def test_an_approved_write_appends_after_existing_content_without_disturbing_it(
    tmp_path: Path,
):
    # Req 9.7: pre-existing content stays untruncated, unoverwritten and in
    # order, which for an append means the file starts with the bytes it had.
    notes = tmp_path / "notes.md"
    notes.write_bytes(EXISTING)
    tool = SaveNoteTool(make_config(notes), Conversation("y\n").gate)

    assert invoke(tool, text=NOTE).status == "success"

    after = written_bytes(notes)
    assert after.startswith(EXISTING)
    assert after.endswith(f"{NOTE}\n\n".encode())


# --------------------------------------------------------------------------- #
# Declined interactive write (Req 16.5, Req 9.4, Req 9.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reply", ["n\n", "\n", "yes\n", "Yes\n", "y y\n", "   \n"])
def test_a_declined_write_leaves_the_file_byte_identical(tmp_path: Path, reply: str):
    notes = tmp_path / "notes.md"
    notes.write_bytes(EXISTING)
    talk = Conversation(reply)
    tool = SaveNoteTool(make_config(notes), talk.gate)

    result = invoke(tool, text=NOTE)

    # Req 9.5: byte-identical unless the operator typed exactly `y` or `Y`.
    assert notes.read_bytes() == EXISTING
    # Req 9.4: an error, so the model cannot read this as "the note is saved".
    assert result.status == "error"
    assert result.approval == "declined"
    assert "declined" in result.content
    assert str(notes) in result.content


def test_a_declined_write_does_not_create_an_absent_notes_file(tmp_path: Path):
    notes = tmp_path / "notes.md"
    tool = SaveNoteTool(make_config(notes), Conversation("n\n").gate)

    assert invoke(tool, text=NOTE).status == "error"
    assert not notes.exists()


# --------------------------------------------------------------------------- #
# No operator input available (Req 9.12)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "talk, situation",
    [
        (Conversation(None), "end-of-input before a line arrived"),
        (Conversation("y\n").close(), "a closed terminal"),
    ],
    ids=["eof", "closed"],
)
def test_unavailable_operator_input_declines_and_writes_nothing(
    tmp_path: Path, talk: Conversation, situation: str
):
    notes = tmp_path / "notes.md"
    notes.write_bytes(EXISTING)
    tool = SaveNoteTool(make_config(notes), talk.gate)

    result = invoke(tool, text=NOTE)

    assert notes.read_bytes() == EXISTING, situation
    assert result.status == "error"
    # Req 9.12: a decline in the trace vocabulary, with its own message.
    assert result.approval == "declined"
    assert "approval could not be obtained" in result.content
    assert str(notes) in result.content


# --------------------------------------------------------------------------- #
# Non-interactive mode (Req 16.5, Req 9.6)
# --------------------------------------------------------------------------- #


def test_non_interactive_mode_appends_without_an_approval_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    notes = tmp_path / "notes.md"
    config = make_config(notes, interactive=False)
    # Nothing may be read from or written to the terminal: a read raises, and an
    # empty stdout is the assertion that no prompt was displayed.
    stdout: IO[str] = io.StringIO()
    monkeypatch.setattr(sys, "stdin", ExplodingStream())
    monkeypatch.setattr(sys, "stdout", stdout)

    tool = SaveNoteTool(config, AutoApproveGate(config))
    result = invoke(tool, text=NOTE)

    assert result.status == "success", result.content
    # Req 9.6: recorded as granted automatically.
    assert result.approval == "auto"
    assert stdout.getvalue() == ""
    assert written_bytes(notes).endswith(f"\n{NOTE}\n\n".encode())


# --------------------------------------------------------------------------- #
# Missing parent directory (Req 9.8)
# --------------------------------------------------------------------------- #


def test_a_missing_parent_directory_is_created_before_appending(tmp_path: Path):
    notes = tmp_path / "notes" / "2026" / "lab.md"
    tool = SaveNoteTool(make_config(notes), Conversation("y\n").gate)

    result = invoke(tool, text=NOTE)

    assert result.status == "success", result.content
    assert notes.parent.is_dir()
    assert written_bytes(notes).endswith(f"\n{NOTE}\n\n".encode())


# --------------------------------------------------------------------------- #
# Failed write (Req 9.10)
# --------------------------------------------------------------------------- #


def test_a_failed_write_reports_the_reason_and_the_configured_path(tmp_path: Path):
    # A directory standing where the notes file should be: the append cannot
    # succeed, and the failure arrives as the OSError Req 9.10 is written for.
    notes = tmp_path / "notes.md"
    notes.mkdir()
    tool = SaveNoteTool(make_config(notes), Conversation("y\n").gate)

    result = invoke(tool, text=NOTE)

    assert result.status == "error"
    # The approval still happened, so the trace records what the operator said.
    assert result.approval == "granted"
    assert str(notes) in result.content
    assert "failed" in result.content
    # Req 9.10: no second attempt, and nothing of the note anywhere.
    assert "no further write was attempted" in result.content
    assert list(notes.iterdir()) == []
