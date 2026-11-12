"""Unit tests for the read_notes tool (Req 16.3, Req 10.1 - 10.7).

Req 16.3 names three cases — an absent notes file, a file of zero characters,
and a file with text whose text comes back — each of which must also assert that
the file's contents are identical after the read, because Req 10.4 makes "never
writes" a property of the tool rather than of the caller. Req 10.5 adds the
supplied-parameter case, and Req 10.6 adds the over-budget trim, including the
two edge cases the implementation documents: nothing fits, and the file carries
no entry heading at all.

Two choices run through the file.

The notes fixtures are built from ``entry_heading`` rather than from hand-typed
timestamps, so a test asserts against the format ``save_note`` actually writes
instead of against a copy of it, and ``split_entries`` is used as the oracle for
where an entry begins and ends. A trim assertion then reduces to "the body is
exactly this suffix of whole entries", which is the Req 10.6 guarantee stated
directly.

The budget is set per test with ``dataclasses.replace`` on a loaded config.
Small budgets keep the fixtures readable; ``replace`` bypasses the documented
NOTES_READ_BUDGET range, which is a parsing rule for operator input (Req 1.9)
and not a constraint on the trimming logic under test here.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_lab import grounding
from agent_lab.config import Config, load_config
from agent_lab.tools.base import ToolResult
from agent_lab.tools.read_notes import (
    ReadNotesTool,
    entry_heading,
    split_entries,
)

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def notes_file(tmp_path: Path) -> Path:
    """The configured notes path, inside pytest's temporary directory (Req 16.9).

    Deliberately not created: the absent-file case (Req 10.2) is the state a
    clean clone starts in, and every other test writes the file itself.
    """
    return tmp_path / "notes" / "notes.md"


@pytest.fixture
def config(notes_file: Path) -> Config:
    """A config whose only interesting fields are the notes path and budget.

    ``probe=False`` because the access check writes nothing a unit test needs
    and would warn about a directory that does not exist yet; the API key is a
    placeholder, since no test here reaches a provider (Req 16.8).
    """
    loaded = load_config({"OPENAI_API_KEY": "sk-test"}, probe=False)
    return dataclasses.replace(loaded, notes_file=notes_file)


def make_tool(config: Config, budget: int | None = None) -> ReadNotesTool:
    """The tool under test, optionally with a narrowed read budget."""
    if budget is not None:
        config = dataclasses.replace(config, notes_read_budget=budget)
    return ReadNotesTool(config)


def write(path: Path, data: bytes | str) -> bytes:
    """Create ``path`` with ``data`` and return the bytes now on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    path.write_bytes(raw)
    return raw


def snapshot(path: Path) -> bytes | None:
    """The file's bytes, or ``None`` when no regular file sits at the path.

    ``is_file`` rather than ``exists`` so the directory-at-the-notes-path case
    below can use the same unchanged-afterwards check as every other test.
    """
    return path.read_bytes() if path.is_file() else None


def read(tool: ReadNotesTool, notes: Path, /, **kwargs: object) -> ToolResult:
    """Invoke the tool and assert the file is byte-identical afterwards.

    Req 16.3 asks for this assertion in each of its three cases and Req 10.4
    makes it true of every invocation, so it lives in the one place every test
    goes through rather than being repeated per test. The two helper arguments
    are positional-only, because ``**kwargs`` carries names a model might
    supply — ``path`` among them (Req 10.5) — and those must reach the tool.
    """
    before = snapshot(notes)
    result = tool.execute(**kwargs)
    assert isinstance(result, ToolResult)
    assert snapshot(notes) == before, "read_notes must never write the notes file"
    return result


def succeeds(tool: ReadNotesTool, notes: Path, /, **kwargs: object) -> str:
    result = read(tool, notes, **kwargs)
    assert result.status == "success", result.content
    return result.content


def fails(tool: ReadNotesTool, notes: Path, /, **kwargs: object) -> str:
    result = read(tool, notes, **kwargs)
    assert result.status == "error", result.content
    return result.content


def quoted_body(content: str) -> str:
    """The text between the untrusted-data delimiters of a rendered result.

    Asserting through this helper checks the Req 10.1 enclosure as a side
    effect: the markers must be present exactly once each and each alone on its
    own line, or there is no single body to return.
    """
    lines = content.split("\n")
    assert lines.count(grounding.BEGIN) == 1, content
    assert lines.count(grounding.END) == 1, content
    start = lines.index(grounding.BEGIN)
    end = lines.index(grounding.END)
    assert start < end, content
    return "\n".join(lines[start + 1 : end])


def make_entries(count: int) -> tuple[str, ...]:
    """``count`` whole notes entries in the format ``save_note`` appends.

    Headings come from :func:`entry_heading` so the fixture cannot drift from
    the written format, and each body names its own index so a trimmed result
    can be checked for the entries it must *not* contain.
    """
    return tuple(
        "{heading}\n{body}\n".format(
            heading=entry_heading(
                datetime(2026, 2, 14, 9, index, 0, tzinfo=timezone.utc)
            ),
            body=f"note body number {index}",
        )
        for index in range(count)
    )


def test_fixture_entries_split_back_into_the_same_entries():
    """The oracle for every trim test below: the fixture really is N entries."""
    entries = make_entries(4)
    text = "".join(entries)
    assert split_entries(text) == entries


# --------------------------------------------------------------------------- #
# The declared spec (Req 10.5)
# --------------------------------------------------------------------------- #


def test_spec_declares_zero_parameters(config):
    spec = make_tool(config).spec
    assert spec.name == "read_notes"
    assert spec.declared_parameters == ()
    assert spec.required_parameters == ()
    assert spec.parameters["properties"] == {}
    assert spec.parameters["additionalProperties"] is False


# --------------------------------------------------------------------------- #
# Absent file (Req 16.3, Req 10.2)
# --------------------------------------------------------------------------- #


def test_absent_file_reports_that_nothing_has_been_saved(config, notes_file):
    content = succeeds(make_tool(config), notes_file)

    assert "no notes have been saved yet" in content.lower()
    assert str(notes_file) in content
    assert grounding.BEGIN not in content


def test_absent_file_is_not_created_by_the_read(config, notes_file):
    succeeds(make_tool(config), notes_file)

    assert not notes_file.exists()
    assert not notes_file.parent.exists()


# --------------------------------------------------------------------------- #
# Zero-character file (Req 16.3, Req 10.3)
# --------------------------------------------------------------------------- #


def test_empty_file_reports_that_the_file_is_empty(config, notes_file):
    write(notes_file, b"")

    content = succeeds(make_tool(config), notes_file)

    assert "empty" in content.lower()
    assert str(notes_file) in content
    assert grounding.BEGIN not in content


def test_empty_file_and_absent_file_give_distinct_messages(config, notes_file):
    absent = succeeds(make_tool(config), notes_file)
    write(notes_file, b"")
    empty = succeeds(make_tool(config), notes_file)

    assert absent != empty
    # The absent-file wording must not survive into the empty-file case: it
    # would send an operator looking for a file that is sitting right there.
    assert "does not exist" not in empty


# --------------------------------------------------------------------------- #
# File with text (Req 16.3, Req 10.1)
# --------------------------------------------------------------------------- #


def test_file_with_text_returns_that_text_enclosed_in_the_delimiters(
    config, notes_file
):
    text = "## 2026-02-14T09:31:07+00:00\nchroma persistence is at .chroma\n"
    write(notes_file, text)

    content = succeeds(make_tool(config), notes_file)

    assert text in content
    assert quoted_body(content) == text
    assert str(notes_file) in content
    assert "omitted" not in content


def test_text_within_budget_is_returned_whole_with_no_omission_marker(
    config, notes_file
):
    text = "".join(make_entries(3))
    write(notes_file, text)

    content = succeeds(make_tool(config, budget=len(text)), notes_file)

    assert quoted_body(content) == text
    assert "omitted" not in content


def test_result_is_unchanged_by_reading_twice(config, notes_file):
    """A read has no effect on the next read, which is the other half of 10.4."""
    text = "".join(make_entries(2))
    raw = write(notes_file, text)
    tool = make_tool(config)

    first = succeeds(tool, notes_file)
    second = succeeds(tool, notes_file)

    assert first == second
    assert notes_file.read_bytes() == raw


# --------------------------------------------------------------------------- #
# Supplied parameters (Req 10.5)
# --------------------------------------------------------------------------- #


def test_supplied_parameter_is_ignored_and_the_result_says_so(config, notes_file):
    text = "## 2026-02-14T09:31:07+00:00\nthe configured notes\n"
    write(notes_file, text)
    decoy = notes_file.parent / "other.md"
    write(decoy, "DECOY CONTENT THAT MUST NOT BE READ\n")

    content = succeeds(make_tool(config), notes_file, path=str(decoy))

    assert "accepts no parameters" in content
    assert str(notes_file) in content
    assert "DECOY" not in content
    # Still a normal successful read of the configured path, not a refusal.
    assert quoted_body(content) == text


def test_supplied_parameter_notice_names_the_parameter(config, notes_file):
    write(notes_file, "note\n")

    content = succeeds(make_tool(config), notes_file, file_name="../../etc/passwd")

    assert "file_name" in content
    assert "ignored" in content
    assert "../../etc/passwd" not in content


def test_several_supplied_parameters_are_all_named(config, notes_file):
    write(notes_file, "note\n")

    content = succeeds(make_tool(config), notes_file, path="/tmp/a", limit=5)

    assert "limit, path" in content
    assert "parameters" in content


def test_no_supplied_parameter_adds_no_notice(config, notes_file):
    write(notes_file, "note\n")

    content = succeeds(make_tool(config), notes_file)

    assert "accepts no parameters" not in content


# --------------------------------------------------------------------------- #
# Over-budget trimming (Req 10.6)
# --------------------------------------------------------------------------- #


def test_over_budget_returns_the_newest_whole_entries_oldest_first(
    config, notes_file
):
    entries = make_entries(4)
    text = "".join(entries)
    write(notes_file, text)
    kept = entries[-2:]
    dropped = entries[:-2]
    budget = sum(len(entry) for entry in kept)

    content = succeeds(make_tool(config, budget=budget), notes_file)

    # Exactly the two newest entries, in file order, and nothing of the two
    # older ones — not even a fragment.
    assert quoted_body(content) == "".join(kept)
    for entry in dropped:
        assert entry not in content
    assert "note body number 0" not in content
    assert "note body number 1" not in content


def test_over_budget_marker_states_the_omitted_and_total_characters(
    config, notes_file
):
    entries = make_entries(4)
    text = "".join(entries)
    write(notes_file, text)
    budget = sum(len(entry) for entry in entries[-2:])
    omitted = sum(len(entry) for entry in entries[:-2])

    content = succeeds(make_tool(config, budget=budget), notes_file)

    assert f"{omitted} of {len(text)} characters omitted" in content
    assert f"{budget}-character read budget" in content
    assert "2 most recent entries, oldest first" in content
    # The marker is this tool's own words and must sit outside the quoted block,
    # so that everything inside the delimiters is document text (Req 12.2).
    assert "omitted" not in quoted_body(content)


def test_a_budget_that_fits_one_entry_returns_exactly_that_entry(
    config, notes_file
):
    entries = make_entries(3)
    text = "".join(entries)
    write(notes_file, text)
    newest = entries[-1]

    content = succeeds(make_tool(config, budget=len(newest)), notes_file)

    assert quoted_body(content) == newest
    assert "most recent entry" in content
    assert f"{len(text) - len(newest)} of {len(text)} characters omitted" in content


def test_trim_never_cuts_inside_an_entry_for_any_budget(config, notes_file):
    """Every budget yields a suffix of whole entries, never a partial one.

    The per-budget sweep is the unit-test counterpart of the read-back property:
    a body is always ``"".join(entries[i:])`` for some ``i``.
    """
    entries = make_entries(4)
    text = "".join(entries)
    write(notes_file, text)
    suffixes = {"".join(entries[index:]) for index in range(len(entries) + 1)}

    for budget in range(1, len(text) + 1):
        content = succeeds(make_tool(config, budget=budget), notes_file)
        body = quoted_body(content) if grounding.BEGIN in content else ""
        assert body in suffixes, f"budget {budget} produced a partial entry"
        assert len(body) <= budget


def test_no_entry_fits_returns_no_entry_text_and_says_everything_was_omitted(
    config, notes_file
):
    entries = make_entries(3)
    text = "".join(entries)
    write(notes_file, text)
    budget = len(entries[-1]) - 1

    content = succeeds(make_tool(config, budget=budget), notes_file)

    # No quoted block at all: a partial note is worse than no note.
    assert grounding.BEGIN not in content
    assert grounding.END not in content
    for entry in entries:
        assert entry not in content
    assert f"All {len(text)} characters were omitted" in content
    assert f"{budget}-character read budget" in content
    assert str(notes_file) in content


def test_file_without_entry_headings_falls_back_to_a_line_aligned_tail(
    config, notes_file
):
    lines = [f"hand written line {index}" for index in range(12)]
    text = "\n".join(lines) + "\n"
    write(notes_file, text)
    budget = 65

    content = succeeds(make_tool(config, budget=budget), notes_file)
    body = quoted_body(content)

    assert body, "a hand-written notes file must still return its tail"
    assert len(body) <= budget
    # A contiguous suffix of the file that starts exactly after a newline, and
    # one that cannot be extended by another whole line within the budget.
    assert text.endswith(body)
    start = len(text) - len(body)
    assert text[start - 1] == "\n"
    previous_line_start = text.rfind("\n", 0, start - 1) + 1
    assert len(text) - previous_line_start > budget
    assert f"{start} of {len(text)} characters omitted" in content
    assert "begins mid-file" in content
    assert "no timestamped entries" in content


# --------------------------------------------------------------------------- #
# Read failures (Req 10.7)
# --------------------------------------------------------------------------- #


def test_undecodable_bytes_return_an_error_naming_the_path_and_position(
    config, notes_file
):
    raw = write(notes_file, b"valid text\n\xff\xfe more\n")

    content = fails(make_tool(config), notes_file)

    assert "not valid UTF-8" in content
    assert str(notes_file) in content
    assert "position 11" in content
    assert notes_file.read_bytes() == raw


def test_an_os_level_read_failure_returns_an_error_naming_the_path(
    config, notes_file
):
    # A directory where the notes file should be: read_bytes raises OSError on
    # every platform, with the reason differing between them, which is exactly
    # the class of failure this branch exists for.
    notes_file.mkdir(parents=True)

    content = fails(make_tool(config), notes_file)

    assert "Could not read the notes file" in content
    assert str(notes_file) in content
    assert notes_file.is_dir()


def test_a_read_failure_still_reports_the_ignored_parameters(config, notes_file):
    notes_file.mkdir(parents=True)

    content = fails(make_tool(config), notes_file, path="/etc/passwd")

    assert "accepts no parameters" in content
    assert "Could not read the notes file" in content
