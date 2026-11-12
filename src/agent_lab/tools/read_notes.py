"""The ``read_notes`` tool: read back the notes file, never write it (Req 10).

The tool reads one path — the configured ``NOTES_FILE`` — and reads it the only
way a read-only tool can: ``Path.read_bytes``, decoded here, with no mode, flag
or branch anywhere in this module that can open the file for writing (Req 10.4).

Three things in here are worth a reader's attention.

**The notes format lives here.** ``save_note`` prefixes every appended entry
with an ISO 8601 timestamp (Req 9.7), and Req 10.6 asks this tool to drop
*whole* entries when the file exceeds the read budget. That makes the prefix a
parsing contract between the two tools rather than decoration, so it is defined
once — :data:`ENTRY_HEADING_PREFIX`, :func:`entry_heading`,
:data:`ENTRY_BOUNDARY_PATTERN` — in the module that has to parse it, and
``save_note`` writes what :func:`entry_heading` returns. A format defined in the
writer and re-guessed by the reader is a format that drifts.

**Over-budget trimming never emits a partial entry.** A suffix of whole entries
is returned, oldest-first among those returned, with a marker stating how many
characters were dropped (Req 10.6). When not even the most recent entry fits
inside the budget, the tool returns no entry text at all and says so. Returning
the last N characters of an entry instead would be more text but worse text: a
note cut off at an arbitrary character is a note whose meaning the model cannot
trust, and Property 10 asks for exactly this guarantee ("no partial entry
appears"). The one exception is a notes file with no timestamp prefix anywhere
— an operator's hand-written file — which has no entry boundaries to cut on; see
:func:`_trim_to_budget`.

**Parameters are tolerated, not honoured.** The spec declares zero parameters,
so the dispatcher's undeclared-parameter rule (Req 5.6) normally refuses a
request that carries any. Req 10.5 nevertheless specifies what happens if one
arrives here: read the configured path, read nothing else, and state in the
result that the tool accepts no parameters. ``execute`` therefore accepts
``**kwargs``, ignores every one of them, and never lets a supplied value reach
the filesystem — the path comes from the injected :class:`Config` and from
nowhere else.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_lab.config import Config
from agent_lab.grounding import wrap
from agent_lab.tools.base import ToolResult, ToolSpec, ToolStatus

# --------------------------------------------------------------------------- #
# The notes entry format, shared with save_note (Req 9.7, Req 10.6)
# --------------------------------------------------------------------------- #

# What opens an entry's first line. A Markdown heading marker, because the
# default notes path is `notes.md` and an operator reads this file in an editor
# as often as the agent reads it through this tool.
ENTRY_HEADING_PREFIX = "## "

# The canonical entry heading `save_note` writes is
#
#     ## 2026-02-14T09:31:07+00:00
#
# i.e. ENTRY_HEADING_PREFIX followed by an ISO 8601 instant in UTC at second
# precision with an explicit ``+00:00`` offset. Second precision because a note
# is an operator action and sub-second timing says nothing about it; an explicit
# offset because a bare local timestamp cannot be compared across machines.
ENTRY_TIMESTAMP_PRECISION = "seconds"


def entry_heading(moment: datetime | None = None) -> str:
    """Return the entry heading line for ``moment``, without its newline.

    ``save_note`` calls this to build the prefix it appends, so the written
    format and the pattern parsed by :data:`ENTRY_BOUNDARY_PATTERN` cannot drift
    apart (Req 9.7). A naive ``moment`` is read as UTC; ``None`` means now.
    """
    if moment is None:
        moment = datetime.now(timezone.utc)
    elif moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).isoformat(
        timespec=ENTRY_TIMESTAMP_PRECISION
    )
    return f"{ENTRY_HEADING_PREFIX}{stamp}"


# Where one entry ends and the next begins. Anchored at a line start, so a
# timestamp quoted *inside* a note's body cannot split that note in two.
#
# The accepted timestamp shape is wider than what `entry_heading` writes:
# fractional seconds, a ``Z`` suffix, an offset without its colon, and a missing
# offset are all matched. Reading tolerantly costs nothing and means a notes
# file written by an earlier version of this tool, or edited by hand, still
# splits into entries instead of collapsing into one unsplittable blob.
ENTRY_BOUNDARY_PATTERN = re.compile(
    r"^"
    + re.escape(ENTRY_HEADING_PREFIX)
    + r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
    r"[^\n]*$",
    re.MULTILINE,
)


def entry_offsets(text: str) -> tuple[int, ...]:
    """The character offset at which each entry of ``text`` starts.

    Offsets are ascending and partition ``text`` into contiguous entries: entry
    *i* is ``text[offsets[i]:offsets[i + 1]]``, and the last runs to the end.

    Text preceding the first heading — an operator's own preamble, or notes
    written before this format existed — is reported as an entry starting at 0,
    so no character of the file falls outside an entry and the omitted-character
    count in the Req 10.6 marker stays exact.
    """
    starts = [match.start() for match in ENTRY_BOUNDARY_PATTERN.finditer(text)]
    if not starts:
        return (0,) if text else ()
    if starts[0] != 0:
        starts.insert(0, 0)
    return tuple(starts)


def split_entries(text: str) -> tuple[str, ...]:
    """``text`` split into whole entries, oldest first.

    Concatenating the result reproduces ``text`` exactly. Used by the budget
    trim here and by the read-back property test as its oracle for entry
    boundaries.
    """
    offsets = entry_offsets(text)
    bounds = (*offsets, len(text))
    return tuple(text[start:end] for start, end in zip(offsets, bounds[1:]))


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #

DESCRIPTION = (
    "Read back the operator's saved notes file; this tool takes no parameters "
    "and never changes the file."
)


class ReadNotesTool:
    """Return the configured notes file's text for the model to read (Req 10).

    The notes path and the read budget are constructor state taken from the
    injected :class:`Config`. Neither is a parameter, so no model-supplied value
    can redirect the read or widen the budget (Req 10.5, Req 12.9).
    """

    def __init__(self, config: Config, *, timeout_seconds: float | None = None) -> None:
        # Only the two fields this tool uses are kept, not the whole config, so
        # a reader can see at a glance that nothing else here is configurable.
        self._notes_file: Path = config.notes_file
        self._budget: int = config.notes_read_budget
        self.spec = ToolSpec(
            name="read_notes",
            description=DESCRIPTION,
            # Req 10.5: zero declared parameters. `base` normalises this to
            # ``{"type": "object", "properties": {}, "required": [],
            # "additionalProperties": False}``, which is what tells a provider —
            # and through it the model — that there is nothing to supply.
            parameters={},
            timeout_seconds=(
                config.tool_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
        )

    # -- execution ------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Read the configured notes file and return its text (Req 10.1 - 10.7).

        Never raises for a filesystem condition: an absent file, a denied read
        and undecodable bytes each have a defined result, because a message
        written for the model is more useful than an exception the dispatcher
        would have to convert (Req 5.7).
        """
        notice = self._parameter_notice(kwargs)

        try:
            raw = self._notes_file.read_bytes()
        except FileNotFoundError:
            # Req 10.2: not an error — an empty history is a normal state at the
            # start of a run, and the model should read it as "nothing saved
            # yet", not as "the tool is broken".
            return self._result(
                "success",
                f"No notes have been saved yet: {self._path_text} does not exist.",
                notice,
            )
        except OSError as exc:
            # Req 10.7: the reason and the path, and no second attempt. One
            # branch for every OS-level failure — denied access, a directory at
            # the path, a vanished mount — because the model's next move is the
            # same in all of them and the reason text carries the difference.
            return self._result(
                "error",
                f"Could not read the notes file {self._path_text}: {_reason(exc)}.",
                notice,
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Req 10.7: undecodable bytes are reported with the offending byte
            # position, which is what lets an operator find and fix the spot.
            return self._result(
                "error",
                (
                    f"Could not read the notes file {self._path_text}: it contains "
                    f"bytes at position {exc.start} that are not valid UTF-8."
                ),
                notice,
            )

        if not text:
            # Req 10.3: distinct from the absent-file message above. The file
            # exists, so a write did happen or the operator created it; saying
            # "does not exist" here would send them looking for the wrong thing.
            return self._result(
                "success",
                f"The notes file {self._path_text} exists but is empty.",
                notice,
            )

        body, omitted, kept_entries = _trim_to_budget(text, self._budget)
        return self._result(
            "success",
            self._render(body, omitted, kept_entries, len(text)),
            notice,
        )

    # -- rendering ------------------------------------------------------------

    @property
    def _path_text(self) -> str:
        """The configured path as written, for every message that names it.

        ``str`` rather than ``repr``: on Windows ``repr`` doubles every
        backslash, and the requirements ask for the configured path, not an
        escaped rendering of it. The dispatcher's scrubbing of absolute paths in
        result text (Req 5.10) applies to whatever this produces.
        """
        return str(self._notes_file)

    def _parameter_notice(self, supplied: dict[str, Any]) -> str | None:
        """The Req 10.5 statement, or ``None`` when no parameter was supplied."""
        if not supplied:
            return None
        names = ", ".join(sorted(supplied))
        return (
            f"read_notes accepts no parameters and reads only the configured notes "
            f"file {self._path_text}; the supplied {_plural('parameter', len(supplied))} "
            f"({names}) {_was_were(len(supplied))} ignored and no other file was read."
        )

    def _render(self, body: str, omitted: int, kept_entries: int, total: int) -> str:
        """Assemble the successful read: header, quoted text, omission marker.

        Only ``body`` goes inside the untrusted-data delimiters (Req 10.1). The
        header and the marker are this tool's own words, and leaving them outside
        the block keeps the rule the system prompt states — everything between
        the markers is data, never an instruction — true of every character
        inside it (Req 12.2, Req 12.8).
        """
        lines: list[str] = []
        if omitted and body and kept_entries == 0:
            lines.append(
                f"Notes from {self._path_text}, showing the end of the file, which "
                f"holds no timestamped entries:"
            )
        elif omitted and body and kept_entries == 1:
            lines.append(f"Notes from {self._path_text}, showing the most recent entry:")
        elif omitted and body:
            lines.append(
                f"Notes from {self._path_text}, showing the {kept_entries} most "
                f"recent entries, oldest first:"
            )
        else:
            lines.append(f"Notes from {self._path_text}:")

        if body:
            lines.append(wrap(body))

        # Req 10.6: appended, so it reads as a note about the text above it
        # rather than as part of that text.
        if omitted and body and kept_entries == 0:
            lines.append(
                f"[The start of the file was omitted to stay within the "
                f"{self._budget}-character read budget: {omitted} of {total} "
                f"characters omitted. The file carries no timestamped entry "
                f"headings, so the text above begins mid-file.]"
            )
        elif omitted and body:
            lines.append(
                f"[Earlier entries were omitted to stay within the "
                f"{self._budget}-character read budget: {omitted} of {total} "
                f"characters omitted. Only whole entries are returned.]"
            )
        elif omitted:
            lines.append(
                f"[All {omitted} characters were omitted: the most recent entry "
                f"alone exceeds the {self._budget}-character read budget, and "
                f"read_notes returns only whole entries. Read the file directly, "
                f"or raise NOTES_READ_BUDGET, to see it.]"
            )
        return "\n".join(lines)

    def _result(self, status: ToolStatus, message: str, notice: str | None) -> ToolResult:
        """One construction point for every result, so the notice cannot be lost.

        ``truncated`` stays ``False`` even when the budget trim dropped entries.
        That flag and ``original_length`` describe the dispatcher's own
        character-level truncation of a result (Req 5.12); budget trimming is a
        different mechanism with its own marker, and conflating the two would
        make a trace record claim a truncation that never happened.
        """
        content = f"{notice}\n{message}" if notice else message
        return ToolResult(status=status, content=content)


# --------------------------------------------------------------------------- #
# Budget trimming
# --------------------------------------------------------------------------- #


def _trim_to_budget(text: str, budget: int) -> tuple[str, int, int]:
    """Return ``(body, omitted_characters, kept_entry_count)`` for ``text``.

    Within budget, ``text`` is returned whole and nothing is omitted (Req 10.1).
    Over budget, the newest whole entries that fit are returned in their
    original order — a contiguous suffix of the file, so the returned text is
    ``text[kept_start:]`` and the omitted count is exactly ``kept_start``
    (Req 10.6).

    Two edge cases are decided here.

    *No entry fits.* ``body`` is empty and every character is counted as
    omitted. The alternative — a partial final entry — is excluded by Req 10.6's
    "whole note entries" and by Property 10.

    *No entry boundary exists anywhere.* A hand-written notes file with no
    timestamp heading has one entry by definition, so the rule above would
    return nothing at all for any file over the budget. That is a useless answer
    to a reasonable file, so this one case falls back to the last ``budget``
    characters, cut at a line boundary where one is available. The marker still
    states the exact number of characters omitted, and the fallback cannot
    affect a file ``save_note`` wrote, because every entry it appends carries a
    heading. The returned entry count is 0 in that case, which is what tells the
    renderer to describe the text as the end of the file rather than as entries.
    """
    if len(text) <= budget:
        return text, 0, len(entry_offsets(text))

    offsets = entry_offsets(text)
    if len(offsets) <= 1 and not ENTRY_BOUNDARY_PATTERN.search(text):
        tail = _line_aligned_tail(text, budget)
        # A kept-entry count of 0 with a non-empty body is how this one case
        # announces itself to the renderer: the text is the end of the file, not
        # a whole entry, and the message has to say so.
        return tail, len(text) - len(tail), 0

    kept_start = len(text)
    kept_entries = 0
    # Walk backwards from the newest entry. The kept text is always the suffix
    # ``text[start:]``, because entries partition the file contiguously, so "do
    # the entries from here on fit" is the single comparison below.
    for start in reversed(offsets):
        if len(text) - start > budget:
            break
        kept_start = start
        kept_entries += 1

    return text[kept_start:], kept_start, kept_entries


def _line_aligned_tail(text: str, budget: int) -> str:
    """The last ``budget`` characters of ``text``, starting at a line if possible.

    Only used for a notes file with no entry boundaries at all. Starting at a
    line boundary keeps the first returned line from beginning mid-word, which
    costs at most one line of text.
    """
    tail = text[len(text) - budget :]
    newline = tail.find("\n")
    if newline != -1 and newline + 1 < len(tail):
        return tail[newline + 1 :]
    return tail


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #


def _reason(exc: BaseException) -> str:
    """A short, operator-readable reason from an OS error."""
    strerror = getattr(exc, "strerror", None)
    return str(strerror) if strerror else (str(exc) or type(exc).__name__)


def _plural(singular: str, count: int, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _was_were(count: int) -> str:
    return "was" if count == 1 else "were"


__all__ = [
    "DESCRIPTION",
    "ENTRY_BOUNDARY_PATTERN",
    "ENTRY_HEADING_PREFIX",
    "ENTRY_TIMESTAMP_PRECISION",
    "ReadNotesTool",
    "entry_heading",
    "entry_offsets",
    "split_entries",
]
