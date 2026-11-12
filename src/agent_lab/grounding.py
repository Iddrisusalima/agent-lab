"""Untrusted-data delimiters and the sanitiser that protects them.

Req 12.8 requires that text produced by ``search_docs`` or ``read_notes`` be
enclosed in the Untrusted_Data_Delimiters before it enters the conversation,
with any occurrence of either delimiter inside that text escaped or removed.
Req 8.4 and Req 10.1 require the same enclosure at the tools themselves, which
is why ``sanitize`` is idempotent: the loop may wrap text a tool already
wrapped, and the second pass must change nothing.

The threat this module answers is narrow and concrete. The System_Prompt tells
the model that anything between the markers is data to summarise and never an
instruction to follow (Req 12.2). A document that could write the end marker
into its own body would appear to close the quoted block early, so the text
after it would read as trusted conversation. Removing the marker from document
text is what keeps the block boundary meaningful.

Matching is therefore deliberately wider than an exact string compare. A forged
marker is caught when it differs from the canonical form only by letter case or
by internal whitespace — ``<<< begin document data >>>`` and a marker split
across two lines are both neutralised — because the model reads those the same
way a reviewer does. The replacement text contains no angle brackets, so a
substitution can never produce a new marker, which is what makes repeated
application a fixed point.
"""

import re

# Req 12.8 / the Untrusted_Data_Delimiters of the glossary. One canonical pair,
# used by the loop and by both document tools, so a reviewer can grep for them.
BEGIN = "<<<BEGIN DOCUMENT DATA>>>"
END = "<<<END DOCUMENT DATA>>>"

# What a stripped marker leaves behind. Visible, so a reader of a trace can see
# that something was removed rather than silently losing text, and free of
# angle brackets, so it cannot combine with surrounding characters to form a
# marker on a later pass.
REMOVED_MARKER = "[delimiter removed]"

# A marker is fenced by ``<<<`` and ``>>>`` with words in between; this splits a
# canonical marker into those parts so the matching pattern is derived from the
# constants above and cannot drift out of step with them.
_FENCE_OPEN = "<<<"
_FENCE_CLOSE = ">>>"


def _pattern_for(delimiter: str) -> re.Pattern[str]:
    """Build the pattern matching ``delimiter`` up to case and whitespace.

    The inner words are joined with ``\\s*`` rather than ``\\s+`` so that the
    match covers every whitespace variant of the marker, including the run-free
    one. Collapsing whitespace runs in the text and comparing is equivalent for
    the forms the requirements name, and this form additionally refuses the
    squeezed spelling, which a model would still read as the marker.
    """
    inner = delimiter.removeprefix(_FENCE_OPEN).removesuffix(_FENCE_CLOSE)
    words = inner.split()
    body = r"\s*".join(re.escape(word) for word in words)
    source = (
        re.escape(_FENCE_OPEN)
        + r"\s*"
        + body
        + r"\s*"
        + re.escape(_FENCE_CLOSE)
    )
    # Req 12.8: case-insensitive, and DOTALL is irrelevant here because every
    # wildcard is an explicit whitespace class, which already spans newlines.
    return re.compile(source, re.IGNORECASE)


_DELIMITER_PATTERNS = (_pattern_for(BEGIN), _pattern_for(END))


def sanitize(text: str) -> str:
    """Return ``text`` with every forged delimiter replaced by a notice.

    Both markers are removed, matched without regard to letter case and without
    regard to how much whitespace separates their words, so a document cannot
    close its own quoted block (Req 12.8).

    The substitution is repeated until the text stops changing. A single pass is
    enough for every input that does not nest one marker inside another, but
    nesting is exactly what an attacker would try, and iterating to a fixed
    point is cheap: ``REMOVED_MARKER`` introduces no ``<<<``, so each pass that
    changes anything strictly reduces the number of opening fences and the loop
    terminates.

    Applying this function to its own output returns that output unchanged,
    which is what lets the loop wrap a result a tool already wrapped (Req 8.4,
    Req 10.1).
    """
    while True:
        cleaned = text
        for pattern in _DELIMITER_PATTERNS:
            cleaned = pattern.sub(REMOVED_MARKER, cleaned)
        if cleaned == text:
            return cleaned
        text = cleaned


def wrap(text: str) -> str:
    """Return ``text`` sanitised and enclosed in the delimiters.

    The begin marker opens the result and the end marker closes it, each alone
    on its line, so the boundary of the quoted block is unambiguous even when
    the document text itself ends mid-line (Req 8.4, Req 10.1, Req 12.8). The
    enclosed text is otherwise untouched: only marker occurrences are altered,
    because a summariser needs the document's own wording.
    """
    return f"{BEGIN}\n{sanitize(text)}\n{END}"


__all__ = ["BEGIN", "END", "REMOVED_MARKER", "sanitize", "wrap"]
