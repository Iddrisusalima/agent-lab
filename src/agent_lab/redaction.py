"""The single redaction function every output path goes through.

Req 1.11 requires that no character sequence equal to the provider API key value
appears on any terminal output line, any trace file line, or anywhere in the eval
report, and that the marker stands at every position where the value would
otherwise have been.

Keeping that guarantee in one function is what makes it checkable: exactly three
call sites wrap it — the renderer's write function (Req 13.9), the trace
serialiser, applied to the already-serialised JSON line (Req 11.5), and the eval
report writer. Nothing else formats output, so nothing else can bypass this.

The replacement is a plain string substitution, never a regular expression: an
API key may contain regex metacharacters, and ``str.replace`` treats the secret
as literal text.
"""

# Req 1.11: the Redaction_Marker. One fixed sequence, used everywhere, so a
# reviewer can grep for it and a property test can assert it appears where the
# secret would have been.
REDACTION_MARKER = "***REDACTED***"


def redact(text: str, secret: str | None, marker: str = REDACTION_MARKER) -> str:
    """Return ``text`` with every occurrence of ``secret`` replaced by ``marker``.

    ``text`` is returned unchanged when ``secret`` is ``None`` or empty. That
    case matters: an empty secret is a substring of every position in every
    string, so substituting it would shred the output rather than redact it.
    Configuration rejects a missing or empty key before a run starts (Req 1.8),
    but this function is also called from the eval writer and from error paths,
    and a sink must never corrupt what it is given.

    The substitution is literal and unanchored, so a secret that appears inside
    a longer token, a JSON string, or a serialised trace line is still caught.
    """
    if not secret:
        return text
    return text.replace(secret, marker)


__all__ = ["REDACTION_MARKER", "redact"]
