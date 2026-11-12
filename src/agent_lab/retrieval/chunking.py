"""The chunker: one pure function, fixed-width character windows (Req 7.2).

``chunk(text, size, overlap)`` cuts a document into windows of at most ``size``
characters, advancing by ``size - overlap`` each time. It reads no
configuration, touches no filesystem, and keeps no state, so the same arguments
always produce the same list — which is what makes an ingested chunk ordering,
and therefore a query ordering, reproducible across builds.

**Why fixed-width windows.** Splitting on sentences or paragraphs would produce
nicer excerpts, but the size of a chunk would then depend on the punctuation of
the document, and Req 7.2 states a hard ceiling: no chunk longer than the
configured target length. Character windows make that ceiling true by
construction rather than by a post-hoc check, and they make the inverse true as
well — dropping the overlap from every chunk after the first and concatenating
reproduces the original text exactly, so nothing in the document is lost or
duplicated into the index.

**Two guarantees the windowing is arranged around.**

* Every chunk but the last is exactly ``size`` characters long; the last one
  ends at the end of the text. The loop stops as soon as a window reaches the
  end rather than continuing to ``len(text)``, so no trailing window repeats
  text that the previous one already covered in full.
* Consecutive chunks share exactly ``overlap`` characters. Window ``i`` starts
  at ``i * step`` and window ``i - 1`` ended at ``(i - 1) * step + size``, which
  is ``i * step + overlap`` — so ``chunks[i][overlap:]`` begins precisely where
  ``chunks[i - 1]`` stopped. The last chunk is strictly longer than ``overlap``
  (if it were not, the previous window would already have reached the end and
  the loop would have stopped), so that reassembly holds for it too.

**Validation comes first, before any work.** Req 7.10 requires an out-of-range
target length or overlap to reject the build and leave an existing persisted
collection unchanged. That is enforced here, at the one place both the loader
and the index call, rather than relying on the caller: :func:`validate_params`
raises :class:`ChunkingError` before a single window is cut, and the index
command can call it on its own to refuse a build before it opens the store.

:class:`~agent_lab.config.Config` already validates ``CHUNK_SIZE`` and
``CHUNK_OVERLAP`` when it loads them (Req 1.9), so a configured run never
reaches the rejection path. The checks are kept anyway: this function is called
directly with integers, and a ceiling that depends on the caller having
validated its arguments is not a ceiling.
"""

from __future__ import annotations

from agent_lab.config import ConfigError

# Req 7.2 / Req 7.10: the inclusive range the configured target chunk length is
# accepted in. The same bounds as ``INT_RANGES["CHUNK_SIZE"]`` in
# ``agent_lab.config``; a test pins the two together so neither can drift.
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4_000


class ChunkingError(ConfigError):
    """The target chunk length or the overlap lies outside its accepted range.

    A subclass of :class:`~agent_lab.config.ConfigError` because it reports the
    same kind of fault — a configured value out of range — and so takes the same
    path out of the CLI: the message is printed to stderr and the process exits
    with ``exit_codes.CONFIGURATION_ERROR`` (Req 7.10). Raised before any
    chunking, any ingestion, and any write to the persistence directory, so an
    existing collection survives a rejected build untouched.
    """


def validate_params(size: object, overlap: object) -> tuple[int, int]:
    """Return ``(size, overlap)`` as validated integers, or raise (Req 7.10).

    Split out from :func:`chunk` so that the ``index`` command can reject a
    build before it opens the Chroma store, which is the difference between
    "rejects the build" and "rejects the build *and* leaves the existing
    collection unchanged".

    ``size`` is checked first: the accepted range for ``overlap`` is stated in
    terms of ``size``, and quoting a bound derived from a nonsense target length
    would only confuse the operator reading the message.

    :raises ChunkingError: if ``size`` is not an integer in
        ``MIN_CHUNK_SIZE``-``MAX_CHUNK_SIZE`` inclusive, or ``overlap`` is not
        an integer in ``0``-``size - 1`` inclusive.
    """
    checked_size = _as_int(
        size,
        name="chunk size",
        accepted=(
            f"an integer between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} "
            f"characters inclusive"
        ),
    )
    if not MIN_CHUNK_SIZE <= checked_size <= MAX_CHUNK_SIZE:
        raise ChunkingError(
            f"chunk size {checked_size} is out of range: accepted values are "
            f"an integer between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} "
            f"characters inclusive"
        )

    accepted_overlap = (
        f"an integer between 0 and {checked_size - 1} inclusive "
        f"(0 to chunk size - 1, with chunk size {checked_size})"
    )
    checked_overlap = _as_int(
        overlap, name="chunk overlap", accepted=accepted_overlap
    )
    if not 0 <= checked_overlap <= checked_size - 1:
        raise ChunkingError(
            f"chunk overlap {checked_overlap} is out of range: accepted values "
            f"are {accepted_overlap}"
        )

    return checked_size, checked_overlap


def chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of at most ``size`` characters.

    Pure and deterministic: no configuration, no filesystem, no state, so equal
    arguments always give an equal list.

    :param text: the extracted text of one document. Returned content is
        verbatim — no stripping, no whitespace collapsing, no normalisation.
        Deciding that a document holds nothing worth indexing belongs to the
        loader (Req 7.9), which has the file name needed to report it.
    :param size: the target chunk length in characters, ``MIN_CHUNK_SIZE`` to
        ``MAX_CHUNK_SIZE`` inclusive. No returned chunk is longer than this.
    :param overlap: the number of characters each chunk repeats from its
        predecessor, ``0`` to ``size - 1`` inclusive.
    :returns: the chunks in document order, so the index of a chunk in this list
        is the zero-based position the loader stores with it (Req 7.2). Empty
        text yields no chunks — an empty document contributes nothing to index.
    :raises ChunkingError: if ``size`` or ``overlap`` is out of range. Raised
        before any chunk is produced (Req 7.10).
    :raises TypeError: if ``text`` is not a string. A caller that passes bytes
        or ``None`` has a decoding fault upstream, and silently coercing it
        would put a repr of an object into the index as if it were a document.
    """
    size, overlap = validate_params(size, overlap)

    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, not {type(text).__name__}"
        )
    if not text:
        return []

    step = size - overlap  # 1 <= step <= size, since 0 <= overlap <= size - 1
    chunks: list[str] = []
    start = 0
    while True:
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            # This window reached the end of the document. Advancing again
            # would only repeat text already covered in full.
            return chunks
        start += step


def _as_int(value: object, *, name: str, accepted: str) -> int:
    """Return ``value`` as an ``int``, or raise :class:`ChunkingError`.

    ``bool`` is rejected even though it is an ``int`` subclass: ``True`` as a
    chunk size is a caller mistake worth naming, not a size of 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChunkingError(
            f"{name} must be {accepted}, not {value!r}"
        )
    return value


__all__ = [
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "ChunkingError",
    "chunk",
    "validate_params",
]
