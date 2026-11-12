"""The keyword fallback: searchable before any index exists (Req 7.6).

``KeywordRetriever`` answers a query by reading the corpus files, cutting them
with the same chunker the index uses, and scoring each chunk by how often the
query's terms occur in it, plus a bonus when the whole query appears as a
substring. It imports no third-party search package and needs no persisted
collection, so a reviewer who has cloned the repository and set only an API key
can still ask a question about the sample notes and get grounded passages back.

**Where it sits.** It satisfies :class:`~agent_lab.retrieval.base.Retriever` —
``search(query, k) -> list[(chunk_text, source_name)]`` and nothing else — so
``search_docs`` cannot tell it apart from the Chroma retriever (Req 7.7).
Selecting between the two is the wiring layer's decision, made once at startup
on whether the persistence directory holds a collection; this module neither
knows nor asks about Chroma.

**How it says it was used.** Req 7.6 requires the tool result to state that
keyword fallback search was used. That is the :attr:`KeywordRetriever.notice`
attribute, the optional extra described in
:mod:`agent_lab.retrieval.base`: ``search_docs`` reads ``notice`` with
``getattr`` and includes it when it is a non-empty string, so the notice travels
with the retriever that earned it rather than being re-derived by the tool.

**What it shares with the index.** The corpus walk
(:func:`~agent_lab.retrieval.base.corpus_files`, Req 7.1), the chunker
(:func:`~agent_lab.retrieval.chunking.chunk`, Req 7.2), and the tie-breaking
rule ``(-similarity, source_name, position)`` (Req 7.5). Sharing the first two
is what makes Req 8.9 hold for free: every source name this retriever returns is
the corpus-relative path of a file that the walk just found, so the check
``search_docs`` performs on its results cannot drop one of them as stale.

**What it does not share.** The loader's per-file reporting (Req 7.9). A file
that cannot be opened, is larger than 25 MB, or yields only whitespace is
skipped here too, by the same rules, but silently — the fallback runs inside a
tool call where a printed line would arrive in the middle of the agent's
narration, and the ``index`` command is the place that owes the operator a
skipped-file report. When the corpus loader is wired in, pass it as
``chunk_source`` and this module will use its chunks instead of walking the
corpus itself; the scoring, ordering and caching below are unchanged by that
substitution.

**Why text is re-read on demand rather than at construction.** A retriever is
built during startup wiring, before it is known whether the run will search at
all. Reading the corpus then would charge every run — a calculator-only question
included — for I/O it may never use. The chunks are therefore built on the first
``search`` and cached behind a fingerprint of the corpus (path, size,
modification time) and the chunk settings, so repeated searches in one run cost
one ``stat`` per file, and a corpus edited mid-run is still picked up.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from agent_lab.config import Config
from agent_lab.retrieval.base import corpus_files
from agent_lab.retrieval.chunking import chunk as chunk_document
from agent_lab.retrieval.chunking import validate_params

# --------------------------------------------------------------------------- #
# The accepted parameter ranges (Req 7.5, Req 7.11)
# --------------------------------------------------------------------------- #

# Req 7.5: the query length the retrieval layer accepts. Wider than the 500
# characters ``search_docs`` admits (Req 8.1), because this is the retrieval
# contract and a substituted caller may pass more.
MIN_QUERY_CHARS = 1
MAX_QUERY_CHARS = 4_000

# Req 7.5 / Req 7.11: the accepted result count. Also wider than the tool's own
# 1-10 clamp (Req 8.8) for the same reason.
MIN_RESULT_COUNT = 1
MAX_RESULT_COUNT = 100

# Req 7.9: the per-file size ceiling. A file above it is skipped, which keeps a
# corpus directory pointed at a disk image from making one tool call unbounded.
MAX_FILE_BYTES = 25 * 1024 * 1024

# Req 7.6: the sentence ``search_docs`` includes in the result so the operator
# and the model both know these passages came from substring matching and not
# from the vector index. One sentence, because it is counted against the tool
# result's 4000-character budget (Req 8.3).
NOTICE = (
    "Keyword fallback search was used: no document index has been built, so "
    "these passages were found by matching the query's words against the text "
    "of the files in the document directory rather than by semantic "
    "similarity — run the index command for better matches."
)

# --------------------------------------------------------------------------- #
# Scoring weights
# --------------------------------------------------------------------------- #
#
# The three weights sum to 1.0, so a score is a fraction between 0.0 and 1.0.
# Their absolute values never leave this module — ``search`` returns pairs, not
# scores, and only the induced *order* is observable — but a bounded scale keeps
# the three components comparable with one another and makes a score readable in
# a test failure.

# How many of the query's distinct terms occur in the chunk at all. Weighted
# highest: a chunk mentioning every term once is a better answer than one
# mentioning a single term ten times.
COVERAGE_WEIGHT = 0.6

# How often they occur, saturating at TF_SATURATION occurrences per term so that
# one term repeated throughout a chunk cannot outweigh coverage of the rest.
FREQUENCY_WEIGHT = 0.2
TF_SATURATION = 5

# The whole query appearing verbatim, after whitespace collapsing and case
# folding — the substring half of the search, and the only part that can tell
# "retention policy" from a chunk that happens to contain both words far apart.
PHRASE_WEIGHT = 0.2

# Word-ish runs: letters, digits and marks, with underscores and punctuation
# acting as separators. ``\w`` minus ``_`` keeps "well-known" two terms and
# leaves non-ASCII words intact, which a plain ``[a-z0-9]+`` would cut up.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_WHITESPACE_RE = re.compile(r"\s+")

# Terms that occur in nearly every document and so separate nothing. Dropped
# from the term list only when at least one other term survives, so a query that
# is nothing but stop words ("what is it") still searches for them instead of
# silently matching everything. They remain part of the phrase, which is the
# point of the phrase bonus: "the cost of the run" should reward a chunk holding
# that exact wording.
_STOP_WORDS = frozenset(
    {
        "a", "about", "all", "an", "and", "any", "are", "as", "at", "be",
        "been", "but", "by", "can", "do", "does", "for", "from", "has", "have",
        "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no",
        "not", "of", "on", "or", "our", "should", "so", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "to", "up",
        "was", "we", "were", "what", "when", "where", "which", "who", "why",
        "will", "with", "would", "you", "your",
    }
)


class RetrievalError(ValueError):
    """A retrieval parameter lies outside its accepted range (Req 7.11).

    Raised before the corpus is read, so a rejected call performs no retrieval
    and returns no chunk. The message names the offending parameter and its
    permitted range.

    Raising rather than returning is what keeps :class:`Retriever` at a single
    operation returning a list of pairs: there is no error channel in the return
    type to put this in. ``search_docs`` catches it and says the search was not
    performed, which is the tool-level half of Req 7.11.

    A :class:`ValueError`, because an out-of-range argument is what that type is
    for, and because a caller that wants to treat it as one — a future retrieval
    implementation sharing this contract — need not import this module.
    """


class _Chunk(NamedTuple):
    """One searchable unit: its text, its source, and where in the source.

    The same three fields the loader's ``Chunk`` carries (Req 7.2), by the same
    names, so an injected ``chunk_source`` producing loader chunks drops in
    without conversion.

    ``source_name`` is the corpus-relative path with forward slashes, not the
    bare file name: two files called ``notes.md`` in different subdirectories
    have to be distinguishable in a result, and ``search_docs`` resolves a
    relative path against the corpus directory when it checks Req 8.9.
    """

    text: str
    source_name: str
    position: int


# --------------------------------------------------------------------------- #
# Scoring — pure functions, so the ranking is testable without a filesystem
# --------------------------------------------------------------------------- #


def normalise(text: str) -> str:
    """Case-fold ``text`` and collapse every whitespace run to one space.

    Applied to both sides of every comparison. Collapsing is what lets a phrase
    match across the line break a chunk boundary or a wrapped paragraph put in
    the middle of it, and case folding is why ``Retention`` answers a query for
    ``retention``.
    """
    return _WHITESPACE_RE.sub(" ", text.casefold()).strip()


def query_terms(query: str) -> list[str]:
    """The distinct terms of ``query``, in order of first appearance.

    Duplicates are dropped because coverage counts terms, not mentions: a query
    repeating a word does not make a chunk containing it twice as relevant.
    Stop words are dropped only when something else remains, so a query made
    entirely of them is still searched for rather than matching everything.
    """
    seen: list[str] = []
    for token in _TOKEN_RE.findall(normalise(query)):
        if token not in seen:
            seen.append(token)
    content = [term for term in seen if term not in _STOP_WORDS]
    return content or seen


def score_chunk(text: str, terms: Sequence[str], phrase: str) -> float:
    """Score ``text`` against pre-normalised ``terms`` and ``phrase``.

    Pure, and takes the normalised query parts rather than the query, so the
    tokenising and case folding happen once per search instead of once per
    chunk.

    :param text: the chunk text, as stored. Normalised here.
    :param terms: the distinct query terms from :func:`query_terms`.
    :param phrase: the whole query from :func:`normalise`.
    :returns: ``0.0`` when nothing matched — the caller drops those, which is
        what makes "no matching content" (Req 8.5) an empty list rather than a
        list of irrelevant passages — otherwise a value in ``(0.0, 1.0]``.

    Term occurrences are counted as substrings, not whole words, so a query for
    ``policy`` scores a chunk that says ``policies``. The cost of that choice is
    that short terms match inside longer unrelated words; coverage weighting
    limits the damage, since a chunk has to hit several terms to rank well.
    """
    haystack = normalise(text)
    if not haystack:
        return 0.0

    phrase_hit = bool(phrase) and phrase in haystack

    if not terms:
        # A query with no word characters at all — punctuation, symbols, a bare
        # number sign. Req 7.11 rejects only an empty or whitespace-only query,
        # so this is a legitimate search: fall back to the substring half alone.
        return PHRASE_WEIGHT if phrase_hit else 0.0

    present = 0
    frequency = 0.0
    for term in terms:
        count = haystack.count(term)
        if count:
            present += 1
            frequency += min(count, TF_SATURATION) / TF_SATURATION

    if not present and not phrase_hit:
        return 0.0

    coverage = present / len(terms)
    return (
        COVERAGE_WEIGHT * coverage
        + FREQUENCY_WEIGHT * (frequency / len(terms))
        + (PHRASE_WEIGHT if phrase_hit else 0.0)
    )


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #


class KeywordRetriever:
    """Substring and term-frequency search over the corpus (Req 7.6).

    Satisfies :class:`~agent_lab.retrieval.base.Retriever`, and additionally
    exposes :attr:`notice` so ``search_docs`` can state that the fallback was
    used.

    The corpus directory and the chunk settings are constructor state taken from
    :class:`~agent_lab.config.Config`. Neither is a search argument, so no
    model-supplied value can point the search at another directory (Req 12.9).
    """

    def __init__(
        self,
        config: Config,
        *,
        chunk_source: Callable[[], Sequence[Any]] | None = None,
        notice: str = NOTICE,
    ) -> None:
        """Wire the retriever, validating the chunk settings now rather than later.

        :param config: supplies ``docs_dir``, ``chunk_size`` and
            ``chunk_overlap``. Only those three are kept.
        :param chunk_source: optional callable returning the chunks to search,
            for wiring the corpus loader in once it exists. Each item needs
            ``text``, ``source_name`` and ``position`` attributes, or may be a
            ``(text, source_name, position)`` tuple; an item of any other shape
            is ignored. When absent, this retriever walks the corpus itself.
        :param notice: the Req 7.6 sentence, overridable so a caller that
            selected this retriever for some other reason can say which.
        :raises ChunkingError: if the configured chunk size or overlap is out of
            range (Req 7.10). Raised at construction, so a misconfigured
            retriever fails during startup wiring rather than inside a tool
            call three steps into a run.
        """
        self._docs_dir: Path = config.docs_dir
        self._chunk_size, self._chunk_overlap = validate_params(
            config.chunk_size, config.chunk_overlap
        )
        self._chunk_source = chunk_source
        self.notice = notice

        # The chunk cache and the corpus fingerprint it was built from. Both
        # replaced together, so a stale cache cannot outlive its fingerprint.
        self._chunks: list[_Chunk] = []
        self._fingerprint: tuple[Any, ...] | None = None

    # -- the protocol operation ----------------------------------------------

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        """Return at most ``k`` ``(chunk_text, source_name)`` pairs (Req 7.5).

        Ordered by descending score, with ties broken by ascending source name
        then ascending chunk position, so two identical queries over an
        unchanged corpus return the same pairs in the same order.

        Chunks that score zero are excluded rather than returned to fill ``k``:
        a passage that does not match is worse than no passage, because the
        model would have to decide to ignore it, and Req 8.5 wants "no matching
        content" to be sayable.

        :raises RetrievalError: if ``query`` is empty, whitespace-only or longer
            than :data:`MAX_QUERY_CHARS`, or ``k`` is not an integer in
            :data:`MIN_RESULT_COUNT`-:data:`MAX_RESULT_COUNT`. Raised before the
            corpus is read, so no retrieval is performed (Req 7.11).
        """
        checked_query = self._checked_query(query)
        count = self._checked_count(k)

        terms = query_terms(checked_query)
        phrase = normalise(checked_query)

        scored: list[tuple[float, str, int, str]] = []
        for candidate in self._searchable_chunks():
            score = score_chunk(candidate.text, terms, phrase)
            if score > 0.0:
                # Negated score first: one ascending sort then expresses the
                # whole rule — descending similarity, ascending source name,
                # ascending position (Req 7.5).
                scored.append(
                    (-score, candidate.source_name, candidate.position, candidate.text)
                )

        scored.sort(key=lambda row: (row[0], row[1], row[2]))
        return [(text, source) for _, source, _, text in scored[:count]]

    # -- parameter validation (Req 7.11) -------------------------------------

    def _checked_query(self, query: object) -> str:
        """Return ``query`` as a usable query string, or raise (Req 7.11)."""
        accepted = (
            f"a string of {MIN_QUERY_CHARS} to {MAX_QUERY_CHARS} characters that "
            f"is not only whitespace"
        )
        if not isinstance(query, str):
            raise RetrievalError(
                f"query must be {accepted}, not {type(query).__name__}; "
                f"no retrieval was performed"
            )
        if not query.strip():
            raise RetrievalError(
                f"query must be {accepted}, but the supplied query is "
                f"{'empty' if not query else 'only whitespace'}; "
                f"no retrieval was performed"
            )
        if len(query) > MAX_QUERY_CHARS:
            raise RetrievalError(
                f"query must be {accepted}, but the supplied query is "
                f"{len(query)} characters; no retrieval was performed"
            )
        return query.strip()

    def _checked_count(self, k: object) -> int:
        """Return ``k`` as a usable result count, or raise (Req 7.11).

        Rejected rather than clamped: clamping belongs to ``search_docs``, which
        has a model to be forgiving towards (Req 8.8). At this layer the caller
        is code, and code that asks for 0 or 1000 results has a defect worth
        surfacing.

        ``bool`` is refused even though it is an ``int`` subclass, for the same
        reason the rest of the package refuses it: ``True`` as a result count is
        a caller mistake, not a request for one chunk.
        """
        accepted = (
            f"an integer between {MIN_RESULT_COUNT} and {MAX_RESULT_COUNT} inclusive"
        )
        if isinstance(k, bool) or not isinstance(k, int):
            raise RetrievalError(
                f"k (the result count) must be {accepted}, not "
                f"{type(k).__name__}; no retrieval was performed"
            )
        if not MIN_RESULT_COUNT <= k <= MAX_RESULT_COUNT:
            raise RetrievalError(
                f"k (the result count) must be {accepted}, received {k}; "
                f"no retrieval was performed"
            )
        return k

    # -- the chunk cache ------------------------------------------------------

    def _searchable_chunks(self) -> list[_Chunk]:
        """The corpus chunks, rebuilt only when the corpus or settings changed.

        The fingerprint is one ``stat`` per corpus file plus the chunk settings.
        Cheap next to reading and chunking every file, and it means a corpus
        edited between two steps of the same run is searched as it is now, not
        as it was when the first search happened.
        """
        paths = corpus_files(self._docs_dir)
        fingerprint = self._corpus_fingerprint(paths)
        if fingerprint != self._fingerprint:
            self._chunks = self._build_chunks(paths)
            self._fingerprint = fingerprint
        return self._chunks

    def _corpus_fingerprint(self, paths: Sequence[Path]) -> tuple[Any, ...]:
        """A value that changes whenever the searchable text could have changed.

        Size and modification time, not content: hashing every byte of the
        corpus on every search would cost more than the search. A file rewritten
        within the filesystem's timestamp resolution and to exactly its previous
        length would be missed; for a fallback searcher over an operator's notes
        that is an acceptable trade, and the ``index`` command is the supported
        way to force a rebuild.

        A file that cannot be ``stat``-ed contributes a distinct marker rather
        than raising, so one unreadable file does not make every search fail.
        """
        entries: list[tuple[Any, ...]] = []
        for path in paths:
            try:
                info = path.stat()
            except OSError:
                entries.append((str(path), None, None))
            else:
                entries.append((str(path), info.st_size, info.st_mtime_ns))
        return (self._chunk_size, self._chunk_overlap, tuple(entries))

    def _build_chunks(self, paths: Sequence[Path]) -> list[_Chunk]:
        """Read, extract and chunk the corpus, skipping what cannot be read.

        Document order is the order :func:`~agent_lab.retrieval.base.corpus_files`
        returns — sorted at every level — and chunk order within a document is
        the chunker's, so the built list is identical for an identical corpus.
        Nothing here depends on that for correctness, since the result is sorted
        by ``(-score, source_name, position)`` anyway, but a reproducible build
        makes a ranking difference traceable to a content difference.
        """
        if self._chunk_source is not None:
            return _coerce_chunks(self._chunk_source())

        chunks: list[_Chunk] = []
        for path in paths:
            text = _extract_text(path)
            if text is None:
                # Req 7.9's skip rules, applied silently here — see the module
                # docstring on why the report belongs to the index command.
                continue
            source_name = self._source_name(path)
            for position, body in enumerate(
                chunk_document(text, self._chunk_size, self._chunk_overlap)
            ):
                chunks.append(_Chunk(body, source_name, position))
        return chunks

    def _source_name(self, path: Path) -> str:
        """The corpus-relative path of ``path``, with forward slashes (Req 8.9).

        Relative so that two same-named files in different subdirectories stay
        distinguishable, and forward-slashed so a result reads the same on
        Windows as on anything else. ``search_docs`` resolves this spelling
        against the corpus when it checks that every quoted source is a file
        that is really there.

        Falls back to the bare file name if the path somehow lies outside the
        corpus directory, which ``corpus_files`` makes unreachable but a future
        ``chunk_source`` need not.
        """
        try:
            relative = path.relative_to(self._docs_dir)
        except ValueError:  # pragma: no cover - corpus paths come from docs_dir
            return path.name
        return relative.as_posix()


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #


def _extract_text(path: Path) -> str | None:
    """The text of one corpus file, or ``None`` if it must be skipped (Req 7.9).

    ``None`` for the three skip conditions: the file cannot be opened or read,
    it exceeds :data:`MAX_FILE_BYTES`, or it yields only whitespace after
    extraction. Returning a sentinel rather than raising keeps one unreadable
    file from ending a search over the rest of the corpus.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_FILE_BYTES:
        return None

    if path.suffix.lower() == ".pdf":
        text = _pdf_text(path)
    else:
        text = _decoded_text(path)

    if text is None or not text.strip():
        return None
    return text


def _decoded_text(path: Path) -> str | None:
    """A ``.txt`` or ``.md`` file decoded as UTF-8, or ``None`` if unreadable.

    A byte sequence that is not valid UTF-8 is decoded again with replacement
    rather than skipped. The loader (Req 7.9) is stricter because it is building
    a persisted collection and owes the operator a report; this searcher is
    answering a question now, and a file with one stray byte from a legacy
    encoding is still worth matching against.
    """
    try:
        raw = path.read_bytes()
    except (OSError, ValueError):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _pdf_text(path: Path) -> str | None:
    """A PDF's text via ``pypdf``, joining per-page extraction, or ``None``.

    ``pypdf`` is imported here rather than at module import time for two
    reasons: this module is the dependency-free fallback and should import on a
    checkout where the optional stack is not installed, and a run that never
    searches a PDF should not pay for loading a PDF parser.

    Every failure — an absent ``pypdf``, an encrypted or malformed document, a
    page that cannot be extracted — is a skip. A corpus is a directory of files
    an operator dropped in, not a validated input, and one bad PDF must not stop
    a search over the other files.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - deliberately lazy
    except Exception:  # noqa: BLE001 - absent or broken install, same outcome
        return None

    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - one unreadable page, not the file
                continue
    except Exception:  # noqa: BLE001 - encrypted, truncated, not a PDF at all
        return None
    return "\n".join(pages)


def _coerce_chunks(items: Any) -> list[_Chunk]:
    """Read an injected ``chunk_source``'s output as :class:`_Chunk` values.

    Accepts anything with ``text``, ``source_name`` and ``position``, which is
    the loader's chunk shape (Req 7.2), or a three-element tuple. Items of any
    other shape are dropped rather than raising: a ``chunk_source`` is a wiring
    convenience, and a search is a better outcome than an exception from inside
    a tool call.
    """
    chunks: list[_Chunk] = []
    if items is None:
        return chunks
    try:
        iterator = iter(items)
    except TypeError:
        return chunks

    for item in iterator:
        text = getattr(item, "text", None)
        source = getattr(item, "source_name", None)
        position = getattr(item, "position", None)
        if text is None and isinstance(item, (list, tuple)) and len(item) == 3:
            text, source, position = item
        if not isinstance(text, str) or not isinstance(source, str):
            continue
        if isinstance(position, bool) or not isinstance(position, int):
            position = 0
        chunks.append(_Chunk(text, source, position))
    return chunks


__all__ = [
    "COVERAGE_WEIGHT",
    "FREQUENCY_WEIGHT",
    "MAX_FILE_BYTES",
    "MAX_QUERY_CHARS",
    "MAX_RESULT_COUNT",
    "MIN_QUERY_CHARS",
    "MIN_RESULT_COUNT",
    "NOTICE",
    "PHRASE_WEIGHT",
    "TF_SATURATION",
    "KeywordRetriever",
    "RetrievalError",
    "normalise",
    "query_terms",
    "score_chunk",
]
