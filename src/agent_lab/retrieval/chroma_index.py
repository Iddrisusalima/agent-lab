"""The Chroma index: one persistent collection, built once, queried per step.

Two halves, both of them thin:

* :func:`build_index` — the ``index`` command's work. Validate the chunk
  settings, open the persistent store under the configured directory, read the
  corpus through :func:`~agent_lab.retrieval.loader.load_chunks`, **replace**
  any existing collection, and write every chunk with its source name and
  zero-based position as metadata (Req 7.2, 7.3, 7.4).
* :class:`ChromaRetriever` — a :class:`~agent_lab.retrieval.base.Retriever`
  over that collection: ``search(query, k) -> [(chunk_text, source_name)]`` and
  nothing more, so ``search_docs`` cannot tell it apart from the keyword
  fallback (Req 7.7).

**What this module does not own.** It does not walk the corpus, extract PDF
text, decide which files are skipped, or cut documents into windows: the loader
and the chunker do all of that, and this module ingests whatever they hand it.
So the vector index and the keyword fallback see exactly the same corpus, the
same chunk boundaries and the same source-name spelling, and Req 7.8 ("every
returned chunk carries a source file name that matches a file ingested from the
corpus") holds for both without a second implementation to keep in step.

**Ordering is decided here, not by the store** (Req 7.5). Chroma returns
candidates ordered by its own distance and says nothing about how it breaks a
tie. Two chunks with identical text — easy to produce with overlapping windows
over a repetitive document — would then be free to swap places between two
identical queries. So the returned candidates are re-sorted in this module by
``(-similarity, source_name, position)``, the same rule
:mod:`agent_lab.retrieval.keyword` applies, which makes repeated identical
queries over an unchanged collection return an identical list.

**Embeddings come from Chroma's bundled default function.** No embedding
provider enters the dependency list, and no API key is needed to build an
index. The cost of that choice is that the default function fetches its model
the first time it embeds anything, which is why:

**``chromadb`` is imported lazily, and every Chroma failure becomes one typed
error.** The import sits inside the functions that need it, so
``import agent_lab.retrieval.chroma_index`` costs nothing and works on a
checkout where ``chromadb`` is not installed — the keyword fallback is the
product's floor, and reaching it must not require the package whose absence
made it necessary. Anything that makes the Chroma path unusable — the package
missing, the store unopenable, the embedding model undownloadable because the
machine is offline — is raised as :class:`ChromaUnavailable`, a single type a
caller can catch to choose :class:`~agent_lab.retrieval.keyword.KeywordRetriever`
instead. In particular, an embedding attempt with no network reaches this
module as whatever the socket layer raised and leaves it as
``ChromaUnavailable`` with that exception as its ``__cause__``.

``search_docs`` already catches anything a retriever raises and reports that
the search was not performed (Req 7.11's tool-level half), so a collection that
turns unusable mid-run costs one tool call and a legible message rather than
the run.

**Selection** is :func:`select_retriever`: the Chroma retriever when the
persistence directory holds a non-empty collection, the keyword fallback
otherwise (Req 7.6). It is cheap on the fallback path — the absence of the
store file is checked before ``chromadb`` is imported and before any client is
constructed, so a run on a machine that never built an index neither pays for
the import nor leaves an empty ``.chroma`` directory behind.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import IO, Any, Callable, Sequence, TypeVar

from agent_lab.config import Config
from agent_lab.retrieval.base import MAX_CORPUS_FILES, Retriever
from agent_lab.retrieval.chunking import validate_params
from agent_lab.retrieval.keyword import (
    MAX_QUERY_CHARS,
    MAX_RESULT_COUNT,
    MIN_QUERY_CHARS,
    MIN_RESULT_COUNT,
    KeywordRetriever,
    RetrievalError,
)
from agent_lab.retrieval.loader import BuildReport, Chunk, load_chunks

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The one collection this project uses (Req 7.3). Fixed rather than configured:
#: the persistence directory is already configurable, two collections in one
#: store would have no reader, and a name that can vary is a name that can be
#: misspelt into a silently empty search.
COLLECTION_NAME = "agent_lab_documents"

#: The file a ``PersistentClient`` creates in its directory. Its presence is how
#: :func:`store_exists` answers "does this directory hold a Chroma store?"
#: without importing ``chromadb`` and without creating the store as a side
#: effect of asking — ``PersistentClient(path=...)`` creates the directory, so
#: probing by constructing one would make every run leave a ``.chroma`` behind.
STORE_FILE = "chroma.sqlite3"

#: Req 7.5's five seconds, enforced rather than assumed. A vector query over a
#: local store takes milliseconds; this bound exists for the pathological case
#: (a store on a stalled network share, a corrupt index that spins) where the
#: alternative is a run that never finishes a step.
QUERY_TIMEOUT_SECONDS = 5.0

#: Chunks per ``add`` call. Chroma caps a single batch — ``get_max_batch_size()``
#: reports the limit, which this module asks for and respects — and embedding a
#: whole corpus in one call would also hold every vector in memory at once.
#: 1,000 chunks is roughly a megabyte of text per batch.
ADD_BATCH_SIZE = 1_000

#: What a built index's metadata records about its own build. Written as
#: collection metadata so the chunk settings that produced a collection travel
#: with it: an index built at 1,000 characters and queried after the
#: configuration moved to 400 is still perfectly usable, but an operator
#: comparing two result sets can see which it was.
_SIZE_KEY = "chunk_size"
_OVERLAP_KEY = "chunk_overlap"

_T = TypeVar("_T")


class ChromaUnavailable(RuntimeError):
    """The Chroma retrieval path cannot be used; the keyword fallback can.

    One type for every way the vector path fails, because every one of them has
    the same remedy from a caller's point of view — use
    :class:`~agent_lab.retrieval.keyword.KeywordRetriever`, or tell the operator
    to run the ``index`` command:

    * ``chromadb`` is not installed.
    * The persistence directory holds no store, or no collection in it.
    * The store cannot be opened: a corrupt database, a directory that cannot be
      written, a version of the on-disk format this build does not read.
    * The default embedding function cannot embed, which on a machine with no
      network access is the usual case for a first build.
    * A query did not return within :data:`QUERY_TIMEOUT_SECONDS` (Req 7.5).

    A :class:`RuntimeError` rather than a
    :class:`~agent_lab.config.ConfigError`: nothing the operator configured is
    wrong, so this is not a startup refusal and must not be reported as exit
    code 6. The underlying exception is always attached as ``__cause__``.
    """


# --------------------------------------------------------------------------- #
# Opening the store
# --------------------------------------------------------------------------- #


def store_exists(chroma_dir: Path) -> bool:
    """Does ``chroma_dir`` hold a Chroma persistent store?

    True when the directory contains :data:`STORE_FILE`. Deliberately a
    filesystem question and not a Chroma question: it is asked on the startup
    path of every run, where importing ``chromadb`` to learn that there is
    nothing to import it for would be the most expensive line in the wiring.

    It answers "a store exists", not "an index exists" — a store whose
    collection was deleted still has the file. :func:`collection_size` is the
    question with the real answer, and this is its cheap precondition.
    """
    try:
        return (chroma_dir / STORE_FILE).is_file()
    except OSError:
        # An unreadable or unstat-able path holds nothing this process can
        # search. Reported as "no store" so the fallback is selected.
        return False


def _import_chromadb() -> Any:
    """Import ``chromadb``, or raise :class:`ChromaUnavailable` (lazy on purpose).

    Called from inside every function that needs the package, so this module
    imports on a checkout where ``chromadb`` is absent and the keyword fallback
    stays reachable. ``ImportError`` is the expected failure; a broken partial
    install can raise almost anything else, and both mean the same thing here.
    """
    try:
        import chromadb  # noqa: PLC0415 - deliberately lazy; see the docstring
    except Exception as exc:  # noqa: BLE001 - absent or broken install, one outcome
        raise ChromaUnavailable(
            f"the chromadb package could not be imported "
            f"({type(exc).__name__}: {exc}), so the document index cannot be "
            f"used; keyword fallback search is available without it"
        ) from exc
    return chromadb


def _client(chroma_dir: Path, *, create: bool) -> Any:
    """A ``PersistentClient`` over ``chroma_dir`` (Req 7.3).

    :param create: when true, the directory is created if absent, which is what
        Req 7.3 requires of a build. When false, an absent store raises instead,
        so a *query* never creates an empty persistence directory as a side
        effect of looking for a collection in it.

    ``PersistentClient`` creates the directory and the database itself, so there
    is no ``mkdir`` here: one component owns that, and it is Chroma's.
    """
    if not create and not store_exists(chroma_dir):
        raise ChromaUnavailable(
            f"no document index was found at {chroma_dir}: run the index "
            f"command to build one, or use keyword fallback search"
        )

    chromadb = _import_chromadb()
    try:
        return chromadb.PersistentClient(path=str(chroma_dir))
    except Exception as exc:  # noqa: BLE001 - Req 7.3's directory is not ours to repair
        raise ChromaUnavailable(
            f"the document index at {chroma_dir} could not be opened "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _collection_names(client: Any) -> set[str]:
    """The names of the collections in ``client``, across Chroma versions.

    ``list_collections`` returns collection objects in some versions and bare
    names in others. Reading ``name`` with a fallback to the value itself covers
    both, which matters because the alternative — calling ``get_collection`` and
    catching "not found" — differs between versions in the *exception type* it
    raises, and that is harder to be right about than two shapes of string.
    """
    try:
        listed = client.list_collections()
    except Exception as exc:  # noqa: BLE001 - an unreadable store, not a missing name
        raise ChromaUnavailable(
            f"the collections in the document index could not be listed "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return {str(getattr(item, "name", item)) for item in listed or ()}


def _collection(client: Any, name: str, *, metadata: dict[str, Any] | None = None) -> Any:
    """``get_or_create_collection(name)``, with failures typed.

    Creating a collection does not embed anything, so this succeeds on a machine
    with no network. The embedding model is fetched by the first ``add`` or
    ``query``, and that is where an offline machine finds out.
    """
    try:
        if metadata:
            return client.get_or_create_collection(name=name, metadata=dict(metadata))
        return client.get_or_create_collection(name=name)
    except Exception as exc:  # noqa: BLE001 - one type out, per the module docstring
        raise ChromaUnavailable(
            f"the collection {name!r} in the document index could not be opened "
            f"({type(exc).__name__}: {exc})"
        ) from exc


# --------------------------------------------------------------------------- #
# Building (Req 7.3, 7.4)
# --------------------------------------------------------------------------- #


def build_index(
    config: Config,
    *,
    stream: IO[str] | None = None,
    limit: int = MAX_CORPUS_FILES,
    collection_name: str = COLLECTION_NAME,
) -> BuildReport:
    """Ingest the corpus into a fresh collection and report what happened.

    The order of the body is the requirement:

    1. **Validate the chunk settings first** (Req 7.10). Before the store is
       opened, before the corpus is read, and so before anything is deleted — an
       out-of-range target length or overlap leaves an existing collection
       exactly as it was.
    2. **Open the store**, creating the persistence directory if it is absent
       (Req 7.3).
    3. **Read the corpus** through the loader, which prints one line per skipped
       file to ``stream`` and records every skip in the report (Req 7.1, 7.9).
       Reading before deleting means a corpus that turns out to be unreadable
       has not already cost the operator their previous index.
    4. **Delete any existing collection, then create it** (Req 7.4). Deleted
       rather than overwritten: adding the same ids again would leave a chunk
       from a document that has since been removed from the corpus sitting in
       the collection, still answering queries.
    5. **Add every chunk** with ``source_name`` and ``position`` as metadata
       (Req 7.2), in batches.

    :param config: supplies ``docs_dir``, ``chroma_dir``, ``chunk_size`` and
        ``chunk_overlap``. Nothing else is read.
    :param stream: where the loader prints its skipped-file lines (Req 7.9). The
        ``index`` command passes ``sys.stdout``; ``None`` prints nothing and the
        skips are still in the returned report.
    :param limit: maximum number of corpus files to select (Req 7.1). Lowering
        it is for tests.
    :param collection_name: the collection to replace. Fixed in the product;
        a parameter so a test can use its own name in a shared store.
    :returns: the loader's :class:`~agent_lab.retrieval.loader.BuildReport` —
        files ingested, files skipped with reasons, chunks indexed — which is
        exactly the three numbers Req 7.4 requires the command to print.
    :raises ChunkingError: an out-of-range chunk size or overlap, raised before
        any write (Req 7.10).
    :raises ChromaUnavailable: ``chromadb`` absent, the store unopenable, or the
        embedding model unavailable (typically: no network on a first build).
    """
    # 1. Req 7.10 — first, so a rejected build changes nothing.
    chunk_size, chunk_overlap = validate_params(config.chunk_size, config.chunk_overlap)

    # 2. Req 7.3 — creates the persistence directory if it does not exist.
    client = _client(config.chroma_dir, create=True)

    # 3. Req 7.1, 7.2, 7.9 — the same loader the keyword fallback uses.
    chunks, report = load_chunks(
        config.docs_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        limit=limit,
        stream=stream,
    )

    # 4. Req 7.4 — replace, so no chunk from a prior build survives.
    if collection_name in _collection_names(client):
        try:
            client.delete_collection(collection_name)
        except Exception as exc:  # noqa: BLE001 - a surviving collection is a defect
            raise ChromaUnavailable(
                f"the existing collection {collection_name!r} could not be "
                f"deleted ({type(exc).__name__}: {exc}), so the index was not "
                f"rebuilt; the previous collection is still in place"
            ) from exc

    collection = _collection(
        client,
        collection_name,
        metadata={_SIZE_KEY: chunk_size, _OVERLAP_KEY: chunk_overlap},
    )

    # 5. Req 7.2 — every chunk carries its source name and its position.
    _add_chunks(client, collection, chunks)

    return report


def _add_chunks(client: Any, collection: Any, chunks: Sequence[Chunk]) -> None:
    """Write ``chunks`` to ``collection`` in batches, with failures typed.

    Ids are ``"{source_name}#{position}"``. Deterministic, unique — a corpus
    cannot hold two files at one relative path, and the loader numbers positions
    within a file — and readable, so a chunk found in the store can be traced
    back to the document and the window it came from without a lookup table.

    An empty corpus writes nothing and is not an error: rebuilding over a corpus
    whose files were all removed should leave an empty collection, which is what
    step 4 of :func:`build_index` already produced.
    """
    if not chunks:
        return

    batch_size = _batch_size(client)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        try:
            collection.add(
                ids=[f"{item.source_name}#{item.position}" for item in batch],
                documents=[item.text for item in batch],
                metadatas=[
                    {"source_name": item.source_name, "position": item.position}
                    for item in batch
                ],
            )
        except Exception as exc:  # noqa: BLE001 - embedding, disk, or batch limit
            raise ChromaUnavailable(
                f"{len(batch)} chunk(s) could not be added to the document "
                f"index ({type(exc).__name__}: {exc}); the bundled embedding "
                f"model is downloaded on first use, so this build needs network "
                f"access once — keyword fallback search needs none"
            ) from exc


def _batch_size(client: Any) -> int:
    """:data:`ADD_BATCH_SIZE`, lowered to the store's own cap if it reports one.

    Asking rather than assuming: the cap is a property of the Chroma build, and
    a batch one element over it fails the whole ``add``. A version that does not
    expose ``get_max_batch_size`` keeps the module's own figure.
    """
    reported = getattr(client, "get_max_batch_size", None)
    if reported is None:
        return ADD_BATCH_SIZE
    try:
        cap = int(reported())
    except Exception:  # noqa: BLE001 - an unavailable cap is not a build failure
        return ADD_BATCH_SIZE
    return max(1, min(ADD_BATCH_SIZE, cap))


# --------------------------------------------------------------------------- #
# Querying (Req 7.5, 7.7, 7.8, 7.11)
# --------------------------------------------------------------------------- #


class ChromaRetriever:
    """Vector search over the persisted collection (Req 7.3, 7.5, 7.7).

    Satisfies :class:`~agent_lab.retrieval.base.Retriever` and adds nothing to
    it: ``search(query, k)`` in, a list of ``(chunk_text, source_name)`` pairs
    out. ``search_docs`` therefore works with this retriever and with
    :class:`~agent_lab.retrieval.keyword.KeywordRetriever` without knowing which
    it was given.

    **No ``notice``.** The optional attribute described in
    :mod:`agent_lab.retrieval.base` exists so the keyword fallback can say it
    was used instead of the index (Req 7.6). The index is the expected path, and
    a sentence saying so on every result would spend the tool result's
    4000-character budget (Req 8.3) on telling the model that nothing unusual
    happened.

    **Opened on first search, not at construction.** A retriever is built during
    startup wiring, before it is known whether the run will search at all, and
    opening a store a calculator-only run never queries is pure cost. The client
    and collection are then cached for the rest of the run.

    The persistence directory and the collection name are constructor state
    taken from :class:`~agent_lab.config.Config`; neither is a search argument,
    so no model-supplied value can point a search at another store (Req 12.9).
    """

    def __init__(
        self,
        config: Config,
        *,
        collection_name: str = COLLECTION_NAME,
        timeout_seconds: float | None = QUERY_TIMEOUT_SECONDS,
    ) -> None:
        """Record what to query. Nothing is opened and nothing is imported here.

        :param config: supplies ``chroma_dir``. Nothing else is read — the
            chunk settings belong to the build, and the collection was written
            with the settings in force when it was built.
        :param collection_name: the collection to search.
        :param timeout_seconds: Req 7.5's bound on one query, or ``None`` to
            wait indefinitely on the calling thread. ``None`` is for a caller
            that supplies its own bound.
        """
        self._chroma_dir: Path = config.chroma_dir
        self._collection_name = collection_name
        self._timeout_seconds = timeout_seconds
        self._collection: Any | None = None

    # -- the protocol operation ----------------------------------------------

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        """Return at most ``k`` ``(chunk_text, source_name)`` pairs (Req 7.5).

        Ordered by descending similarity, ties broken by ascending source name
        then ascending chunk position, so two identical queries over an
        unchanged collection return the same pairs in the same order. The sort
        happens here rather than being inherited from the store, which says
        nothing about its own tie order.

        An empty collection and a query that matches nothing both return an
        empty list; that is an answer, not an error, and ``search_docs`` has a
        result for it (Req 8.5).

        :raises RetrievalError: ``query`` empty, whitespace-only or over-long,
            or ``k`` outside its accepted range. Raised before the store is
            opened, so a rejected call performs no retrieval (Req 7.11).
        :raises ChromaUnavailable: the collection cannot be opened or queried,
            or the query exceeded :data:`QUERY_TIMEOUT_SECONDS`.
        """
        # Req 7.11 first: validation before the store is touched. The same
        # checks, the same ranges and the same error type as the keyword
        # retriever, because the two share one retrieval contract.
        checked_query = _checked_query(query)
        count = _checked_count(k)

        collection = self._open()
        size = _count(collection)
        if size == 0:
            # Chroma rejects n_results below 1 and warns when it exceeds the
            # collection size; an empty collection has no query worth issuing.
            return []

        response = _with_timeout(
            lambda: _query(collection, checked_query, min(count, size)),
            self._timeout_seconds,
            label=f"a query of the collection {self._collection_name!r}",
        )

        rows = _rows(response)
        # Negated similarity first, so one ascending sort expresses the whole
        # rule: descending similarity, ascending source name, ascending position
        # (Req 7.5).
        rows.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [(text, source) for _, source, _, text in rows[:count]]

    # -- the cached collection ------------------------------------------------

    def _open(self) -> Any:
        """The collection, opened on first use and cached (Req 7.3).

        ``create=False``: a query must not bring a persistence directory into
        existence, and a collection that is not there is
        :class:`ChromaUnavailable` — the operator has not run ``index`` yet, and
        the caller's remedy is the keyword fallback.
        """
        if self._collection is not None:
            return self._collection

        client = _client(self._chroma_dir, create=False)
        if self._collection_name not in _collection_names(client):
            raise ChromaUnavailable(
                f"the document index at {self._chroma_dir} holds no collection "
                f"named {self._collection_name!r}: run the index command to "
                f"build one, or use keyword fallback search"
            )
        self._collection = _collection(client, self._collection_name)
        return self._collection


def _query(collection: Any, query: str, n_results: int) -> Any:
    """One ``collection.query`` call, with every failure typed.

    This is where an offline machine discovers that the bundled embedding
    function has no model yet: embedding the query text fetches it, and whatever
    the socket layer raises arrives here. It becomes
    :class:`ChromaUnavailable` with the original attached, so one ``except``
    around a search covers "no index", "no network" and "corrupt store" alike,
    and the product stays usable through the keyword fallback.
    """
    try:
        return collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring
        raise ChromaUnavailable(
            f"the document index could not be queried "
            f"({type(exc).__name__}: {exc}); the bundled embedding model is "
            f"downloaded on first use, so a machine with no network access "
            f"needs keyword fallback search instead"
        ) from exc


def _count(collection: Any) -> int:
    """How many chunks the collection holds, or raise :class:`ChromaUnavailable`."""
    try:
        return int(collection.count())
    except Exception as exc:  # noqa: BLE001 - an uncountable collection is unusable
        raise ChromaUnavailable(
            f"the size of the document index could not be read "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _rows(response: Any) -> list[tuple[float, str, int, str]]:
    """Read a Chroma query response into ``(similarity, source, position, text)``.

    Chroma answers a one-query request with one list per field, each holding one
    list of results. Everything is read defensively — a missing field, a short
    list, a chunk with no metadata — because a malformed response should cost
    the affected hit and not the search, and because reading it loosely keeps
    this module working across versions that include or omit a field.

    **Similarity from distance.** Chroma reports a distance, where smaller is
    more similar; similarity is its negation. That is monotone, which is all
    Req 7.5's "descending similarity" needs, and it avoids inventing a
    normalised score whose scale would depend on the space the collection was
    built with. A hit with no distance sorts last.
    """
    documents = _first_list(response, "documents")
    metadatas = _first_list(response, "metadatas")
    distances = _first_list(response, "distances")

    rows: list[tuple[float, str, int, str]] = []
    for index, document in enumerate(documents):
        if not isinstance(document, str) or not document:
            # No text to quote: a chunk that is not text cannot be shown to the
            # model as a passage (Req 8.2).
            continue

        metadata = metadatas[index] if index < len(metadatas) else None
        source = _metadata_str(metadata, "source_name")
        if not source:
            # Req 7.8 / Req 8.9: a hit that cannot name its source file is not
            # quotable. Dropped rather than labelled "unknown", which would read
            # to the model like a file called "unknown".
            continue

        distance = distances[index] if index < len(distances) else None
        similarity = -float(distance) if isinstance(distance, (int, float)) else float("-inf")

        rows.append((similarity, source, _metadata_int(metadata, "position"), document))
    return rows


def _first_list(response: Any, key: str) -> list[Any]:
    """``response[key][0]`` as a list, or ``[]`` for any other shape."""
    try:
        outer = response[key]
    except (TypeError, KeyError, IndexError):
        return []
    if not isinstance(outer, (list, tuple)) or not outer:
        return []
    inner = outer[0]
    if isinstance(inner, (list, tuple)):
        return list(inner)
    return []


def _metadata_str(metadata: Any, key: str) -> str:
    """A string metadata value, or ``""`` when it is absent or not a string."""
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) else ""


def _metadata_int(metadata: Any, key: str) -> int:
    """An integer metadata value, or ``0`` when it is absent or unreadable.

    ``0`` rather than a raise: the position only breaks ties, so a hit that lost
    its position is still worth returning — it just sorts as if it were the
    first chunk of its file.
    """
    if not isinstance(metadata, dict):
        return 0
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _with_timeout(call: Callable[[], _T], timeout: float | None, *, label: str) -> _T:
    """Run ``call``, abandoning it after ``timeout`` seconds (Req 7.5).

    The same mechanism the tool dispatcher uses, and for the same reason: a
    single-worker executor with ``future.result(timeout=...)`` works on Windows,
    where ``signal.alarm`` does not exist. ``shutdown(wait=False)`` is what
    makes this return now rather than when the abandoned query eventually
    finishes — a thread already running cannot be cancelled, so the call is left
    to end on its own while the caller gets its error.

    ``timeout=None`` runs inline, with no executor and no thread.
    """
    if timeout is None:
        return call()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma-query")
    try:
        future = executor.submit(call)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise ChromaUnavailable(
                f"{label} did not complete within {timeout:g} seconds and was "
                f"abandoned"
            ) from exc
    finally:
        executor.shutdown(wait=False)


# --------------------------------------------------------------------------- #
# Parameter validation (Req 7.11)
# --------------------------------------------------------------------------- #
#
# The same ranges, the same error type and the same wording as the keyword
# retriever, imported from it rather than restated: Req 7.11 is a property of
# the retrieval contract, and two retrievers that disagree about which k is
# acceptable would make the contract untestable.


def _checked_query(query: object) -> str:
    """Return ``query`` as a usable query string, or raise (Req 7.11)."""
    accepted = (
        f"a string of {MIN_QUERY_CHARS} to {MAX_QUERY_CHARS} characters that is "
        f"not only whitespace"
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


def _checked_count(k: object) -> int:
    """Return ``k`` as a usable result count, or raise (Req 7.11)."""
    accepted = f"an integer between {MIN_RESULT_COUNT} and {MAX_RESULT_COUNT} inclusive"
    if isinstance(k, bool) or not isinstance(k, int):
        raise RetrievalError(
            f"k (the result count) must be {accepted}, not {type(k).__name__}; "
            f"no retrieval was performed"
        )
    if not MIN_RESULT_COUNT <= k <= MAX_RESULT_COUNT:
        raise RetrievalError(
            f"k (the result count) must be {accepted}, received {k}; "
            f"no retrieval was performed"
        )
    return k


# --------------------------------------------------------------------------- #
# Selection (Req 7.6)
# --------------------------------------------------------------------------- #


def collection_size(
    config: Config, *, collection_name: str = COLLECTION_NAME
) -> int | None:
    """How many chunks the persisted collection holds, or ``None`` if there is none.

    ``None`` for every reason the collection cannot be counted — no store, no
    collection, no ``chromadb``, an unopenable database — because the caller's
    decision is the same in all of them. ``0`` is a different answer: a
    collection that exists and is empty, which an ``index`` run over an empty
    corpus produces, and which should send a query to the fallback rather than
    return nothing from a vector search over no chunks.

    Raises nothing. This is a question asked during startup wiring, and a
    startup question whose honest answer is "there is no index" must not be able
    to end the process.
    """
    if not store_exists(config.chroma_dir):
        return None
    try:
        client = _client(config.chroma_dir, create=False)
        if collection_name not in _collection_names(client):
            return None
        return _count(_collection(client, collection_name))
    except ChromaUnavailable:
        return None


def has_index(config: Config, *, collection_name: str = COLLECTION_NAME) -> bool:
    """Is there a persisted collection with at least one chunk in it (Req 7.6)?"""
    size = collection_size(config, collection_name=collection_name)
    return size is not None and size > 0


def select_retriever(
    config: Config, *, collection_name: str = COLLECTION_NAME
) -> Retriever:
    """The retriever a run should search through (Req 7.6, 7.7).

    :class:`ChromaRetriever` when the persistence directory holds a non-empty
    collection; :class:`~agent_lab.retrieval.keyword.KeywordRetriever`
    otherwise, which is the case on a fresh clone, after a corpus was added but
    ``index`` was not run, and on a machine where ``chromadb`` cannot be
    imported. The fallback states in every result that it was used (Req 7.6),
    so the operator is never left guessing which search answered.

    This is the one place in the product that chooses between the two, and it
    returns the :class:`~agent_lab.retrieval.base.Retriever` protocol, so
    nothing downstream — ``build_tool_registry``, ``search_docs``, the loop —
    learns which it got.

    Raises nothing: an absent or broken index selects the fallback. An index
    that exists but turns unusable at query time — the embedding model is not
    cached and the machine is offline — is reported by ``search_docs`` as a
    failed search rather than ending the run, because the selection decision has
    been made by then.
    """
    if has_index(config, collection_name=collection_name):
        return ChromaRetriever(config, collection_name=collection_name)
    return KeywordRetriever(config)


__all__ = [
    "ADD_BATCH_SIZE",
    "COLLECTION_NAME",
    "QUERY_TIMEOUT_SECONDS",
    "STORE_FILE",
    "ChromaRetriever",
    "ChromaUnavailable",
    "build_index",
    "collection_size",
    "has_index",
    "select_retriever",
    "store_exists",
]
