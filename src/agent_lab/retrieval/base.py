"""The retrieval contract: one protocol, one definition of "a corpus file".

Req 7.7 asks for a retrieval operation narrow enough that an external retrieval
implementation can replace the one shipped here without changing
``search_docs``. This module is that contract and nothing else:

* :class:`Retriever` — ``search(query, k) -> list[(chunk_text, source_name)]``.
  Two arguments in, a list of pairs out. That is the whole substitution
  contract.
* :data:`SUPPORTED_EXTENSIONS`, :data:`MAX_CORPUS_FILES`,
  :func:`corpus_files` — what counts as a document in the corpus directory
  (Req 7.1), which ``search_docs`` has to know independently of any retriever
  in order to answer Req 8.6 (no documents available, naming the configured
  path) and Req 8.9 (every source name in a result names a file present in the
  corpus).

Nothing here imports from the rest of the package, reads configuration, or
builds an index. The corpus walk lives beside the protocol rather than inside
``search_docs`` because the loader (Req 7.1, Req 7.9) selects exactly the same
files when it ingests them: one definition of "supported file", used by the side
that indexes and by the side that reports.

The loader will do more than this — size limits, PDF text extraction, per-file
skip reasons — but it will select the same paths in the same order, because it
will select them from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

# Req 7.1: the three extensions the corpus admits, compared without regard to
# letter case. Everything else in the directory is ignored rather than reported.
SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})

# Req 7.1: the per-build ceiling on file count. Applied by the walk below so
# that a corpus directory pointed at something enormous — a home directory, a
# mounted share — cannot make an operation unbounded.
MAX_CORPUS_FILES = 10_000


@runtime_checkable
class Retriever(Protocol):
    """Answer a query with chunk text and the source each chunk came from.

    ``search`` returns at most ``k`` pairs of ``(chunk_text, source_name)``
    ordered by descending similarity to ``query`` (Req 7.5, Req 7.8). An empty
    list is the answer for a query that matches nothing; it is not an error.

    ``source_name`` names a file in the document corpus. A bare file name and a
    corpus-relative path are both accepted spellings — ``search_docs`` resolves
    either against the corpus when it checks Req 8.9 — so an implementation that
    walks subdirectories can disambiguate two files of the same name by
    returning the relative path.

    A runtime-checkable protocol, so a factory can refuse an object that has no
    ``search`` at wiring time instead of failing in the middle of a step. The
    check is structural and shallow: it confirms the attribute exists, not that
    the signature matches.

    **Optional extra.** An implementation may also expose a ``notice``
    attribute holding one sentence about how the search was performed;
    ``search_docs`` includes it in the result when it is a non-empty string.
    That is how the keyword fallback states that it was used instead of the
    vector index (Req 7.6) without adding a required member to this protocol
    and so narrowing what can be substituted.
    """

    def search(self, query: str, k: int) -> list[tuple[str, str]]: ...


def corpus_files(docs_dir: Path, *, limit: int = MAX_CORPUS_FILES) -> list[Path]:
    """Every supported file under ``docs_dir``, to unlimited depth (Req 7.1).

    Returns paths in a deterministic order — directories and file names sorted
    at every level — so that two calls over an unchanged corpus produce the same
    list, which is what lets an ingested-file count and a chunk ordering be
    reproducible.

    Returns an empty list when ``docs_dir`` does not exist, is not a directory,
    or holds no supported file. Those three cases are distinguished by the
    caller, not here: ``search_docs`` has to name which one occurred (Req 8.6),
    and a function that returns a list is the wrong place to explain why the
    list is empty.

    Unreadable subdirectories are skipped silently and the walk continues, which
    is ``os.walk``'s default behaviour. A directory that cannot be listed
    contributes no files, and the files that could be read are still found;
    refusing to search the whole corpus because one subdirectory is protected
    would be worse. A file that cannot be *opened* is a different matter and is
    the loader's to report (Req 7.9) — this function only selects paths and
    never opens them.
    """
    selected: list[Path] = []
    if limit <= 0:
        return selected

    try:
        if not docs_dir.is_dir():
            return selected
    except OSError:
        # A path that cannot even be stat'ed — a broken mount, a denied parent —
        # holds no searchable file as far as this process is concerned.
        return selected

    for root, dirnames, filenames in os.walk(docs_dir):
        # Sorted in place so ``os.walk`` descends in this order too.
        dirnames.sort()
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            selected.append(Path(root) / filename)
            if len(selected) >= limit:
                return selected
    return selected


__all__ = [
    "MAX_CORPUS_FILES",
    "SUPPORTED_EXTENSIONS",
    "Retriever",
    "corpus_files",
]
