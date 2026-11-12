"""The corpus loader: paths in, chunks and a build report out (Req 7.1, 7.9).

:func:`load_chunks` is the one entry point. It takes a corpus directory and the
chunking parameters and returns every :class:`Chunk` the corpus yields, in a
deterministic order, together with a :class:`BuildReport` stating how many files
were ingested, which were skipped and why, and how many chunks resulted. Both
the Chroma index (Req 7.3) and the keyword fallback (Req 7.6) build on this, so
the two see exactly the same corpus.

**It does not choose the files.** The walk, the case-insensitive extension
filter and the 10,000-file ceiling live in :mod:`agent_lab.retrieval.base` and
this module calls :func:`~agent_lab.retrieval.base.corpus_files` for its paths.
That is deliberate: ``search_docs`` uses the same function to decide whether the
corpus holds anything at all (Req 8.6) and to check that a returned source names
a real corpus file (Req 8.9). One definition of "a corpus file", used by the
side that indexes and the side that reports, is what keeps those two answers
from disagreeing.

**One file's problem is not the build's problem.** Req 7.9 lists three ways a
selected file drops out — it cannot be opened or read, it exceeds 25 MB, or it
yields nothing but whitespace after extraction — and in all three the build
continues with the remaining files, the skipped file is excluded from the
ingested count, and its name and reason are reported. So every per-file failure
here is converted into a :class:`DocumentSkipped` and recorded; nothing short of
a programming error escapes :func:`load_chunks`. A corpus of 500 notes with one
encrypted PDF in it still produces an index of 499 notes.

**Reporting is to a stream the caller passes, not to stdout.** Req 7.9 asks for
the skipped file names and reasons to be printed, and the ``index`` command
passes ``sys.stdout`` to do that (Req 7.4). The keyword fallback calls this same
loader in the middle of a run, where printing a dozen skip lines would land in
the operator's transcript between the agent's steps; it passes no stream and
reads ``report.files_skipped`` instead. The report always carries every skip,
whether or not anything was printed.

**Source names are corpus-relative, with forward slashes.** ``notes/api.md``
rather than ``api.md``, so two files of the same name in different
subdirectories stay distinguishable in a result — ``search_docs`` accepts either
spelling. Forward slashes on every platform, so an index built on Windows and
read on Linux carries the same source names.

**Why validation happens before the first read.** :func:`load_chunks` validates
``chunk_size`` and ``chunk_overlap`` through
:func:`~agent_lab.retrieval.chunking.validate_params` before it touches the
filesystem, so a build with an out-of-range parameter fails having read nothing
and written nothing, leaving an existing persisted collection untouched
(Req 7.10).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from agent_lab.retrieval.base import MAX_CORPUS_FILES, corpus_files
from agent_lab.retrieval.chunking import chunk, validate_params

# Req 7.9: the per-file size ceiling. Measured in bytes on disk, before any
# decoding or PDF parsing, because the point of the limit is to refuse to pull
# the file into memory at all. 25 MB of prose is some four million words; a
# corpus file past that is an archive, a dump or a mistake, and either way one
# file should not be allowed to make a build unbounded.
MAX_FILE_BYTES = 25 * 1024 * 1024

# Extensions read as text. Everything admitted by ``SUPPORTED_EXTENSIONS`` that
# is not in here goes through the PDF path, so adding an extension to the corpus
# definition without deciding how to extract its text is a visible omission
# rather than a silent fallthrough.
_TEXT_EXTENSIONS = frozenset({".txt", ".md"})


@dataclass(frozen=True)
class Chunk:
    """One indexable window of one document.

    :param text: the chunk text, verbatim from the extracted document.
    :param source_name: the corpus-relative path of the file it came from, with
        forward slashes (Req 8.9).
    :param position: the zero-based index of this chunk within its own document
        (Req 7.2) — not within the corpus, so positions restart at 0 for every
        file and the pair ``(source_name, position)`` identifies a chunk.
    """

    text: str
    source_name: str
    position: int


@dataclass(frozen=True)
class BuildReport:
    """What one pass over the corpus did, in the numbers the operator sees.

    ``files_ingested + len(files_skipped)`` is the number of files the corpus
    walk selected, so the two counts the ``index`` command prints account for
    every supported file it looked at (Req 7.4, Req 7.9).
    """

    files_ingested: int
    files_skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks_indexed: int = 0


class DocumentSkipped(Exception):
    """One file cannot be ingested; the build continues without it (Req 7.9).

    Carries the operator-facing reason, which is the message. Raised only by the
    single-file extraction helpers and caught by :func:`load_chunks`, which
    records it against the file's source name — it never reaches a caller of
    :func:`load_chunks`, because no single file's problem ends a build.
    """


def load_chunks(
    docs_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    limit: int = MAX_CORPUS_FILES,
    stream: IO[str] | None = None,
) -> tuple[list[Chunk], BuildReport]:
    """Ingest the corpus under ``docs_dir`` into chunks (Req 7.1, 7.2, 7.9).

    Deterministic: the corpus walk is sorted at every level and the chunker is
    pure, so two calls over an unchanged corpus return equal chunks in equal
    order. That is what lets a rebuilt index and a repeated query be compared.

    :param docs_dir: the corpus directory. An absent or unreadable directory is
        not an error here — it yields no chunks and an empty report, and naming
        the configured path is the caller's job (Req 8.6).
    :param chunk_size: target chunk length in characters, validated before any
        file is opened (Req 7.10).
    :param chunk_overlap: characters each chunk repeats from its predecessor.
    :param limit: maximum number of files to select, defaulting to the
        ``MAX_CORPUS_FILES`` ceiling of Req 7.1. Lowering it is for tests.
    :param stream: where to print one line per skipped file, naming the file and
        the reason (Req 7.9). ``None`` prints nothing; the skips are in the
        returned report either way.
    :returns: ``(chunks, report)``. Chunks are grouped by file in corpus order
        and ascending position within each file.
    :raises ChunkingError: if ``chunk_size`` or ``chunk_overlap`` is out of
        range — raised before any file is read, so a rejected build leaves
        everything, including an existing collection, untouched (Req 7.10).
    """
    # First, so that an out-of-range parameter costs nothing and changes nothing.
    chunk_size, chunk_overlap = validate_params(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    ingested = 0

    for path in corpus_files(docs_dir, limit=limit):
        source_name = _source_name(path, docs_dir)
        try:
            text = read_document(path)
        except DocumentSkipped as exc:
            skipped.append((source_name, str(exc)))
            if stream is not None:
                print(f"skipped {source_name}: {exc}", file=stream)
            continue

        windows = chunk(text, chunk_size, chunk_overlap)
        chunks.extend(
            Chunk(text=window, source_name=source_name, position=position)
            for position, window in enumerate(windows)
        )
        ingested += 1

    return chunks, BuildReport(
        files_ingested=ingested,
        files_skipped=skipped,
        chunks_indexed=len(chunks),
    )


def read_document(path: Path) -> str:
    """Extract the text of one corpus file, or explain why it cannot be used.

    The size check comes before the read for both kinds of file, and the
    "nothing but whitespace" check comes after extraction for both, so the three
    skip conditions of Req 7.9 are applied uniformly however the text was
    obtained.

    :returns: the extracted text, which always contains at least one
        non-whitespace character.
    :raises DocumentSkipped: if the file cannot be stat'ed, opened, read,
        decoded or parsed, if it exceeds :data:`MAX_FILE_BYTES`, or if it yields
        no text or only whitespace.
    """
    _check_size(path)

    if path.suffix.lower() in _TEXT_EXTENSIONS:
        text = _read_text(path)
    else:
        text = _read_pdf(path)

    if not text.strip():
        # Req 7.9: an empty file, a page of blank lines, or a scanned PDF with
        # no text layer. All three index nothing, and all three are worth
        # telling the operator about rather than counting as ingested.
        raise DocumentSkipped(
            "yielded no extractable text (the file is empty or holds only "
            "whitespace after text extraction)"
        )
    return text


def _check_size(path: Path) -> None:
    """Reject a file over the size ceiling, and one that cannot be stat'ed.

    A failing ``stat`` is reported as "cannot be read" rather than ignored: a
    path that the corpus walk listed but that cannot be inspected a moment later
    was deleted, renamed, or sits behind a permission the walk did not need.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DocumentSkipped(f"could not be read ({_reason(exc)})") from exc

    if size > MAX_FILE_BYTES:
        # The actual size is given in bytes rather than rounded megabytes: a
        # file one byte over the ceiling would otherwise be reported as
        # "25.0 MB exceeds the 25 MB limit", which reads like a bug.
        raise DocumentSkipped(
            f"exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB per-file limit "
            f"({size:,} bytes)"
        )


def _read_text(path: Path) -> str:
    """Read a ``.txt`` or ``.md`` file as UTF-8.

    Strict UTF-8, with a byte-order mark tolerated. A file in some other
    encoding is skipped with the offending byte offset named, rather than
    decoded with replacement characters: mojibake in the index is a passage the
    agent would quote back to the operator as if it were their note, while a
    skip line says exactly which file needs converting.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentSkipped(f"could not be read ({_reason(exc)})") from exc

    try:
        # ``utf-8-sig`` strips a leading BOM if present and is plain UTF-8
        # otherwise, so a note saved by a Windows editor does not start with a
        # stray character.
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentSkipped(
            f"could not be decoded as UTF-8 text (invalid byte at offset "
            f"{exc.start})"
        ) from exc


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF by joining its pages' ``extract_text()`` output.

    Pages are joined with a blank line so a sentence ending one page and a
    heading opening the next do not run together into a single word.

    Everything ``pypdf`` can raise is caught. It raises its own
    ``PdfReadError`` for a malformed file, but a truncated, encrypted or
    adversarial PDF can also surface a ``KeyError``, a ``ValueError``, a
    ``struct.error`` or a recursion failure from deep inside the parser, and
    Req 7.9 requires every one of those to skip one file rather than end the
    build. A page that fails on its own is likewise skipped, keeping the pages
    that did extract: a 300-page manual with one broken font table is still
    worth indexing.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a declared dependency
        raise DocumentSkipped(
            "could not be read (PDF support requires the pypdf package)"
        ) from exc

    with _quiet_pypdf():
        try:
            reader = PdfReader(str(path))
            pages = list(reader.pages)
        except Exception as exc:  # noqa: BLE001 - Req 7.9: skip the file, not the build
            raise DocumentSkipped(
                f"could not be parsed as a PDF ({_reason(exc)})"
            ) from exc

        extracted: list[str] = []
        for page in pages:
            try:
                text = page.extract_text()
            except Exception:  # noqa: BLE001 - one bad page, not one lost file
                continue
            if text:
                extracted.append(text)

    return "\n\n".join(extracted)


@contextmanager
def _quiet_pypdf() -> Iterator[None]:
    """Keep ``pypdf``'s own log output out of the operator's terminal.

    ``pypdf`` reports its complaints about a malformed file ("EOF marker not
    found") through the ``logging`` module, and with no handler configured those
    records land on stderr beside — or instead of — the skip line this module
    prints. The skip report is the operator's channel for a file that could not
    be read (Req 7.9); a parser's internal commentary arriving separately, and
    for files that were ingested successfully anyway, only obscures it. It
    matters most on the fallback path, where the loader runs in the middle of a
    run and stray lines would break up the transcript.

    The level is raised for the duration of one file's parse and restored
    afterwards, so a program that configures ``pypdf`` logging itself gets its
    setting back.
    """
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _source_name(path: Path, docs_dir: Path) -> str:
    """The corpus-relative path of ``path``, with forward slashes (Req 8.9).

    Falls back to the bare file name if the path somehow does not sit under
    ``docs_dir`` — ``search_docs`` resolves a bare name too, so a source name is
    always something it can check against the corpus.
    """
    try:
        relative = path.relative_to(docs_dir)
    except ValueError:  # pragma: no cover - corpus paths come from docs_dir
        return path.name
    return relative.as_posix()


def _reason(exc: BaseException) -> str:
    """A short, operator-readable reason from an exception.

    ``strerror`` for an OS error ("Permission denied"), because the ``str`` of
    one repeats the whole path the message already names. Otherwise the
    exception's own message, or its type when the message is empty, so a skip
    line is never just ``skipped notes/a.pdf: ()``.
    """
    strerror = getattr(exc, "strerror", None)
    if strerror:
        return str(strerror)
    return str(exc) or type(exc).__name__


__all__ = [
    "MAX_FILE_BYTES",
    "BuildReport",
    "Chunk",
    "DocumentSkipped",
    "load_chunks",
    "read_document",
]
