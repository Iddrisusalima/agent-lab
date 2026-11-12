"""Unit tests for the retrieval layer: the corpus walk, the loader, the
keyword fallback, and the choice between the two (Req 7).

Specific examples and edge cases only. The universal statement — Property 7,
"retrieval returns at most k results from real corpus files" — lives in
`tests/test_properties.py`, and the chunker has its own file
(`tests/test_chunking.py`) and its own property, so nothing here re-tests
windowing.

What each section pins down:

* **``corpus_files``** (Req 7.1) — which files are a corpus: the three
  extensions compared without regard to letter case, to unlimited depth, in a
  deterministic order, up to the per-build ceiling.
* **``load_chunks``** (Req 7.1, 7.2, 7.9, 7.10) — the three ways one file drops
  out of a build without ending it, the counts the ``index`` command prints,
  and the source-name spelling every result is checked against (Req 8.9).
* **``KeywordRetriever``** (Req 7.5, 7.6, 7.11) — at most ``k`` pairs in the
  documented order, the fallback notice, and the parameters it refuses before
  reading anything.
* **``select_retriever``** (Req 7.6) — the fallback whenever there is no usable
  collection, without creating a persistence directory to find that out.

**Nothing here touches the repository.** Every path comes from the ``tmp_path``
fixtures in ``conftest.py`` (Req 16.9), so the ``.chroma`` directory at the
repository root is neither read nor written. The one Chroma-backed test builds
an index under ``chroma_dir`` and skips itself if the vector path is unusable:
the suite runs with every socket connect refused (Req 16.8) and Chroma's
bundled embedding function downloads its model on first use, which is exactly
what :class:`ChromaUnavailable` is for.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest

from agent_lab.config import Config
from agent_lab.retrieval import loader as loader_module
from agent_lab.retrieval.base import (
    MAX_CORPUS_FILES,
    SUPPORTED_EXTENSIONS,
    Retriever,
    corpus_files,
)
from agent_lab.retrieval.chroma_index import (
    ChromaRetriever,
    ChromaUnavailable,
    build_index,
    collection_size,
    has_index,
    select_retriever,
    store_exists,
)
from agent_lab.retrieval.chunking import ChunkingError
from agent_lab.retrieval.keyword import (
    MAX_QUERY_CHARS,
    MAX_RESULT_COUNT,
    MIN_RESULT_COUNT,
    NOTICE,
    KeywordRetriever,
    RetrievalError,
    normalise,
    query_terms,
    score_chunk,
)
from agent_lab.retrieval.loader import (
    BuildReport,
    Chunk,
    DocumentSkipped,
    load_chunks,
    read_document,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def write(directory: Path, name: str, text: str = "content") -> Path:
    """Write one corpus file, creating any parent directory the name implies."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def names(paths: list[Path], root: Path) -> list[str]:
    """The corpus-relative names of ``paths``, for comparison with a literal."""
    return [path.relative_to(root).as_posix() for path in paths]


def retriever_for(config: Config, **overrides: object) -> KeywordRetriever:
    """A keyword retriever over ``config``, with fields optionally replaced."""
    return KeywordRetriever(dataclasses.replace(config, **overrides))


def sources(pairs: list[tuple[str, str]]) -> list[str]:
    """The source names of a result, in result order."""
    return [source for _, source in pairs]


# --------------------------------------------------------------------------- #
# The corpus walk (Req 7.1)
# --------------------------------------------------------------------------- #


def test_the_three_supported_extensions_are_the_documented_ones():
    assert SUPPORTED_EXTENSIONS == frozenset({".txt", ".md", ".pdf"})
    assert MAX_CORPUS_FILES == 10_000


def test_extensions_are_matched_without_regard_to_letter_case(corpus_dir):
    """Req 7.1: ``.MD`` is as much a corpus file as ``.md``."""
    for name in ("lower.md", "UPPER.MD", "Mixed.Md", "shout.TXT", "paper.PDF"):
        write(corpus_dir, name)

    assert names(corpus_files(corpus_dir), corpus_dir) == [
        "Mixed.Md",
        "UPPER.MD",
        "lower.md",
        "paper.PDF",
        "shout.TXT",
    ]


def test_every_other_extension_is_ignored_including_near_misses(corpus_dir):
    """Req 7.1: anything else is ignored, not reported — including ``.md.bak``."""
    write(corpus_dir, "keep.md")
    for name in ("notes.rst", "data.json", "README", "archive.md.bak", "script.py", ".md"):
        write(corpus_dir, name)

    assert names(corpus_files(corpus_dir), corpus_dir) == ["keep.md"]


def test_subdirectories_are_walked_to_unlimited_depth_in_sorted_order(corpus_dir):
    """Req 7.1: every depth, and a deterministic order at every level."""
    write(corpus_dir, "b.md")
    write(corpus_dir, "a.md")
    write(corpus_dir, "zeta/m.txt")
    write(corpus_dir, "alpha/one.md")
    write(corpus_dir, "alpha/deep/deeper/deepest/buried.txt")

    # ``os.walk`` is top-down, and the walk sorts the directory names in place
    # so it descends in that order: this directory's own files first, then each
    # subdirectory in turn, depth first.
    found = names(corpus_files(corpus_dir), corpus_dir)
    assert found == [
        "a.md",
        "b.md",
        "alpha/one.md",
        "alpha/deep/deeper/deepest/buried.txt",
        "zeta/m.txt",
    ]
    # Deterministic: a second walk over an unchanged corpus is the same list,
    # which is what makes an ingested-file count reproducible.
    assert names(corpus_files(corpus_dir), corpus_dir) == found


def test_the_file_ceiling_caps_the_walk_and_a_zero_limit_selects_nothing(corpus_dir):
    """Req 7.1: a maximum per build, applied by the walk itself."""
    for index in range(8):
        write(corpus_dir, f"note-{index}.md")

    assert len(corpus_files(corpus_dir, limit=3)) == 3
    # The cap keeps the deterministic order, so the limited walk is a prefix.
    assert names(corpus_files(corpus_dir, limit=3), corpus_dir) == names(
        corpus_files(corpus_dir), corpus_dir
    )[:3]
    assert corpus_files(corpus_dir, limit=0) == []
    assert corpus_files(corpus_dir, limit=-1) == []


def test_an_absent_directory_and_a_file_path_both_select_nothing(corpus_dir, tmp_path):
    """Three different emptinesses, all answered with an empty list (Req 8.6)."""
    # An empty directory, a path that does not exist, and a path that is a file
    # rather than a directory. ``search_docs`` tells them apart; the walk does
    # not have to.
    assert corpus_files(corpus_dir) == []
    assert corpus_files(tmp_path / "nowhere") == []
    assert corpus_files(write(corpus_dir, "a.md")) == []


# --------------------------------------------------------------------------- #
# The loader (Req 7.1, 7.2, 7.9, 7.10)
# --------------------------------------------------------------------------- #


def test_the_report_accounts_for_every_file_the_walk_selected(corpus_dir):
    """Req 7.4: ingested plus skipped is what the walk found."""
    write(corpus_dir, "good.md", "alpha beta gamma")
    write(corpus_dir, "also-good.txt", "delta epsilon")
    write(corpus_dir, "blank.md", "   \n\t  \n")
    write(corpus_dir, "ignored.rst", "not a corpus file at all")

    chunks, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert isinstance(report, BuildReport)
    assert report.files_ingested == 2
    assert len(report.files_skipped) == 1
    assert report.files_ingested + len(report.files_skipped) == len(
        corpus_files(corpus_dir)
    )
    assert report.chunks_indexed == len(chunks) == 2


def test_chunks_carry_a_corpus_relative_source_name_with_forward_slashes(corpus_dir):
    """Req 7.2 and Req 8.9: the spelling a result is checked against."""
    write(corpus_dir, "notes/sub dir/api.md", "the retention policy is ninety days")

    chunks, _ = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert [chunk.source_name for chunk in chunks] == ["notes/sub dir/api.md"]
    assert "\\" not in chunks[0].source_name


def test_positions_restart_at_zero_for_every_document(corpus_dir):
    """Req 7.2: the position is within the source document, not the corpus."""
    write(corpus_dir, "a.md", "A" * 450)
    write(corpus_dir, "b.md", "B" * 450)

    chunks, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    by_source: dict[str, list[int]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_name, []).append(chunk.position)
    assert by_source == {"a.md": [0, 1, 2], "b.md": [0, 1, 2]}
    assert report.chunks_indexed == 6
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_a_file_with_no_extractable_text_is_skipped_with_a_reason(corpus_dir):
    """Req 7.9, the third skip condition: whitespace only after extraction."""
    write(corpus_dir, "empty.md", "")
    write(corpus_dir, "blank.txt", " \n\r\t ")
    write(corpus_dir, "real.md", "something to index")

    chunks, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert report.files_ingested == 1
    assert sorted(name for name, _ in report.files_skipped) == ["blank.txt", "empty.md"]
    assert all("no extractable text" in reason for _, reason in report.files_skipped)
    # The build continued, and the skipped files contributed nothing.
    assert [chunk.source_name for chunk in chunks] == ["real.md"]


def test_a_file_that_is_not_valid_utf8_is_skipped_naming_the_byte_offset(corpus_dir):
    """Req 7.9, the first skip condition: the file cannot be read as text."""
    (corpus_dir / "legacy.txt").write_bytes(b"caf\xe9 au lait")
    write(corpus_dir, "fine.md", "ordinary prose")

    chunks, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert report.files_ingested == 1
    assert [name for name, _ in report.files_skipped] == ["legacy.txt"]
    reason = report.files_skipped[0][1]
    assert "UTF-8" in reason and "offset 3" in reason
    assert [chunk.source_name for chunk in chunks] == ["fine.md"]


def test_a_file_over_the_size_ceiling_is_skipped_naming_its_size(
    corpus_dir, monkeypatch
):
    """Req 7.9, the second skip condition, checked before the file is read.

    The ceiling is lowered to a megabyte for the test rather than writing
    25 MB to disk: the bound is what is under test, not the constant.
    """
    monkeypatch.setattr(loader_module, "MAX_FILE_BYTES", 1024 * 1024)
    (corpus_dir / "huge.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    write(corpus_dir, "small.md", "a modest note")

    chunks, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert report.files_ingested == 1
    assert [name for name, _ in report.files_skipped] == ["huge.txt"]
    assert "exceeds the 1 MB per-file limit" in report.files_skipped[0][1]
    assert "1,048,577 bytes" in report.files_skipped[0][1]
    assert [chunk.source_name for chunk in chunks] == ["small.md"]


def test_a_pdf_that_cannot_be_parsed_is_skipped_and_the_build_continues(corpus_dir):
    """Req 7.9: one unreadable document costs its own file and nothing else."""
    (corpus_dir / "broken.pdf").write_bytes(b"%PDF-1.4 but not really a pdf")
    write(corpus_dir, "good.md", "the readable one")

    chunks, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert report.files_ingested == 1
    assert [name for name, _ in report.files_skipped] == ["broken.pdf"]
    assert [chunk.source_name for chunk in chunks] == ["good.md"]


def test_skipped_files_are_printed_to_the_stream_the_caller_passes(corpus_dir):
    """Req 7.9, Req 7.4: the ``index`` command's skipped-file report."""
    write(corpus_dir, "blank.md", "")
    write(corpus_dir, "real.md", "indexable")

    stream = io.StringIO()
    _, report = load_chunks(
        corpus_dir, chunk_size=200, chunk_overlap=0, stream=stream
    )

    printed = stream.getvalue().splitlines()
    assert len(printed) == 1
    assert printed[0].startswith("skipped blank.md: ")
    assert report.files_skipped[0][1] in printed[0]


def test_no_stream_prints_nothing_but_the_report_still_carries_the_skip(
    corpus_dir, capsys
):
    """The fallback path: the skips are in the report, not in the transcript."""
    write(corpus_dir, "blank.md", "")
    write(corpus_dir, "real.md", "indexable")

    _, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert capsys.readouterr().out == ""
    assert [name for name, _ in report.files_skipped] == ["blank.md"]


def test_a_byte_order_mark_is_not_indexed_as_text(corpus_dir):
    """A note saved by a Windows editor must not start with a stray character."""
    (corpus_dir / "bom.md").write_bytes(b"\xef\xbb\xbfheading")

    chunks, _ = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)

    assert chunks[0].text == "heading"


def test_an_absent_corpus_directory_yields_no_chunks_and_is_not_an_error(tmp_path):
    """Naming the configured path is ``search_docs``'s job, not the loader's."""
    chunks, report = load_chunks(
        tmp_path / "nowhere", chunk_size=200, chunk_overlap=0
    )

    assert chunks == []
    assert report == BuildReport(files_ingested=0, files_skipped=[], chunks_indexed=0)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(199, 0), (4_001, 0), (1_000, 1_000), (1_000, -1)],
)
def test_out_of_range_chunk_settings_are_rejected_before_any_file_is_read(
    corpus_dir, monkeypatch, size, overlap
):
    """Req 7.10: a rejected build reads nothing, so it can change nothing.

    The walk is replaced with one that fails if it is called, which is the only
    way to assert "before any file is read" rather than merely "raises".
    """

    def unreachable(*args: object, **kwargs: object) -> list[Path]:
        raise AssertionError("the corpus was walked despite invalid parameters")

    monkeypatch.setattr(loader_module, "corpus_files", unreachable)
    write(corpus_dir, "a.md", "content")

    with pytest.raises(ChunkingError):
        load_chunks(corpus_dir, chunk_size=size, chunk_overlap=overlap)


def test_read_document_raises_document_skipped_for_one_unusable_file(corpus_dir):
    """The per-file helper raises; only :func:`load_chunks` converts to a report."""
    path = write(corpus_dir, "blank.md", "   ")

    with pytest.raises(DocumentSkipped):
        read_document(path)

    assert read_document(write(corpus_dir, "ok.md", " text ")) == " text "


# --------------------------------------------------------------------------- #
# The keyword fallback (Req 7.5, 7.6, 7.11)
# --------------------------------------------------------------------------- #


def test_the_keyword_retriever_satisfies_the_retriever_protocol(tmp_config):
    """Req 7.7: ``search`` alone is the substitution contract."""
    assert isinstance(KeywordRetriever(tmp_config), Retriever)


def test_the_fallback_states_that_it_was_used(tmp_config):
    """Req 7.6: the sentence ``search_docs`` includes in the result."""
    retriever = KeywordRetriever(tmp_config)

    assert retriever.notice == NOTICE
    assert "keyword fallback search was used" in retriever.notice.lower()
    assert KeywordRetriever(tmp_config, notice="because X").notice == "because X"


def test_a_misconfigured_chunk_setting_fails_at_construction(tmp_config):
    """Req 7.10: during wiring, not three steps into a run."""
    with pytest.raises(ChunkingError):
        retriever_for(tmp_config, chunk_size=199)


def test_at_most_k_matching_chunks_are_returned(tmp_config, corpus_dir):
    """Req 7.5: ``k`` is a ceiling, and only matching chunks count towards it."""
    write(corpus_dir, "a.md", "retention policy " * 20)
    write(corpus_dir, "b.md", "retention policy " * 20)
    write(corpus_dir, "c.md", "unrelated prose " * 20)

    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    for k in (1, 2, 3, 10, MAX_RESULT_COUNT):
        results = retriever.search("retention policy", k)
        assert len(results) <= k
        # Nothing from the file that does not mention either term.
        assert "c.md" not in sources(results)


def test_a_query_matching_nothing_returns_no_chunks_rather_than_filler(
    tmp_config, corpus_dir
):
    """Req 8.5 needs "no matching content" to be sayable, so it must be empty."""
    write(corpus_dir, "a.md", "harbour tides and moorings")

    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    assert retriever.search("zzzqqq", 5) == []


def test_results_name_the_corpus_file_each_passage_came_from(tmp_config, corpus_dir):
    """Req 7.8 and Req 8.9: every source is a file the walk just found."""
    write(corpus_dir, "notes/deep.md", "the mooring fee is forty euro")
    write(corpus_dir, "top.txt", "the mooring fee is unchanged")

    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)
    results = retriever.search("mooring fee", 10)

    assert set(sources(results)) <= {"notes/deep.md", "top.txt"}
    assert all((corpus_dir / source).is_file() for source in sources(results))
    # The pair shape is the whole return contract: (chunk text, source name).
    assert all(isinstance(text, str) and text for text, _ in results)


def test_ties_are_broken_by_source_name_then_position(tmp_config, corpus_dir):
    """Req 7.5: identical text in two files still has one documented order."""
    body = "mooring fee " * 40  # two chunks at a 200-character target
    write(corpus_dir, "b.md", body)
    write(corpus_dir, "a.md", body)

    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)
    results = retriever.search("mooring fee", MAX_RESULT_COUNT)

    assert sources(results) == ["a.md", "a.md", "a.md", "b.md", "b.md", "b.md"]
    # Equal scores, so position decides within a file: the chunks arrive in
    # document order.
    chunks, _ = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)
    in_order = [chunk.text for chunk in chunks if chunk.source_name == "a.md"]
    assert [text for text, source in results if source == "a.md"] == in_order


def test_a_repeated_query_over_an_unchanged_corpus_returns_the_same_list(
    tmp_config, corpus_dir
):
    """Req 7.5: repeated identical queries return the same ordering."""
    write(corpus_dir, "a.md", "berth allocation and mooring fees")
    write(corpus_dir, "b.md", "mooring fees are reviewed yearly")

    config = dataclasses.replace(tmp_config, chunk_size=200, chunk_overlap=0)
    first = KeywordRetriever(config).search("mooring fees", 5)

    # Twice through one retriever (the cached chunks) and once through a fresh
    # one (a rebuilt cache), because only the second rules out an ordering that
    # depends on build order.
    repeated = KeywordRetriever(config)
    assert repeated.search("mooring fees", 5) == repeated.search("mooring fees", 5)
    assert repeated.search("mooring fees", 5) == first


def test_a_corpus_edited_between_searches_is_searched_as_it_is_now(
    tmp_config, corpus_dir
):
    """The cache is keyed on a corpus fingerprint, so a mid-run edit is seen."""
    write(corpus_dir, "a.md", "the old wording")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)
    assert retriever.search("wording", 5) != []

    write(corpus_dir, "b.md", "a brand new document about pontoons")
    assert sources(retriever.search("pontoons", 5)) == ["b.md"]


def test_an_injected_chunk_source_is_searched_instead_of_the_corpus(
    tmp_config, corpus_dir
):
    """Req 7.6's wiring seam: the loader's chunks drop in without conversion."""
    write(corpus_dir, "on-disk.md", "pontoons and pilings")
    injected = [
        Chunk(text="pontoons from the loader", source_name="injected.md", position=0)
    ]

    retriever = KeywordRetriever(
        dataclasses.replace(tmp_config, chunk_size=200, chunk_overlap=0),
        chunk_source=lambda: injected,
    )

    assert retriever.search("pontoons", 5) == [
        ("pontoons from the loader", "injected.md")
    ]


def test_a_file_the_loader_would_refuse_to_decode_is_still_searchable(
    tmp_config, corpus_dir
):
    """The documented difference: the fallback answers now, with replacement.

    The loader is stricter because it writes a persisted collection and owes
    the operator a skip line; this searcher is answering a question.
    """
    (corpus_dir / "legacy.txt").write_bytes(b"caf\xe9 mooring fee")

    _, report = load_chunks(corpus_dir, chunk_size=200, chunk_overlap=0)
    assert [name for name, _ in report.files_skipped] == ["legacy.txt"]

    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)
    assert sources(retriever.search("mooring fee", 5)) == ["legacy.txt"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [("", "empty"), ("   ", "only whitespace"), ("\n\t ", "only whitespace")],
)
def test_an_empty_or_whitespace_query_is_refused(tmp_config, corpus_dir, query, expected):
    """Req 7.11: no chunks, and an error naming the parameter and its range."""
    write(corpus_dir, "a.md", "content")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    with pytest.raises(RetrievalError) as excinfo:
        retriever.search(query, 4)

    message = str(excinfo.value)
    assert message.startswith("query must be")
    assert expected in message
    assert f"1 to {MAX_QUERY_CHARS} characters" in message
    assert "no retrieval was performed" in message


def test_a_query_at_the_bound_is_accepted_and_one_character_longer_is_not(
    tmp_config, corpus_dir
):
    """Req 7.5 states the accepted length; the bound itself is inside it."""
    write(corpus_dir, "a.md", "content")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    assert retriever.search("q" * MAX_QUERY_CHARS, 4) == []
    with pytest.raises(RetrievalError) as excinfo:
        retriever.search("q" * (MAX_QUERY_CHARS + 1), 4)
    assert str(MAX_QUERY_CHARS + 1) in str(excinfo.value)


@pytest.mark.parametrize("query", [None, 7, b"bytes", ["a"]])
def test_a_query_that_is_not_a_string_is_refused(tmp_config, corpus_dir, query):
    write(corpus_dir, "a.md", "content")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    with pytest.raises(RetrievalError) as excinfo:
        retriever.search(query, 4)
    assert type(query).__name__ in str(excinfo.value)


@pytest.mark.parametrize("k", [0, -1, MAX_RESULT_COUNT + 1, 1_000])
def test_a_result_count_outside_its_range_is_refused(tmp_config, corpus_dir, k):
    """Req 7.11: rejected at this layer, not clamped — clamping is the tool's."""
    write(corpus_dir, "a.md", "content")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    with pytest.raises(RetrievalError) as excinfo:
        retriever.search("content", k)

    message = str(excinfo.value)
    assert message.startswith("k (the result count) must be")
    assert f"between {MIN_RESULT_COUNT} and {MAX_RESULT_COUNT} inclusive" in message
    assert f"received {k}" in message


@pytest.mark.parametrize("k", ["4", 4.0, 2.5, True, None, [4]])
def test_a_result_count_that_is_not_an_integer_is_refused(tmp_config, corpus_dir, k):
    """``True`` included: a boolean result count is a caller mistake."""
    write(corpus_dir, "a.md", "content")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    with pytest.raises(RetrievalError) as excinfo:
        retriever.search("content", k)
    assert type(k).__name__ in str(excinfo.value)


def test_a_rejected_call_reads_no_file(tmp_config, corpus_dir, monkeypatch):
    """Req 7.11: "returns no chunks" is also "performs no retrieval"."""
    write(corpus_dir, "a.md", "content")
    retriever = retriever_for(tmp_config, chunk_size=200, chunk_overlap=0)

    def unreachable(*args: object, **kwargs: object) -> list[Path]:
        raise AssertionError("the corpus was read despite an invalid parameter")

    monkeypatch.setattr(
        "agent_lab.retrieval.keyword.corpus_files", unreachable, raising=True
    )

    with pytest.raises(RetrievalError):
        retriever.search("  ", 4)
    with pytest.raises(RetrievalError):
        retriever.search("content", 0)


# --------------------------------------------------------------------------- #
# Scoring (the pure half of the fallback)
# --------------------------------------------------------------------------- #


def test_normalise_folds_case_and_collapses_whitespace():
    assert normalise("  The\tRetention\n\nPolicy  ") == "the retention policy"


def test_query_terms_drops_duplicates_and_stop_words():
    assert query_terms("the cost of the run") == ["cost", "run"]
    assert query_terms("fee fee FEE") == ["fee"]


def test_a_query_of_nothing_but_stop_words_is_still_searched_for():
    """Dropping every term would make the query match everything."""
    assert query_terms("what is it") == ["what", "is", "it"]


def test_a_chunk_matching_nothing_scores_zero():
    assert score_chunk("harbour tides", ["pontoon"], "pontoon") == 0.0
    assert score_chunk("   ", ["tides"], "tides") == 0.0


def test_the_whole_phrase_scores_above_the_same_words_scattered():
    """The substring half of the search, and the reason it exists."""
    terms = query_terms("retention policy")
    phrase = normalise("retention policy")
    together = score_chunk("our retention policy is annual", terms, phrase)
    apart = score_chunk("retention is discussed; the policy is elsewhere", terms, phrase)

    assert together > apart > 0.0
    assert 0.0 < together <= 1.0


def test_covering_more_query_terms_outscores_repeating_one():
    terms = query_terms("mooring fee schedule")
    phrase = normalise("mooring fee schedule")
    covering = score_chunk("mooring fee schedule", terms, phrase)
    repeating = score_chunk("mooring mooring mooring mooring mooring", terms, phrase)

    assert covering > repeating


# --------------------------------------------------------------------------- #
# Selecting a retriever (Req 7.6)
# --------------------------------------------------------------------------- #


def test_no_store_selects_the_keyword_fallback_and_creates_nothing(
    tmp_config, corpus_dir, chroma_dir
):
    """Req 7.6 on a fresh clone: the fallback, and no ``.chroma`` left behind."""
    write(corpus_dir, "a.md", "content")
    assert not chroma_dir.exists()

    retriever = select_retriever(tmp_config)

    assert isinstance(retriever, KeywordRetriever)
    assert collection_size(tmp_config) is None
    assert has_index(tmp_config) is False
    assert store_exists(chroma_dir) is False
    assert not chroma_dir.exists()


def test_a_store_that_cannot_be_opened_selects_the_fallback(tmp_config, chroma_dir):
    """An unusable index is answered with the fallback, not an exception."""
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").write_bytes(b"not a database")

    assert store_exists(chroma_dir) is True
    assert collection_size(tmp_config) is None
    assert isinstance(select_retriever(tmp_config), KeywordRetriever)


def test_a_built_index_is_selected_and_a_rebuild_leaves_no_prior_chunk(
    tmp_config, corpus_dir, chroma_dir
):
    """Req 7.3, 7.4, 7.6: the vector path, when the machine can embed.

    Skipped rather than marked ``allow_network``: the suite refuses every
    socket connect (Req 16.8) and Chroma's bundled embedding function fetches
    its model on first use, which arrives here as
    :class:`ChromaUnavailable` — the same signal that sends a real run to the
    keyword fallback.
    """
    write(corpus_dir, "keep.md", "the mooring fee is forty euro a night")
    write(corpus_dir, "drop.md", "the pontoon inspection happens in March")
    config = dataclasses.replace(tmp_config, chunk_size=200, chunk_overlap=0)

    try:
        first = build_index(config)
    except ChromaUnavailable as exc:
        pytest.skip(f"the Chroma path is unavailable in this environment: {exc}")

    assert first.files_ingested == 2
    assert first.chunks_indexed == 2
    assert store_exists(chroma_dir) is True
    assert collection_size(config) == 2
    assert has_index(config) is True

    retriever = select_retriever(config)
    assert isinstance(retriever, ChromaRetriever)
    assert isinstance(retriever, Retriever)

    try:
        results = retriever.search("mooring fee", 2)
    except ChromaUnavailable as exc:
        pytest.skip(f"the Chroma path is unavailable in this environment: {exc}")

    assert len(results) <= 2
    assert set(sources(results)) <= {"keep.md", "drop.md"}

    # Req 7.4: the second build replaces the collection, so the chunk from the
    # file that is no longer in the corpus cannot answer a query.
    (corpus_dir / "drop.md").unlink()
    second = build_index(config)

    assert second.files_ingested == 1
    assert collection_size(config) == 1
    rebuilt = select_retriever(config).search("pontoon inspection", 5)
    assert sources(rebuilt) == ["keep.md"] or rebuilt == []
    assert "drop.md" not in sources(rebuilt)


def test_a_build_with_an_out_of_range_setting_touches_no_store(
    tmp_config, corpus_dir, chroma_dir
):
    """Req 7.10: rejected before the persistence directory is even created."""
    write(corpus_dir, "a.md", "content")

    with pytest.raises(ChunkingError):
        build_index(dataclasses.replace(tmp_config, chunk_size=199))

    assert not chroma_dir.exists()
