"""Unit tests for the chunker (Req 7.2, Req 7.10).

Specific examples and edge cases only. The universal statement — Property 8,
"chunking preserves the document and respects its bounds" — is task 9.2.
"""

import pytest

from agent_lab.config import INT_RANGES, ConfigError
from agent_lab.retrieval.chunking import (
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    ChunkingError,
    chunk,
    validate_params,
)


def reassemble(chunks: list[str], overlap: int) -> str:
    """Drop the repeated overlap from every chunk after the first, join."""
    if not chunks:
        return ""
    return chunks[0] + "".join(part[overlap:] for part in chunks[1:])


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


def test_text_shorter_than_one_window_is_a_single_chunk():
    text = "a" * 500
    assert chunk(text, 1000, 150) == [text]


def test_text_exactly_one_window_long_is_a_single_chunk():
    # The boundary case the loop's stop condition exists for: a second window
    # would repeat text the first already covered in full.
    text = "b" * 1000
    assert chunk(text, 1000, 150) == [text]


def test_windows_advance_by_size_minus_overlap():
    text = "".join(str(i % 10) for i in range(2500))
    chunks = chunk(text, 1000, 150)
    assert chunks[0] == text[0:1000]
    assert chunks[1] == text[850:1850]
    assert chunks[2] == text[1700:2500]
    assert len(chunks) == 3


def test_no_chunk_exceeds_the_target_length():
    text = "x" * 9_999
    chunks = chunk(text, 200, 199)
    assert max(len(part) for part in chunks) == 200


def test_every_chunk_but_the_last_is_exactly_the_target_length():
    text = "y" * 3_333
    chunks = chunk(text, 1000, 150)
    assert [len(part) for part in chunks[:-1]] == [1000] * (len(chunks) - 1)
    assert 0 < len(chunks[-1]) <= 1000


def test_the_last_chunk_ends_at_the_end_of_the_text():
    text = "".join(str(i % 7) for i in range(4_321))
    chunks = chunk(text, 400, 50)
    assert text.endswith(chunks[-1])


def test_removing_the_overlap_reproduces_the_text_exactly():
    text = "Chroma stores vectors on disk.\n" * 300  # 9000 characters
    for size, overlap in ((200, 0), (200, 199), (1000, 150), (257, 1), (4000, 1)):
        chunks = chunk(text, size, overlap)
        assert reassemble(chunks, overlap) == text, (size, overlap)


def test_reassembly_holds_when_the_overlap_is_nearly_the_whole_window():
    # The tightest case for the "last chunk is longer than the overlap"
    # argument: one step past the first window.
    text = "m" * 4_001
    chunks = chunk(text, 4000, 3999)
    assert len(chunks) == 2
    assert reassemble(chunks, 3999) == text


def test_zero_overlap_partitions_the_text_without_repetition():
    text = "z" * 1_050
    chunks = chunk(text, 200, 0)
    assert "".join(chunks) == text
    assert len(chunks) == 6


def test_chunk_positions_are_contiguous_from_zero():
    # The loader stores a chunk's index in this list as its zero-based
    # position, so the list order *is* the position sequence (Req 7.2).
    chunks = chunk("w" * 5_000, 1000, 100)
    assert list(range(len(chunks))) == [i for i, _ in enumerate(chunks)]
    assert len(chunks) == 6


# --------------------------------------------------------------------------- #
# Edge cases and purity
# --------------------------------------------------------------------------- #


def test_empty_text_yields_no_chunks():
    assert chunk("", 1000, 150) == []


def test_whitespace_only_text_is_returned_verbatim():
    # Deciding a document holds nothing worth indexing is the loader's job
    # (Req 7.9); the chunker does not strip or normalise.
    text = "  \n\t "
    assert chunk(text, 200, 0) == [text]


def test_single_character_text_yields_one_chunk():
    assert chunk("q", 200, 199) == ["q"]


def test_repeated_calls_return_equal_results():
    text = "determinism matters\n" * 400
    assert chunk(text, 1000, 150) == chunk(text, 1000, 150)


def test_the_input_text_is_not_mutated_or_consumed():
    text = "unicode: café — naïve 漢字\n" * 200
    before = text
    chunk(text, 300, 30)
    assert text == before


def test_text_is_sliced_by_characters_not_bytes():
    text = "漢" * 450
    chunks = chunk(text, 200, 0)
    assert [len(part) for part in chunks] == [200, 200, 50]
    assert "".join(chunks) == text


# --------------------------------------------------------------------------- #
# Range rejection (Req 7.10)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("size", [0, 1, 199, 4_001, -1000])
def test_out_of_range_size_is_rejected(size):
    with pytest.raises(ChunkingError) as excinfo:
        chunk("text", size, 0)
    message = str(excinfo.value)
    assert "chunk size" in message
    assert str(size) in message
    assert "200" in message and "4000" in message.replace(",", "")


@pytest.mark.parametrize("overlap", [-1, 1_000, 1_001, 5_000])
def test_out_of_range_overlap_is_rejected(overlap):
    with pytest.raises(ChunkingError) as excinfo:
        chunk("text", 1000, overlap)
    message = str(excinfo.value)
    assert "chunk overlap" in message
    assert str(overlap) in message
    assert "0 and 999" in message


@pytest.mark.parametrize("bad", ["1000", 1000.0, None, True])
def test_non_integer_parameters_are_rejected(bad):
    with pytest.raises(ChunkingError):
        chunk("text", bad, 0)
    with pytest.raises(ChunkingError):
        chunk("text", 1000, bad)


def test_the_accepted_bounds_are_accepted():
    assert len(chunk("t" * 10, MIN_CHUNK_SIZE, 0)) == 1
    assert len(chunk("t" * 10, MAX_CHUNK_SIZE, MAX_CHUNK_SIZE - 1)) == 1


def test_overlap_equal_to_size_is_rejected():
    # Not merely out of range: step would be 0 and the loop would never end.
    with pytest.raises(ChunkingError):
        chunk("t" * 5_000, 1000, 1000)


def test_rejection_names_the_size_before_the_overlap():
    # Both are invalid; the overlap's accepted range is stated in terms of the
    # size, so a nonsense size is reported first.
    with pytest.raises(ChunkingError) as excinfo:
        chunk("text", 10, 99)
    assert "chunk size" in str(excinfo.value)


def test_validate_params_returns_the_checked_values():
    assert validate_params(1000, 150) == (1000, 150)


def test_validate_params_rejects_without_chunking():
    # The index command calls this before it opens the store, which is what
    # leaves an existing collection unchanged after a rejected build.
    with pytest.raises(ChunkingError):
        validate_params(100, 0)


def test_chunking_error_is_a_config_error():
    # So the CLI's single `except ConfigError` in main covers it and exits
    # with the configuration exit code.
    assert issubclass(ChunkingError, ConfigError)


def test_bounds_agree_with_the_configured_range():
    assert INT_RANGES["CHUNK_SIZE"] == (MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)


def test_non_string_text_is_a_type_error():
    with pytest.raises(TypeError):
        chunk(b"bytes are not text", 1000, 150)
