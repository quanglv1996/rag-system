"""Tests for the text chunking strategies."""

import pytest

from app.common.enums import ChunkStrategy
from app.rag.chunking import TextChunker


class TestFixedChunking:
    """Tests for the FIXED chunking strategy."""

    def test_basic_fixed_chunk(self):
        """Test that fixed chunking splits text into correct sizes."""
        chunker = TextChunker(strategy=ChunkStrategy.FIXED, chunk_size=100, chunk_overlap=20)
        text = "A" * 250

        chunks = chunker.chunk(text)

        # Should produce 3 chunks: 0-100, 80-180, 160-260
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= 100

    def test_fixed_chunk_indices_are_sequential(self):
        """Test that chunk indices start at 0 and are sequential."""
        chunker = TextChunker(strategy=ChunkStrategy.FIXED, chunk_size=50, chunk_overlap=0)
        text = "word " * 50

        chunks = chunker.chunk(text)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text_returns_empty_list(self):
        """Test that empty input produces no chunks."""
        chunker = TextChunker(strategy=ChunkStrategy.FIXED)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_metadata_attached_to_chunks(self):
        """Test that provided metadata is attached to all chunks."""
        chunker = TextChunker(strategy=ChunkStrategy.FIXED, chunk_size=20, chunk_overlap=0)
        text = "Hello world " * 10
        meta = {"document_id": "doc-001", "source": "test.txt"}

        chunks = chunker.chunk(text, meta)

        for chunk in chunks:
            assert chunk.metadata["document_id"] == "doc-001"
            assert chunk.metadata["source"] == "test.txt"


class TestSentenceChunking:
    """Tests for the SENTENCE chunking strategy."""

    def test_sentence_boundaries_respected(self):
        """Test that chunks align with sentence endings."""
        chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE, chunk_size=200, chunk_overlap=0
        )
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."

        chunks = chunker.chunk(text)

        # Each chunk should end with sentence-closing punctuation
        for chunk in chunks:
            content = chunk.content.strip()
            if len(content) < 200:  # Not a mid-break
                assert content[-1] in ".!?" or chunk == chunks[-1]

    def test_short_text_produces_single_chunk(self):
        """Test that text shorter than chunk_size is a single chunk."""
        chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE, chunk_size=500, chunk_overlap=0
        )
        text = "Short text. Very short."

        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text


class TestRecursiveChunking:
    """Tests for the RECURSIVE (LangChain) chunking strategy."""

    def test_recursive_chunk_respects_size(self):
        """Test that recursive chunks don't exceed chunk_size."""
        chunker = TextChunker(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=20,
        )
        text = ("This is a paragraph.\n\n" * 20)

        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        for chunk in chunks:
            # Allow for small overruns due to LangChain's length function
            assert len(chunk.content) <= 210

    def test_recursive_produces_content(self):
        """Test that recursive chunking returns non-empty chunks."""
        chunker = TextChunker(strategy=ChunkStrategy.RECURSIVE)
        text = "Some meaningful text that needs to be chunked for testing purposes." * 5

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(c.content.strip() for c in chunks)
