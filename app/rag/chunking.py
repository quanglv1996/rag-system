"""Text chunking strategies for the RAG pipeline.

Provides multiple strategies for splitting long documents into
semantically meaningful chunks that fit within LLM context windows.
The Strategy Pattern allows the chunk algorithm to be changed via config.
"""

from dataclasses import dataclass, field
from typing import Any

from app.common.enums import ChunkStrategy
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text from a document.

    Attributes:
        content: The chunk text.
        index: Zero-based position within the document.
        start_char: Character offset in the original text.
        end_char: End character offset in the original text.
        metadata: Inherited or chunk-specific metadata.
    """

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Strategy-based text chunker.

    Supports FIXED, RECURSIVE, and SENTENCE chunking strategies.
    The strategy is selected at construction time and can be swapped
    without modifying the pipeline.

    Attributes:
        _strategy: Selected chunking strategy.
        _chunk_size: Target chunk size in characters.
        _chunk_overlap: Overlap between consecutive chunks in characters.
    """

    def __init__(
        self,
        strategy: ChunkStrategy | str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize the chunker with a strategy and sizing parameters.

        Args:
            strategy: Chunking strategy. Defaults to RECURSIVE.
            chunk_size: Target chunk size in characters. Defaults to settings value.
            chunk_overlap: Overlap in characters. Defaults to settings value.
        """
        settings = get_settings()

        self._strategy = ChunkStrategy(strategy or ChunkStrategy.RECURSIVE)
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        logger.debug(
            "TextChunker initialized",
            strategy=self._strategy,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split text into chunks using the configured strategy.

        Args:
            text: Full document text to split.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            list[TextChunk]: Ordered list of text chunks.
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}

        if self._strategy == ChunkStrategy.FIXED:
            return self._fixed_chunk(text, base_metadata)
        elif self._strategy == ChunkStrategy.RECURSIVE:
            return self._recursive_chunk(text, base_metadata)
        elif self._strategy == ChunkStrategy.SENTENCE:
            return self._sentence_chunk(text, base_metadata)
        else:
            # Default fallback
            return self._recursive_chunk(text, base_metadata)

    def _fixed_chunk(
        self, text: str, metadata: dict[str, Any]
    ) -> list[TextChunk]:
        """Split text into fixed-size character chunks with overlap.

        Args:
            text: Input text.
            metadata: Base metadata for all chunks.

        Returns:
            list[TextChunk]: Fixed-size chunks.
        """
        chunks: list[TextChunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        index=index,
                        start_char=start,
                        end_char=end,
                        metadata={**metadata, "chunk_strategy": "fixed"},
                    )
                )
                index += 1

            start = end - self._chunk_overlap
            if start >= end:
                break

        return chunks

    def _recursive_chunk(
        self, text: str, metadata: dict[str, Any]
    ) -> list[TextChunk]:
        """Split text recursively on paragraph, newline, and sentence boundaries.

        This is the most balanced strategy — it tries to keep semantic
        units (paragraphs, sentences) together before splitting at characters.

        Args:
            text: Input text.
            metadata: Base metadata for all chunks.

        Returns:
            list[TextChunk]: Recursively split chunks.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )

            texts = splitter.split_text(text)
            chunks: list[TextChunk] = []
            offset = 0

            for idx, chunk_text in enumerate(texts):
                # Find the actual position in original text
                start = text.find(chunk_text, offset)
                if start == -1:
                    start = offset
                end = start + len(chunk_text)
                offset = max(start, offset)  # Track position

                chunks.append(
                    TextChunk(
                        content=chunk_text.strip(),
                        index=idx,
                        start_char=start,
                        end_char=end,
                        metadata={**metadata, "chunk_strategy": "recursive"},
                    )
                )

            return [c for c in chunks if c.content]

        except ImportError:
            # Fallback if LangChain is not available
            logger.warning("LangChain not available, falling back to fixed chunking")
            return self._fixed_chunk(text, metadata)

    def _sentence_chunk(
        self, text: str, metadata: dict[str, Any]
    ) -> list[TextChunk]:
        """Split text into chunks that respect sentence boundaries.

        Accumulates sentences until the chunk_size is reached,
        then starts a new chunk. Respects overlap by including
        trailing sentences from the previous chunk.

        Args:
            text: Input text.
            metadata: Base metadata for all chunks.

        Returns:
            list[TextChunk]: Sentence-boundary-aligned chunks.
        """
        import re

        # Split on sentence-ending punctuation
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(text)

        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_length = 0
        index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If adding this sentence would exceed the chunk size
            if current_length + sentence_len > self._chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                start_char = text.find(current_sentences[0])
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        index=index,
                        start_char=max(0, start_char),
                        end_char=max(0, start_char) + len(chunk_text),
                        metadata={**metadata, "chunk_strategy": "sentence"},
                    )
                )
                index += 1

                # Keep overlap sentences for next chunk
                overlap_chars = 0
                overlap_sentences: list[str] = []
                for s in reversed(current_sentences):
                    if overlap_chars + len(s) <= self._chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_chars += len(s)
                    else:
                        break

                current_sentences = overlap_sentences
                current_length = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sentence_len

        # Handle the last remaining chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = text.find(current_sentences[0])
            chunks.append(
                TextChunk(
                    content=chunk_text,
                    index=index,
                    start_char=max(0, start_char),
                    end_char=max(0, start_char) + len(chunk_text),
                    metadata={**metadata, "chunk_strategy": "sentence"},
                )
            )

        return chunks
