"""Flat-text projection of a parsed document, and the chunker built on it.

This module used to *be* the parser. It imported ``pypdf``, ``python-docx`` and
``python-pptx`` at module scope, dispatched on the file extension and returned
a string — which meant every business-layer caller from the analysis
orchestrator to the upload route knew, at the import line, which three
third-party libraries this product was built on.

Phase 0 moved all three imports into
:mod:`paperwhisperer.ingestion.adapters.legacy_loader`, where the P0 exit
criterion says they belong, and made
:func:`paperwhisperer.ingestion.parse_document` the single way the business
layer turns a file into text. This module is what remains, and it is thin on
purpose: the analysis sections all take a flat string, and turning those
sections into structured, chunk-level consumers is T1.x's job, not this
module's. So the one thing the business layer still needs from here is the
flat text the prompts were written against — and exactly one derivation of it.

Three properties are load-bearing:

* **No parser libraries, not even in a function body.** The criterion is
  checked by parsing this file with :mod:`ast` and rejecting any import whose
  root is ``pypdf``, ``docx`` or ``pptx``, at module level *or* nested. The
  nested case is the one that matters: a function-local import is the cheapest
  possible way for the criterion to rot back, and it would still work, so
  nothing else would notice.
* **One parse per file, and the outcome travels with the text.**
  :meth:`DocumentLoader.load_with_outcome` hands back the flat text *and* the
  :class:`~paperwhisperer.ingestion.ParseOutcome` that produced it, so a
  caller that must tell the user which parser ran and whether their file was
  uploaded to a cloud service (T3.4) never has to parse a second time to find
  out. That is also why this is a method and not a bare string return: the
  old signature could not carry the answer, and dropping the answer is how
  "was my document sent somewhere?" becomes unanswerable after the fact.
* **The legacy errors survive the move.** ``load`` still raises ``ValueError``
  with the same Chinese messages, because the upload route turns a
  ``ValueError`` into a 400 and a caller matching on that text is a caller we
  do not get to break in Phase 0.

:class:`TextChunker` stays exactly where it is and exactly as it is. Chunking
is not a parser concern: it runs on whatever prose it is handed, it did so
before this migration and it will after, and relocating it would have bought a
tidier module boundary at the cost of a behaviour change nobody asked for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from paperwhisperer.core import config
from paperwhisperer.core.text import clean_extracted_text
from paperwhisperer.ingestion import ParseOutcome, parse_document


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """A parsed file as the analysis layer needs it: flat text, plus provenance.

    The two fields are one result, not two. The text answers "what do the
    prompts see"; the outcome answers "who produced it, and did it leave the
    machine". Splitting them into two calls would either make callers parse
    twice or make the second question unanswerable, and a document that was
    uploaded somewhere is exactly the fact a user needs to be told.
    """

    text: str
    outcome: ParseOutcome

    @property
    def used_cloud(self) -> bool:
        """Whether the winning document came from the cloud parser."""
        return self.outcome.used_cloud


class DocumentLoader:
    """The flat-text façade over the ingestion gateway.

    Retained as a class, with the same static-method surface it has always had,
    because three call sites still name it: the upload-facing re-export in
    ``web_app``, the legacy adapter (which needs :meth:`load_txt` and nothing
    else), and the document tests. Only :meth:`load` and :meth:`load_txt` are
    kept. ``load_pdf`` / ``load_docx`` / ``load_pptx`` are gone: they existed
    only to be dispatched to by :meth:`load`, they are the reason this module
    imported three binary-document libraries, and nothing outside the legacy
    adapter has ever called them directly.
    """

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Read a plain-text file verbatim.

        Kept as a real reader, not a gateway call, because the legacy adapter
        calls it while *building* its ``ParsedDocument`` — routing it through
        ``parse_document`` would make the adapter recurse into itself. It is
        the one format whose extraction has no library behind it, so keeping it
        here costs the boundary nothing: the whole point of the migration was
        to move ``pypdf``, not to move ``open()``.
        """
        with open(file_path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()

    @staticmethod
    def load(file_path: str) -> str:
        """Return the flat text for ``file_path``, raising ``ValueError`` on failure.

        The compatibility entry point, kept so that callers which only ever
        wanted a string keep working unchanged. It is :meth:`load_with_outcome`
        with the provenance dropped, and it is not what new code should call —
        anything that reports on a parse to a user needs the outcome, and
        should ask for it.
        """
        return DocumentLoader.load_with_outcome(file_path).text

    @staticmethod
    def load_with_outcome(file_path: str) -> LoadedDocument:
        """Parse ``file_path`` through the gateway and flatten the result.

        The extension is checked before the gateway is entered rather than
        after. Two reasons, and the second is the one that matters: the message
        the user gets is *ours* to keep stable, and the gateway's own answer
        for an unknown type is a different sentence about media types rather
        than about this product's supported list. It also means an upload the
        gate will never accept costs a stat call rather than a full parse
        attempt, a file hash and a parse-cache lookup.
        """
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in config.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {extension} (支持: {config.SUPPORTED_FILE_TYPES_TEXT})"
            )

        outcome = parse_document(file_path)
        if not outcome.is_usable or outcome.document is None:
            # ``error_message`` is the orchestrator's composed, user-readable
            # sentence: which stages ran, which parser each one reached for,
            # and why. A bare "parse failed" here would throw away the only
            # diagnostic a user can act on.
            raise ValueError(outcome.error_message or "文档解析失败。")

        return LoadedDocument(
            text=clean_extracted_text(outcome.document.to_full_text()),
            outcome=outcome,
        )


class TextChunker:
    """Split long text into overlapping chunks, preferring sentence boundaries."""

    _SENTENCE_SEPARATORS = ("。", ". ", "\n")

    def __init__(self, chunk_size: int = 4000, overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            if end < text_length:
                best_pos = -1
                for separator in self._SENTENCE_SEPARATORS:
                    pos = chunk.rfind(separator)
                    if pos > best_pos:
                        best_pos = pos

                if best_pos > self.chunk_size // 2:
                    chunk = chunk[:best_pos + 1]
                    end = start + best_pos + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            # `end` only ever moves forward from start + 1, so the next start
            # is always >= 0; the max() keeps that invariant explicit.
            start = max(0, end - self.overlap)

        return chunks
