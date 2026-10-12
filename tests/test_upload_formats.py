"""Which files this deployment accepts, and who decides.

The answer used to be written down five times -- the upload gate, the upload
contract, the router's extension table, the legacy adapter's support set, and
the frontend's ``accept`` attribute -- and the copies had drifted in a way that
only bites the format somebody adds next:

* the gate and the contract accepted a declared ``text/markdown`` and the
  contract normalised it to ``text/plain``;
* the router sent the same file down with the ``mimetypes`` answer for ``.md``,
  which is ``text/markdown``;
* no adapter declared ``text/markdown``, so routing found no candidate.

A ``.txt`` uploaded with ``Content-Type: text/markdown`` therefore passed both
validation layers and then failed to parse, naming no parser. The tables now
come from :mod:`paperwhisperer.core.formats`, and these tests hold the two
properties that keep them from drifting apart again: every layer reads the same
object, and the gate never accepts an extension no parser can read.

The rest of the file covers what markdown had to become to be worth accepting.
Recording it as text and calling it done would have thrown away the one thing
that makes it a better upload than a ``.txt``: it has headings, and headings are
what the chunker carries into a citation.
"""

from __future__ import annotations

import io

import pytest
from pydantic import ValidationError

from paperwhisperer.core import config, formats
from paperwhisperer.documents import validation
from paperwhisperer.ingestion.adapters.legacy_loader import (
    _EXTENSION_BY_FORMAT,
    LegacyDocumentParser,
)
from paperwhisperer.ingestion.contracts import _UPLOAD_MEDIA_TYPES, UploadRequestContract
from paperwhisperer.ingestion.gateway import MEDIA_TYPE_BY_EXTENSION
from paperwhisperer.ingestion.orchestrator import parse_document

MARKDOWN = (
    "# Findings\n"
    "\n"
    "Prose under the level-one heading.\n"
    "\n"
    "## Method\n"
    "\n"
    "```python\n"
    "# a comment, not a heading\n"
    "print('x')\n"
    "```\n"
    "\n"
    "### Detail\n"
    "\n"
    "## Results\n"
    "\n"
    "#hashtag is body text because no space follows the hash.\n"
)


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _elements(path):
    doc = parse_document(str(path), media_type=None).document
    assert doc is not None
    return doc


class TestOneTableDecidesTheAcceptedFormats:
    def test_every_layer_reads_the_same_table(self):
        # Identity, not equality: a copy that happens to match today is the
        # mechanism by which these drifted apart in the first place.
        assert config.SUPPORTED_EXTENSIONS is formats.SUPPORTED_EXTENSIONS
        assert MEDIA_TYPE_BY_EXTENSION is formats.EXTENSION_MEDIA_TYPES
        assert _UPLOAD_MEDIA_TYPES is formats.EXTENSION_MEDIA_TYPES
        assert validation._GENERIC_BINARY_CONTENT_TYPES is formats.NO_CLAIM_MEDIA_TYPES

    def test_the_gate_never_accepts_a_file_no_parser_claims(self):
        """The invariant the drift broke, stated once.

        An extension the upload gate accepts but ``_EXTENSION_BY_FORMAT`` has no
        builder for is an upload that is stored, hashed, and then fails to
        parse -- with the error arriving long after the moment the user could
        do anything about it.
        """
        accepted = set(formats.EXTENSION_MEDIA_TYPES)
        assert accepted == set(_EXTENSION_BY_FORMAT)

    def test_every_canonical_media_type_is_one_the_adapter_declares(self):
        parser = LegacyDocumentParser()
        for extension, media_type in formats.EXTENSION_MEDIA_TYPES.items():
            assert parser.supports(media_type), f"{extension} routes to {media_type}, unclaimed"

    def test_the_extension_always_resolves_to_the_canonical_type(self):
        for extension, media_type in formats.EXTENSION_MEDIA_TYPES.items():
            assert formats.media_type_for_extension(extension) == media_type

    def test_markdown_is_recorded_as_text_and_read_as_markdown(self):
        # The deliberate part: the media type says text, the extension picks the
        # reader. Both halves are asserted because either one alone would still
        # pass if the other silently stopped being true.
        assert formats.media_type_for_extension(".md") == "text/plain"
        assert _EXTENSION_BY_FORMAT[".md"] is LegacyDocumentParser._parse_markdown


class TestADeclaredTypeIsNormalisedBeforeRouting:
    def test_a_markdown_declaration_routes_a_text_file(self, tmp_path):
        """The regression: this passed validation and then failed to parse.

        A browser is entitled to describe a ``.txt`` as ``text/markdown``, and
        the contract said so while the router disagreed.
        """
        path = _write(tmp_path, "notes.txt", "A paragraph.\n\nAnother paragraph.\n")
        out = parse_document(str(path), media_type="text/markdown")

        assert out.status == "success"
        assert out.document is not None
        assert out.document.parser_name == "legacy"

    def test_a_pdf_alias_reaches_the_pdf_adapter(self):
        # ``application/x-pdf`` is the spelling the contract has always accepted
        # and the reason its alias table exists; before normalisation the
        # adapter compared literals, missed it, and the bytes went down the
        # plain-text path. Both halves matter: the alias has to become the
        # canonical type, and the adapter has to claim that type.
        canonical = formats.canonical_media_type("application/x-pdf")

        assert canonical == formats.EXTENSION_MEDIA_TYPES[".pdf"]
        assert LegacyDocumentParser().supports(canonical)

    @pytest.mark.parametrize(
        "declared",
        ["text/markdown", "text/markdown; charset=utf-8", "TEXT/MARKDOWN", " text/markdown "],
    )
    def test_parameters_case_and_padding_do_not_defeat_the_table(self, declared):
        assert formats.canonical_media_type(declared) == "text/plain"

    def test_a_type_whose_meaning_needs_the_filename_is_left_to_the_extension(self):
        # ``application/zip`` is a legitimate declaration for a ``.docx`` *and*
        # a ``.pptx``, so it cannot have one canonical form. Answering "" sends
        # the caller to the extension, which is where the answer actually is.
        assert formats.canonical_media_type("application/zip") == ""
        assert formats.canonical_media_type("image/png") == ""

    def test_an_untaught_declaration_still_parses_by_extension(self, tmp_path):
        path = _write(tmp_path, "notes.md", MARKDOWN)
        out = parse_document(str(path), media_type="application/octet-stream")

        assert out.status == "success"
        assert out.document is not None


class TestMarkdownArrivesAsStructure:
    def test_each_level_becomes_a_heading_carrying_its_path(self, tmp_path):
        doc = _elements(_write(tmp_path, "notes.md", MARKDOWN))
        headings = [e for e in doc.elements if e.role == "heading"]

        assert [(e.text, e.level) for e in headings] == [
            ("# Findings", 1),
            ("## Method", 2),
            ("### Detail", 3),
            ("## Results", 2),
        ]
        # The level-two heading that follows the level-three one replaces it and
        # the level-two one above it: a path, not a log.
        assert headings[-1].heading_path == ("Findings", "Results")

    def test_the_markers_are_dropped_from_the_path_and_kept_in_the_text(self, tmp_path):
        """Why the heading text still reads ``# Findings``.

        The offsets are the citation handle, and the element text has to begin
        where ``offset_start`` says it does. Stripping the markers here would
        leave a heading whose text starts three characters after its own offset
        -- a trap for whoever aligns a quote against it next.
        """
        doc = _elements(_write(tmp_path, "notes.md", MARKDOWN))
        heading = next(e for e in doc.elements if e.role == "heading")

        assert heading.text == "# Findings"
        assert heading.heading_path == ("Findings",)

    def test_a_hashtag_is_body_text(self, tmp_path):
        doc = _elements(_write(tmp_path, "notes.md", MARKDOWN))
        body = next(e for e in doc.elements if e.text.startswith("#hashtag"))

        assert body.role == "paragraph"

    def test_a_fence_is_one_element_and_hides_what_is_inside_it(self, tmp_path):
        doc = _elements(_write(tmp_path, "notes.md", MARKDOWN))
        fenced = [e for e in doc.elements if e.text.startswith("```")]

        assert len(fenced) == 1
        assert "# a comment, not a heading" in fenced[0].text
        assert fenced[0].role == "paragraph"
        assert "a comment" not in [e.text for e in doc.elements if e.role == "heading"]

    def test_every_offset_still_points_at_its_own_text(self, tmp_path):
        from paperwhisperer.core.text import clean_extracted_text
        from paperwhisperer.documents.loader import DocumentLoader

        path = _write(tmp_path, "notes.md", MARKDOWN)
        doc = _elements(path)
        source = clean_extracted_text(DocumentLoader.load_txt(str(path)))

        for element in doc.elements:
            locator = element.locator
            assert source[locator.offset_start:locator.offset_end].strip() == element.text

    def test_the_broken_hierarchy_one_still_nests(self, tmp_path):
        doc = _elements(_write(tmp_path, "notes.md", MARKDOWN))
        results = next(e for e in doc.elements if e.text == "## Results")

        assert results.heading_path == ("Findings", "Results")
        assert "Method" not in results.heading_path


class TestADeclarationOfIgnoranceIsNotAContradiction:
    """``application/octet-stream`` says "I do not know", not "it is something else".

    A Windows browser has no MIME registration for ``.md`` and sends exactly
    that, so refusing it made the format unreachable from a browser even after
    everything else worked. The header is not the defence that holds here -- the
    magic-byte check is -- which the two refusals below show is still in force.
    """

    def test_the_contract_accepts_it(self):
        for head in sorted(formats.NO_CLAIM_MEDIA_TYPES):
            contract = UploadRequestContract(filename="notes.md", media_type=head)
            assert contract.media_type == ""

    def test_a_declaration_that_does_contradict_the_file_is_still_refused(self):
        with pytest.raises(ValidationError):
            UploadRequestContract(filename="notes.md", media_type="image/png")

    def test_an_extension_outside_the_table_is_still_refused(self, client):
        response = client.post(
            "/v1/documents",
            files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_file_type"

    def test_a_mislabelled_pdf_is_still_refused_by_its_bytes(self, client):
        # The check that matters, and the one the header no longer short-circuits.
        response = client.post(
            "/v1/documents",
            files={"file": ("fake.pdf", io.BytesIO(MARKDOWN.encode()), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "upload_rejected"
        assert "valid PDF" in response.json()["error"]

    def test_a_markdown_upload_is_accepted_without_a_content_type(self, client):
        response = client.post(
            "/v1/documents",
            files={"file": ("notes.md", io.BytesIO(MARKDOWN.encode()), "application/octet-stream")},
        )

        assert response.status_code == 201
        body = response.json()["document"]
        assert body["status"] == "indexed"
        assert body["source_filename"] == "notes.md"

    def test_a_binary_file_named_md_is_refused(self, client):
        # The extension list is not the whole gate: markdown is text, and a
        # payload with NUL bytes in it is not markdown whatever it is called.
        response = client.post(
            "/v1/documents",
            files={"file": ("notes.md", io.BytesIO(b"# Heading\x00\x01\x02"), "text/markdown")},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "upload_rejected"


class TestTheServerOwnsTheList:
    def test_the_picker_learns_the_format_list_from_health(self, client):
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["supported_extensions"] == list(config.SUPPORTED_EXTENSIONS)
