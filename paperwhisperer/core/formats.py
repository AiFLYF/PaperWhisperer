"""The document formats this deployment accepts, and what each one is called.

One table, in ``core/`` rather than in any of the five layers that consult it,
for the same reason :mod:`paperwhisperer.core.identity` exists: they all have to
agree, and agreement by convention is how they stopped agreeing.

Before this module, the answer to "which files can I upload" was written out
five times -- in :mod:`~paperwhisperer.core.config`, in the upload contract, in
the gateway's routing table, in the legacy adapter's support set, and in the
frontend's ``accept`` attribute. The *extension* lists happened to match. The
*media type* each layer derived from a name did not:

* the upload contract accepted a declared ``text/markdown`` and normalised it to
  ``text/plain``;
* :func:`~paperwhisperer.ingestion.gateway.media_type_for_path` sent the same
  file down with the ``mimetypes`` answer for ``.md``, which is
  ``text/markdown``;
* no adapter declared ``text/markdown``, so the routing layer had no candidate.

So a ``.txt`` uploaded with ``Content-Type: text/markdown`` passed both
validation layers and then failed to parse, with no parser named in the attempt
trail. That is the shape of bug a single table removes, and it is why
:func:`~paperwhisperer.ingestion.orchestrator.parse_document` now normalises the
declared type through :data:`MEDIA_TYPE_ALIASES` before it routes.
"""

from __future__ import annotations

import os

#: Extension -> the media type this pipeline handles that extension as.
#:
#: Markdown is recorded as ``text/plain``, and that is a decision rather than an
#: oversight. Two layers had already said so before markdown was an accepted
#: upload -- the contract normalises a declared ``text/markdown`` to
#: ``text/plain``, and ``documents.validation`` lists ``text/markdown`` among
#: the types a ``.txt`` may claim -- and a third spelling in the routing layer
#: would buy nothing, because what reads a file is chosen by its *extension*
#: (see ``legacy_loader._EXTENSION_BY_FORMAT``), and the extension is preserved.
#: Calling it ``text/markdown`` here would instead make one declared type mean
#: two different things depending on the filename, which no alias table can
#: express.
#:
#: The order is the order a reader sees, so plain text comes first and the paper
#: formats after it.
EXTENSION_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/plain",
    ".markdown": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

SUPPORTED_EXTENSIONS = tuple(EXTENSION_MEDIA_TYPES)
ALLOWED_EXTENSIONS = frozenset(SUPPORTED_EXTENSIONS)
SUPPORTED_FILE_TYPES_TEXT = ", ".join(SUPPORTED_EXTENSIONS)

#: Declared content type -> what this pipeline calls it.
#:
#: Only declared types whose meaning does not depend on the filename belong
#: here. ``application/zip`` is the case that explains the rule: it is a
#: legitimate declaration for a ``.docx`` *and* for a ``.pptx``, so its
#: canonical form is not a property of the media type at all. It is deliberately
#: absent, which makes :func:`canonical_media_type` answer ``""`` for it and lets
#: the extension decide -- the same place the decision belongs for every type
#: nobody has taught this table about.
#:
#: ``application/x-pdf`` is the entry that shows what the table is for. Browsers
#: send it, the contract already accepted it, and an adapter comparing media
#: types against the IANA spelling did not match it -- which is how a PDF takes
#: the plain-text path and produces a document full of control characters.
MEDIA_TYPE_ALIASES = {
    "application/pdf": "application/pdf",
    "application/x-pdf": "application/pdf",
    "application/acrobat": "application/pdf",
    "application/vnd.pdf": "application/pdf",
    "text/plain": "text/plain",
    "text/x-python": "text/plain",
    "text/markdown": "text/plain",
    "text/x-markdown": "text/plain",
    EXTENSION_MEDIA_TYPES[".docx"]: EXTENSION_MEDIA_TYPES[".docx"],
    EXTENSION_MEDIA_TYPES[".pptx"]: EXTENSION_MEDIA_TYPES[".pptx"],
}


def extension_of(name: str) -> str:
    """The lower-cased extension of ``name``, including the dot, or ``""``.

    A bare ``".md"`` is a name with no extension as far as
    :func:`os.path.splitext` is concerned, so callers holding an extension
    rather than a filename should pass a name with something in front of it --
    ``f"sample{extension}"`` is the idiom already used in
    :mod:`~paperwhisperer.ingestion.adapters.legacy_loader`.
    """
    return os.path.splitext(str(name or ""))[1].lower()


#: Declarations that carry no claim about what the file is.
#:
#: ``application/octet-stream`` is what a client sends when it does not know
#: what it is holding: ``curl`` sends it for ``.md``, and so does any browser on
#: a host with no MIME registration for the extension, which on Windows
#: includes ``.md``. That is not a second claim contradicting the filename, it
#: is the *absence* of one, so it is treated exactly like a client that
#: declared nothing and the extension answers.
#:
#: This is not a hole. The payload is still checked against its magic bytes on
#: the way in (:func:`paperwhisperer.documents.validation.validate_saved_file_signature`),
#: and that check is the one that holds — it requires ``%PDF-`` for a ``.pdf``,
#: a ZIP header for a ``.docx``, and no NUL byte for a text file, none of which
#: a content-type header can fake. Refusing the header instead only stops
#: legitimate uploads: it is why a ``.md`` file could not be uploaded from a
#: Windows browser at all.
#:
#: The remote-download path has always read these as generic binary; this set
#: used to live there alone, which is why an upload and a download disagreed
#: about the same header.
NO_CLAIM_MEDIA_TYPES = frozenset({
    "application/octet-stream",
    "binary/octet-stream",
    "application/download",
    "application/x-download",
    "application/force-download",
})


def media_type_for_extension(extension: str) -> str:
    """Canonical media type for ``".md"``-style ``extension``, or ``""``."""
    return EXTENSION_MEDIA_TYPES.get(str(extension or "").lower(), "")


def canonical_media_type(declared: str) -> str:
    """What this pipeline calls the ``declared`` content type, or ``""``.

    ``""`` means "this table has no opinion", not "invalid". A caller that gets
    ``""`` should carry on with the extension, which is what the upload edge
    already validated. Parameters are dropped and the value is lower-cased here
    because a content type arrives as ``text/markdown; charset=utf-8`` on one
    client and bare on the next, and a table that only matches one of them
    silently stops matching rather than failing.
    """
    return MEDIA_TYPE_ALIASES.get(str(declared or "").split(";", 1)[0].strip().lower(), "")
