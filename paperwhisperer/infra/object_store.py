"""Blob storage in four layers, and a filesystem implementation of it.

T2.2 asks for ``original / parsed / canonical / derived`` 分层存储 and the plan's
tier table says the object store is 本地文件系统 for the personal tier and
MinIO/S3 in P4. Both halves of that sentence matter here, and they pull in
opposite directions: the *interface* has to be neutral enough that the S3
implementation of P4 is a drop-in, and the *taxonomy* has to be strict enough to
be worth having. The taxonomy is the part that is easy to under-build, because a
store that keeps four prefixes in one bucket and nothing else would satisfy a
reading of the sentence and none of its purpose.

**What the four layers are, and why there are four rather than two.** Each
layer answers a different question about a document, and they have different
replacement rules:

``original``
    The bytes the user uploaded. Nothing in this project ever rewrites them,
    and a re-parse cannot reach them.
``parsed``
    What a parser produced in *its own* vocabulary — the raw Docling JSON, the
    MinerU archive, the legacy loader's extraction. Re-parsing replaces it, but
    it is keyed by ``(sha256, parser_name, parser_version, options_hash)``, so
    the parse that won and the ones that lost are all still on disk. That is
    the evidence for T3.4's "which parser handled this, and what did the other
    one say".
``canonical``
    The :class:`~paperwhisperer.ingestion.schema.ParsedDocument` this project
    actually reasons about: format-aware locators, tables as two-dimensional
    structures, reading order. Exactly one of these is current per document,
    and a re-parse replaces it *in place under a new key* — the version history
    is the :class:`~paperwhisperer.infra.tables.TableStore` table, not a
    directory of near-identical files.
``derived``
    Everything computed from a canonical document: chunks, embeddings, the
    exported report, a rendered page image, a Mermaid mindmap. Disposable by
    construction — a derived object that cannot be rebuilt from a canonical one
    is a bug, and the cheapest way to keep that true is to keep nothing else
    in the layer.

**The taxonomy is enforced, not documented, in four places.** A reader who
skips this module will assume a directory prefix is a naming convention, so each
rule is stated with the failure it prevents:

1. *Sibling roots, not prefixes of one root.* Every layer is its own directory
   under the store root and a key is resolved only inside its layer's
   directory. There is no fallback, no search, no "if the canonical layer does
   not have it, try derived". A derived artifact is therefore unreachable
   through a canonical key even when the two keys are byte-identical, which is
   the answer a test can assert.
2. *A ref carries its layer.* :class:`ObjectRef` is not a string; it is a
   ``(layer, key)`` pair, and :meth:`FilesystemObjectStore.get_ref` raises
   :class:`LayerMismatch` when a caller names the layer it needs and the ref
   names a different one. A caller holding a derived ref cannot launder it into
   a canonical read by extracting the key.
3. *``original`` is write-once.* Writing a named key that already holds
   different bytes raises :class:`ImmutableObject`. A re-parse that reached the
   original layer with the wrong content has a bug, and failing the write is how
   the user finds out.
4. *Keys cannot escape their layer.* Absolute paths, ``..`` segments, empty
   segments, backslashes and control characters are refused by
   :func:`validate_object_key`. On a filesystem a key is a path, and a key
   containing ``..`` is a way to write outside the store — which on the personal
   tier means outside the data directory the owner is told to back up. The
   S3 implementation of P4 would treat the same key as an opaque name, so
   refusing here is what keeps the two tiers agreeing on which object a key
   names.

**Content addressing is what makes the strictness affordable.** A layered store
where finding "the canonical document for this upload" meant a directory walk
would push every caller into scanning, and callers that scan eventually cache
the answer and then serve it stale. So a put with no key is *addressed by the
SHA-256 of its own bytes*, fanned out over two hex characters to keep any one
directory small, and :meth:`FilesystemObjectStore.find_by_digest` turns a digest
back into a reference by arithmetic — no scan, no index, nothing to invalidate.
The consequence the tests pin down is that the same bytes are stored once: a
second identical put reports :attr:`ObjectRef.deduplicated` and does not touch
the file, so the modification time of the user's upload is evidence that nothing
rewrote it.

**The filesystem implementation is the one that exists, and it is honest about
what it is.** It writes with ``mkstemp`` + ``fsync`` + ``os.replace`` so a
reader in another process, or this one after a crash, sees either the old object
or the new one and never a half-written one — the same atomicity
:class:`~paperwhisperer.ingestion.orchestrator.ParseCache` uses for the same
reason. It refuses to guess a MIME type from content and takes one from the
caller, because the Hosted tier's S3 will store whatever it is told and a type
inferred from the first 512 bytes differs between the two. And
:meth:`~FilesystemObjectStore.capabilities` reports ``local=True`` together with
a degradation saying the objects are visible only to this machine: that is the
fact a multi-instance Hosted deployment has to fix by implementing S3, and
reporting it here is cheaper than a user discovering it.

What is deliberately absent: lifecycle policies, retention, and cross-layer
garbage collection. A layer whose objects are unreachable is a bug to be found
by :meth:`FilesystemObjectStore.usage`, not a thing to collect automatically —
"which derived objects are still referenced" is a question about the documents
table, which is T2.3's job to answer once the job pipeline writes both.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, BinaryIO, Protocol, runtime_checkable

from .stores.interfaces import TIER_PERSONAL

logger = logging.getLogger(__name__)

#: Objects are addressed by the SHA-256 of their own bytes when the caller does
#: not name them. Truncated to this many hex characters for the fan-out
#: directory: 256 buckets is enough that no personal-tier directory holds
#: everything, and few enough that the directory listing stays cheap.
DIGEST_FANOUT = 2

#: Upper bound on one path segment, in bytes. The real limits (255 on ext4,
#: 1024 on some network filesystems) differ, and this is the one both tiers can
#: honour; exceeding it is refused rather than discovered at write time on
#: whichever machine happened to have the shorter limit.
MAX_SEGMENT_BYTES = 200

#: The most bytes one object may be. Not a policy: it exists so that a runaway
#: generator turns into a clear refusal at the boundary instead of a full disk.
#: A 16 MB cap would be wrong for scanned papers, so this is deliberately loose
#: and the number that matters operationally is
#: :data:`~paperwhisperer.core.config.MAX_UPLOAD_BYTES`.
MAX_OBJECT_BYTES = 512 * 1024 * 1024


# --------------------------------------------------------------------------
# The four layers
# --------------------------------------------------------------------------


LAYER_ORIGINAL = "original"
LAYER_PARSED = "parsed"
LAYER_CANONICAL = "canonical"
LAYER_DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """What one layer is for, written down as data so it can be tested.

    ``derived_from`` is the layer(s) an object in this layer is allowed to be
    computed from. It is *declaration*, not runtime enforcement — the store
    cannot tell a report from a page image, and pretending a check exists that
    does not would be worse than not having one. What *is* enforced at runtime
    is the isolation between layers (see the module docstring), and this table
    is what the tests assert against so the declaration cannot rot.

    ``replaceable`` is the consequence that matters operationally. Everything
    but ``original`` can be recomputed or re-derived, so overwriting an object
    is routine; the original cannot be, which is why :meth:`put` refuses to
    overwrite one and :meth:`delete` needs ``force`` to remove one.
    """

    name: str
    title: str
    purpose: str
    derived_from: tuple[str, ...] = ()
    replaceable: bool = True
    suffix: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "purpose": self.purpose,
            "derived_from": list(self.derived_from),
            "replaceable": self.replaceable,
        }


LAYER_SPECS: dict[str, LayerSpec] = {
    LAYER_ORIGINAL: LayerSpec(
        name=LAYER_ORIGINAL,
        title="原始上传",
        purpose="用户上传的原始字节，任何解析与分析步骤都不得改写。",
        derived_from=(),
        replaceable=False,
    ),
    LAYER_PARSED: LayerSpec(
        name=LAYER_PARSED,
        title="解析器原始产物",
        purpose="解析器自己的输出（Docling JSON / MinerU 归档 / 旧 loader 抽取），"
        "按 (sha256, parser, version, options) 保留每一次尝试。",
        derived_from=(LAYER_ORIGINAL,),
        replaceable=True,
        suffix=".json",
    ),
    LAYER_CANONICAL: LayerSpec(
        name=LAYER_CANONICAL,
        title="规范化文档",
        purpose="本项目真正使用的 ParsedDocument：格式感知 locator、表格二维结构、阅读顺序。",
        derived_from=(LAYER_ORIGINAL, LAYER_PARSED),
        replaceable=True,
        suffix=".json",
    ),
    LAYER_DERIVED: LayerSpec(
        name=LAYER_DERIVED,
        title="派生产物",
        purpose="由 canonical 计算得出、可随时重建：chunk、embedding、导出报告、页面渲染图。",
        derived_from=(LAYER_CANONICAL,),
        replaceable=True,
        suffix=".json",
    ),
}

#: The four layers in the order a document flows through them. Order is the
#: point: it is the re-parse rule (a new canonical supersedes the old, the
#: original never moves) that the tests assert.
LAYERS: tuple[str, ...] = (LAYER_ORIGINAL, LAYER_PARSED, LAYER_CANONICAL, LAYER_DERIVED)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ObjectStoreError(Exception):
    """Base class, so a caller can catch every storage refusal at once."""


class UnknownLayer(ObjectStoreError):
    """A layer name outside :data:`LAYERS`."""


class UnsafeKey(ObjectStoreError):
    """A key that would escape its layer directory or is not a usable name."""


class ObjectNotFound(ObjectStoreError):
    """No such object in that layer."""


class LayerMismatch(ObjectStoreError):
    """A ref was used against a layer other than the one it names.

    Kept separate from :class:`ObjectNotFound` because the two mean opposite
    things to a caller: one is "it is not there", the other is "you are holding
    the wrong kind of address". Reporting the second as the first would turn a
    layering bug into a mysterious 404.
    """


class ImmutableObject(ObjectStoreError):
    """A write that would replace bytes in the ``original`` layer."""


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


def _now_iso() -> str:
    """UTC, second resolution — the convention the store and job rows use."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """An address *plus* what is known about the object there.

    Deliberately not a string. A ``str`` key is a bare path, a bare path has no
    layer, and a bare path with no layer is how a derived artifact ends up being
    served as a canonical one. Keeping the layer inside the value means the
    check in :meth:`ObjectStore.get_ref` cannot be forgotten at a call site.

    ``sha256`` and ``size`` are filled in by a write and by :meth:`head`, and
    left empty by a :meth:`ObjectStore.list_objects` walk — hashing every object
    in a directory listing would cost more than the query the listing feeds. An
    empty ``sha256`` means "not computed", never "unknown content".
    """

    layer: str
    key: str
    sha256: str = ""
    size: int = 0
    content_type: str = ""
    created_at: str = ""
    #: Only meaningful on the ref a ``put`` returned: ``True`` means the object
    #: was already there with these exact bytes, so nothing was written and the
    #: file's modification time is unchanged. That is the observable a re-parse
    #: uses to prove it did not touch the user's upload.
    deduplicated: bool = False

    def __post_init__(self) -> None:
        if self.layer not in LAYER_SPECS:
            raise UnknownLayer(
                f"未知的对象层：{self.layer!r}（可用值：{', '.join(LAYERS)}）"
            )
        object.__setattr__(self, "key", validate_object_key(self.key))

    @property
    def identity(self) -> tuple[str, str]:
        """The pair that makes a ref unique. Two layers, same key, are different."""
        return (self.layer, self.key)

    def matches(self, layer: str, key: str) -> bool:
        return self.layer == str(layer) and self.key == str(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "key": self.key,
            "sha256": self.sha256,
            "size": self.size,
            "content_type": self.content_type,
            "created_at": self.created_at,
            "deduplicated": self.deduplicated,
        }

    def describe(self) -> str:
        """One Chinese line naming the layer and the key, for a log or an error."""
        return f"{LAYER_SPECS[self.layer].title}（{self.layer}）:{self.key}"


@dataclass(frozen=True, slots=True)
class LayerUsage:
    """How much a layer holds. Object count and bytes, nothing else.

    The first thing T2.6's backup note needs and the thing a "why is my data
    directory 40 GB" question needs answered with a number rather than a
    ``du`` the owner has to run themselves.
    """

    layer: str
    objects: int = 0
    bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "title": LAYER_SPECS[self.layer].title if self.layer in LAYER_SPECS else self.layer,
            "objects": self.objects,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class ObjectStoreCapabilities:
    """What this object store can actually do, as opposed to what it promises.

    :class:`~paperwhisperer.infra.stores.interfaces.StoreCapabilities` is not
    reused even though it is the same idea, because its fields are about
    vectors and full-text. Returning it from an object store would mean
    reporting a lexical engine for a class that has no opinion about one, and
    the store tier's own factory merges three of those reports together — an
    object store's report would have to be a lie to fit.

    ``local``
        The one field a multi-instance deployment has to read. ``True`` means
        these objects are reachable only from this machine, which is correct for
        the personal tier and wrong for two API instances sharing a database, and
        the Hosted tier's fix is the S3 implementation of P4.
    """

    tier: str = TIER_PERSONAL
    backend: str = "filesystem"
    content_addressed: bool = True
    local: bool = True
    fsync: bool = True
    degradations: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "backend": self.backend,
            "content_addressed": self.content_addressed,
            "local": self.local,
            "fsync": self.fsync,
            "degradations": list(self.degradations),
        }


# --------------------------------------------------------------------------
# Key safety
# --------------------------------------------------------------------------


_CONTROL_CHARACTERS = {chr(code) for code in range(32)} | {chr(127)}


def validate_object_key(key: str) -> str:
    """Return ``key`` if it is a safe, layer-relative object name.

    Structural rules only — the key is validated for shape, not for an
    alphabet, because a legitimate key can be a CJK filename and rejecting it
    would push callers into escaping, which is worse than a permissive
    character class. What is refused is everything that changes *which* object a
    key names depending on which machine resolves it:

    * an empty key, or a key with an empty segment (``a//b``) — two spellings of
      one name is one too many;
    * a leading ``/``, a leading ``./`` or a ``..`` segment — the first two and
      the third all escape the layer directory, and a store that resolves a key
      outside its own root is writing outside the data directory the owner was
      told to back up. Nothing is normalised away here: a key that *looks* like
      an escape is refused rather than cleaned, because a cleaned key is a key
      whose stored name is not the name the caller asked for;
    * a backslash — POSIX and S3 read it as a literal character, so a key
      containing one would name a different object on each tier;
    * a control character, including NUL, which truncates the name in the C APIs
      both tiers eventually call;
    * a segment longer than :data:`MAX_SEGMENT_BYTES` UTF-8 bytes, which is
      refused here rather than at ``open(2)`` time on the machine that happens to
      have the shorter limit.
    """
    text = str(key or "").strip()
    if not text:
        raise UnsafeKey("对象键不能为空。")
    if any(character in _CONTROL_CHARACTERS for character in text):
        raise UnsafeKey("对象键不能包含控制字符。")
    if "\\" in text:
        raise UnsafeKey(
            f"对象键不能包含反斜杠：{key!r}"
            "（键是 POSIX/S3 风格的相对路径，反斜杠在两档会指向不同的对象）"
        )
    if text.startswith("/"):
        raise UnsafeKey(f"对象键必须是相对路径：{key!r}")
    while text.endswith("/"):
        text = text[:-1]
    segments = text.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise UnsafeKey(
                f"非法的对象键：{key!r}"
                "（键必须是层目录下的相对路径，不得包含空段、'.' 或 '..'）"
            )
        if len(segment.encode("utf-8")) > MAX_SEGMENT_BYTES:
            raise UnsafeKey(
                f"对象键片段过长：{segment[:40]!r}…"
                f"（单段上限 {MAX_SEGMENT_BYTES} 字节）"
            )
    return text


def digest_key(sha256: str, suffix: str = "") -> str:
    """The content-addressed key for a digest: ``ab/abcd…<suffix>``.

    The two-character fan-out is not cosmetic. Without it a store with a few
    thousand objects has one directory with a few thousand entries, and the
    pathological case is a bulk import: a directory that large stops being cheap
    to list on some filesystems, and the listing is the first thing
    :meth:`FilesystemObjectStore.usage` does.
    """
    digest = str(sha256 or "").lower()
    if not digest:
        raise ValueError("sha256 不能为空。")
    clean = "".join(character for character in digest if character in "0123456789abcdef")
    if len(clean) != 64:
        raise ValueError(f"非法的 sha256：{sha256!r}（期望 64 位十六进制）")
    return f"{clean[:DIGEST_FANOUT]}/{clean}{suffix}"


def digest_from_key(key: str) -> str:
    """Recover the digest from a content-addressed key, or ``""`` if it is not one.

    The inverse of :func:`digest_key` for the name shape the store itself
    writes, which is what lets a directory listing report content digests for
    free. A caller-supplied key is a different name shape and correctly reports
    nothing rather than a wrong digest.
    """
    name = str(key or "").rsplit("/", 1)[-1]
    stem = name.split(".", 1)[0]
    if len(stem) == 64 and all(character in "0123456789abcdef" for character in stem):
        return stem
    return ""


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@runtime_checkable
class ObjectStore(Protocol):
    """Blobs in a named layer, addressed by a key inside it.

    The interface is neutral on purpose: nothing here says filesystem, and
    nothing here returns a local path. An S3 implementation of P4 is this
    protocol over ``put_object``/``get_object``, and the P4 acceptance criterion
    ("两档共用同一套 storage 抽象接口") is only true if a caller cannot tell
    which one it holds. :meth:`capabilities` is where a tier's genuine
    difference is reported, as data.
    """

    def capabilities(self) -> ObjectStoreCapabilities:
        """Describe what this instance is actually doing. No I/O."""
        ...

    def initialize(self) -> None:
        """Create the layer directories. Idempotent."""
        ...

    def put(
        self,
        layer: str,
        data: bytes,
        *,
        key: str | None = None,
        content_type: str = "",
        suffix: str = "",
    ) -> ObjectRef:
        """Write ``data`` into ``layer``, atomically, and return its ref.

        With no ``key`` the object is addressed by the SHA-256 of ``data`` and a
        second identical write is a no-op that reports
        :attr:`ObjectRef.deduplicated`. With a ``key`` the object is at a name
        the caller chose, and a write over different bytes replaces it —
        except in the ``original`` layer, which raises
        :class:`ImmutableObject` instead.
        """
        ...

    def put_file(
        self,
        layer: str,
        path: str | os.PathLike[str],
        *,
        key: str | None = None,
        content_type: str = "",
        suffix: str = "",
    ) -> ObjectRef:
        """:meth:`put`, reading a file without loading it into memory.

        The path an upload arrives on is this method's normal use, and hashing a
        16 MB PDF by reading it once is materially cheaper than reading it twice
        — which is what ``put(open(path).read())`` costs. The file is staged and
        committed under one atomic replace, so a failed write leaves no object.
        """
        ...

    def get(self, layer: str, key: str) -> bytes:
        """Return an object's bytes, or raise :class:`ObjectNotFound`."""
        ...

    def get_ref(self, ref: ObjectRef, *, expect_layer: str = "") -> bytes:
        """:meth:`get`, addressed by a ref — and the layer is checked when named."""
        ...

    def open_stream(self, layer: str, key: str) -> BinaryIO:
        """Open an object for streaming reads. The caller closes it.

        Exists because the personal tier's exports and the Hosted tier's would
        otherwise differ by a whole copy of the object: a caller with a
        300 MB raw artifact needs a file handle, not ``bytes``.
        """
        ...

    def head(self, layer: str, key: str) -> ObjectRef | None:
        """Metadata without reading the object, or ``None`` when it is absent."""
        ...

    def exists(self, layer: str, key: str) -> bool:
        """Whether an object is there. Cheaper than :meth:`head`, which hashes."""
        ...

    def delete(self, layer: str, key: str, *, force: bool = False) -> bool:
        """Remove an object, returning whether it was there.

        ``force`` is required in the ``original`` layer, because that is the one
        object in the store that cannot be produced again: every other layer is
        a computation over it. A caller with a genuine delete request — a user
        asking for their upload to be removed — passes ``force=True`` and the
        store complies; a caller that got there by accident does not.
        """
        ...

    def list_objects(self, layer: str, prefix: str = "") -> list[ObjectRef]:
        """Every object in ``layer`` whose key starts with ``prefix``, key order.

        Key order rather than modification-time order, because it is total and
        therefore identical on the two tiers. A listing does not hash: a ref
        from a listing has an empty :attr:`ObjectRef.sha256` unless the key is
        content-addressed and the digest is in the name.
        """
        ...

    def find_by_digest(self, layer: str, sha256: str) -> ObjectRef | None:
        """The object with this content digest in ``layer``, or ``None``.

        Arithmetic, not a scan: the key is derived from the digest, so the
        question "do we already hold these bytes" is answered by one stat. This
        is what makes the strict layering affordable — a re-parse can ask for the
        canonical object it is about to write before writing it, and get an
        answer without walking anything.
        """
        ...

    def usage(self) -> dict[str, LayerUsage]:
        """Object count and bytes per layer. A directory walk, so not free."""
        ...

    def close(self) -> None:
        """Release anything held. Idempotent, and never raises."""
        ...


# --------------------------------------------------------------------------
# The filesystem implementation
# --------------------------------------------------------------------------


class FilesystemObjectStore:
    """The personal tier's object store: four directories under one root.

    One root, because T2.6 asks the owner to be able to back up a *directory*
    and because the alternative — a root per layer — makes "where is my upload"
    a question with four answers. The layers are still separate directories
    inside it, which is the isolation that matters.

    Writes are staged in the destination directory and committed with
    ``os.replace``, so a concurrent reader (another API instance, or the same one
    after a crash) sees the old object or the new one. Staging in the destination
    directory rather than the system temp directory is deliberate: ``os.replace``
    is only atomic within one filesystem, and a temp file elsewhere would make
    the commit a non-atomic copy on any deployment whose ``/tmp`` is a different
    mount — which is most containers.
    """

    def __init__(self, root: str | os.PathLike[str], *, fsync: bool = True) -> None:
        self.root = os.path.abspath(str(root))
        self.fsync = bool(fsync)
        self._closed = False
        self._capabilities = ObjectStoreCapabilities(
            tier=TIER_PERSONAL,
            backend="filesystem",
            content_addressed=True,
            local=True,
            fsync=self.fsync,
            degradations=(
                "对象存储使用本机文件系统，仅本进程所在机器可见；"
                "多实例部署需在 P4 实现 S3/MinIO 后端。",
            ),
        )

    # -- plumbing ----------------------------------------------------------

    def capabilities(self) -> ObjectStoreCapabilities:
        return self._capabilities

    def close(self) -> None:
        """Nothing is held open between calls, so this only marks the instance.

        It exists because the contract requires it and because a caller that
        closed the object store during shutdown would otherwise have to special
        case it. A close that raised would replace the error that caused the
        shutdown with a second, less useful one, so this never does.
        """
        self._closed = True

    def initialize(self) -> None:
        for layer in LAYERS:
            os.makedirs(self.layer_dir(layer), exist_ok=True)

    def layer_dir(self, layer: str) -> str:
        """The absolute directory backing ``layer``. Filesystem-only, on purpose.

        Exposed because T2.6's backup note and any operator-facing diagnostic
        need to *say where the bytes are*, and because the tests assert the
        sibling-roots property on real paths. It is not on the protocol: a
        caller that resolved a path from this method could not work on the S3
        backend, and the interface's job is to stop that.
        """
        spec = LAYER_SPECS.get(str(layer))
        if spec is None:
            raise UnknownLayer(f"未知的对象层：{layer!r}（可用值：{', '.join(LAYERS)}）")
        return os.path.join(self.root, spec.name)

    def local_path(self, layer: str, key: str) -> str:
        """The absolute file path for ``key``. Filesystem-only; see :meth:`layer_dir`."""
        return os.path.join(self.layer_dir(layer), validate_object_key(key))

    def _resolve(self, layer: str, key: str) -> str:
        base = self.layer_dir(layer)
        path = os.path.normpath(os.path.join(base, validate_object_key(key)))
        # Belt and braces: the key is already validated, so this cannot fail for
        # a key that passed. It is here because the alternative — a traversal
        # reaching outside the data directory — is silent, and a store that
        # silently writes outside its own root is the worst bug this module
        # could have.
        if path != base and not path.startswith(base + os.sep):
            raise UnsafeKey(f"非法的对象键：{key!r}（解析后越出了层目录）")
        return path

    # -- writes ------------------------------------------------------------

    def _existing_digest(self, target: str) -> str:
        """The digest of the file at ``target``, or ``""`` when there is none.

        Reads the file to hash it rather than trusting a recorded digest,
        because the recorded value is a cache and this check is the thing that
        decides whether a user's bytes are replaced. A cache that is wrong here
        loses data; a cache that is merely slow here costs one sequential read
        of an object that is being written anyway.
        """
        if not os.path.exists(target):
            return ""
        try:
            digest = hashlib.sha256()
            with open(target, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except OSError as exc:
            logger.warning("无法校验已存在对象 %s：%s", target, exc)
            return ""

    def _stage(self, layer: str, chunks: Iterator[bytes]) -> tuple[str, str, int]:
        """Write a stream to a temp file, returning ``(path, digest, size)``.

        The temp file is created in the layer directory, and any failure
        unlinks it. That is the "a failed write leaves no partial object"
        property, and it holds for a failure at any point: a disk filling up on
        the last block, a generator raising half-way through a 300 MB artifact,
        or a ``KeyboardInterrupt``. There is no window in which a half-written
        object is visible under its final name.
        """
        directory = self.layer_dir(layer)
        os.makedirs(directory, exist_ok=True)
        descriptor, stage = tempfile.mkstemp(dir=directory, prefix=".staging-")
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for block in chunks:
                    if not block:
                        continue
                    digest.update(block)
                    size += len(block)
                    if size > MAX_OBJECT_BYTES:
                        raise ObjectStoreError(
                            f"单个对象超过上限 {MAX_OBJECT_BYTES} 字节，已放弃写入。"
                        )
                    handle.write(block)
                handle.flush()
                if self.fsync:
                    # The data must be durable before the rename publishes the
                    # name. Without this, a crash right after os.replace can
                    # leave a correctly named object whose contents are not
                    # there — the one failure a backup would faithfully preserve.
                    os.fsync(handle.fileno())
        except BaseException:
            _silent_unlink(stage)
            raise
        return stage, digest.hexdigest(), size

    def _write(
        self,
        layer: str,
        chunks: Iterator[bytes],
        *,
        key: str | None,
        content_type: str,
        suffix: str,
    ) -> ObjectRef:
        stage, digest, size = self._stage(layer, chunks)
        # `key is None` rather than `if not key`: an empty key is a caller's
        # mistake and is refused by validate_object_key, whereas treating it as
        # "no key" would silently store the object at a content address the caller
        # never asked for and never learn the name of.
        named = digest_key(digest, suffix) if key is None else validate_object_key(key)
        target = self._resolve(layer, named)
        if os.path.exists(target) and self._existing_digest(target) == digest:
            _silent_unlink(stage)
            return ObjectRef(
                layer=layer,
                key=named,
                sha256=digest,
                size=size,
                content_type=content_type,
                created_at=_iso_from_mtime(os.path.getmtime(target)),
                deduplicated=True,
            )
        if os.path.exists(target) and not LAYER_SPECS[layer].replaceable:
            _silent_unlink(stage)
            raise ImmutableObject(
                f"原始上传不可覆盖：{LAYER_SPECS[layer].title} 已存在同名对象 "
                f"{named!r} 且内容不同。重新解析只能写入 parsed/canonical/derived 层。"
            )
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            os.replace(stage, target)
        except BaseException:
            _silent_unlink(stage)
            raise
        self._fsync_dir(os.path.dirname(target))
        return ObjectRef(
            layer=layer,
            key=named,
            sha256=digest,
            size=size,
            content_type=content_type or _guess_content_type(named),
            created_at=_now_iso(),
            deduplicated=False,
        )

    def _fsync_dir(self, directory: str) -> None:
        """Best-effort directory fsync so a rename survives a crash.

        Best-effort on purpose: a filesystem that refuses ``O_RDONLY`` fsync on a
        directory (some network mounts, Windows without the right handle) must
        not turn a successful write into a failed one. The rename is still
        ordered correctly in the page cache; only the crash window is wider.
        """
        if not self.fsync:
            return
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def put(
        self,
        layer: str,
        data: bytes,
        *,
        key: str | None = None,
        content_type: str = "",
        suffix: str = "",
    ) -> ObjectRef:
        payload = bytes(data)
        return self._write(
            layer,
            iter((payload,)),
            key=key,
            content_type=content_type,
            suffix=suffix or LAYER_SPECS.get(layer, LayerSpec("", "", "")).suffix,
        )

    def put_file(
        self,
        layer: str,
        path: str | os.PathLike[str],
        *,
        key: str | None = None,
        content_type: str = "",
        suffix: str = "",
    ) -> ObjectRef:
        source = str(path)
        resolved_suffix = suffix
        if not resolved_suffix and key is None:
            resolved_suffix = os.path.splitext(source)[1]

        def _chunks() -> Iterator[bytes]:
            with open(source, "rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        return
                    yield block

        return self._write(
            layer,
            _chunks(),
            key=key,
            content_type=content_type,
            suffix=resolved_suffix,
        )

    # -- reads -------------------------------------------------------------

    def get(self, layer: str, key: str) -> bytes:
        path = self._resolve(layer, key)
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except FileNotFoundError as exc:
            raise ObjectNotFound(
                f"{LAYER_SPECS[layer].title} 中不存在对象：{key!r}"
            ) from exc
        except IsADirectoryError as exc:
            raise UnsafeKey(f"对象键指向一个目录而非对象：{key!r}") from exc

    def get_ref(self, ref: ObjectRef, *, expect_layer: str = "") -> bytes:
        """:meth:`get`, addressed by a ref, with the layer checked when one is named.

        The check is the whole reason :class:`ObjectRef` is not a string. A
        derived ref stays readable here, because a ref names its own layer and
        the store should be able to serve it; what must be impossible is using a
        derived ref to satisfy a *canonical* request. So a caller that has the
        layer it needs passes it, and a ref from any other layer raises
        :class:`LayerMismatch`.

        ``expect_layer`` is optional because most callers legitimately do not
        have one — they hold a ref and want its bytes. A caller that does have a
        layer in hand is a caller about to trust that a piece of evidence is the
        canonical one, which is the check worth making.
        """
        if not isinstance(ref, ObjectRef):
            raise LayerMismatch(f"需要 ObjectRef，收到 {type(ref).__name__}。")
        if expect_layer and ref.layer != str(expect_layer):
            wanted = LAYER_SPECS.get(str(expect_layer))
            raise LayerMismatch(
                f"对象层不匹配：引用属于 {LAYER_SPECS[ref.layer].title}（{ref.layer}），"
                f"调用方要求 {wanted.title if wanted else expect_layer}（{expect_layer}）。"
                "分层存储不允许用一个层的对象冒充另一个层。"
            )
        return self.get(ref.layer, ref.key)

    def open_stream(self, layer: str, key: str) -> BinaryIO:
        path = self._resolve(layer, key)
        try:
            return open(path, "rb")
        except FileNotFoundError as exc:
            raise ObjectNotFound(
                f"{LAYER_SPECS[layer].title} 中不存在对象：{key!r}"
            ) from exc

    def head(self, layer: str, key: str) -> ObjectRef | None:
        path = self._resolve(layer, key)
        if not os.path.isfile(path):
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        named = validate_object_key(key)
        return ObjectRef(
            layer=layer,
            key=named,
            sha256=self._existing_digest(path),
            size=int(stat.st_size),
            content_type=_guess_content_type(named),
            created_at=_iso_from_mtime(stat.st_mtime),
        )

    def exists(self, layer: str, key: str) -> bool:
        return os.path.isfile(self._resolve(layer, key))

    def find_by_digest(self, layer: str, sha256: str) -> ObjectRef | None:
        if layer not in LAYER_SPECS:
            raise UnknownLayer(f"未知的对象层：{layer!r}（可用值：{', '.join(LAYERS)}）")
        base = self.layer_dir(layer)
        for suffix in ("", LAYER_SPECS[layer].suffix):
            candidate = digest_key(sha256, suffix)
            path = os.path.join(base, candidate)
            if os.path.isfile(path):
                return ObjectRef(
                    layer=layer,
                    key=candidate,
                    sha256=str(sha256).lower(),
                    size=os.path.getsize(path),
                    content_type=_guess_content_type(candidate),
                    created_at=_iso_from_mtime(os.path.getmtime(path)),
                )
        return None

    def list_objects(self, layer: str, prefix: str = "") -> list[ObjectRef]:
        base = self.layer_dir(layer)
        found: list[ObjectRef] = []
        for directory, _subdirectories, filenames in os.walk(base):
            for name in sorted(filenames):
                if name.startswith(".staging-"):
                    # A staging file is an object in flight, not an object.
                    continue
                full = os.path.join(directory, name)
                key = validate_object_key(os.path.relpath(full, base))
                if prefix and not key.startswith(prefix):
                    continue
                found.append(
                    ObjectRef(
                        layer=layer,
                        key=key,
                        sha256=digest_from_key(key),
                        size=os.path.getsize(full),
                        content_type=_guess_content_type(key),
                        created_at=_iso_from_mtime(os.path.getmtime(full)),
                    )
                )
        return sorted(found, key=lambda ref: ref.key)

    def delete(self, layer: str, key: str, *, force: bool = False) -> bool:
        spec = LAYER_SPECS.get(layer)
        if spec is None:
            raise UnknownLayer(f"未知的对象层：{layer!r}（可用值：{', '.join(LAYERS)}）")
        if not spec.replaceable and not force:
            raise ImmutableObject(
                f"原始上传需要显式 force=True 才能删除：{key!r}。"
                "它无法重新生成，确认要删除用户的上传文件时请显式传入。"
            )
        path = self._resolve(layer, key)
        if not os.path.isfile(path):
            return False
        try:
            os.unlink(path)
        except FileNotFoundError:
            return False
        return True

    def usage(self) -> dict[str, LayerUsage]:
        totals: dict[str, LayerUsage] = {}
        for layer in LAYERS:
            base = self.layer_dir(layer)
            count = 0
            size = 0
            for directory, _subdirectories, filenames in os.walk(base):
                for name in filenames:
                    if name.startswith(".staging-"):
                        continue
                    try:
                        size += os.path.getsize(os.path.join(directory, name))
                    except OSError:
                        continue
                    count += 1
            totals[layer] = LayerUsage(layer=layer, objects=count, bytes=size)
        return totals

    def describe(self) -> dict[str, Any]:
        """A machine-readable summary, for a health endpoint or a log line."""
        return {"root": self.root, **self._capabilities.describe()}

    def backup_paths(self) -> dict[str, str]:
        """Every directory T2.6 has to copy, keyed by layer.

        The data directory of the personal tier is the SQLite file *and* this
        root, and a backup that takes only the former silently loses every
        upload. Returning the per-layer paths rather than one root is what lets
        the backup note say exactly what to copy.
        """
        return {layer: self.layer_dir(layer) for layer in LAYERS}


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


#: Every key this module reads, with the default used when it is absent. Read
#: with ``getattr`` and a default, like
#: :data:`~paperwhisperer.infra.stores.factory.REQUIRED_CONFIG_KEYS`, because
#: this module must keep working on a checkout where nobody has touched
#: ``core/config.py``. ``core/config.py`` copies these names and defaults when
#: it grows them; the table here is the contract between the two modules.
#:
#: ``OBJECT_STORE_ROOT``
#:     Where the four layer directories live. Default ``<CONTEXT_FOLDER>/objects``,
#:     resolved at call time for the same reason
#:     :func:`~paperwhisperer.ingestion.orchestrator.build_default_parse_cache`
#:     resolves its own default from ``CONTEXT_FOLDER``: a test that redirects the
#:     runtime folders redirects the object store with them, and an operator
#:     with one data directory keeps having one.
#: ``OBJECT_STORE_BACKEND``
#:     ``filesystem`` (default, both tiers today) or ``s3``. ``s3`` is refused
#:     with a message naming P4 rather than silently falling back to a local
#:     directory — the same rule the store factory applies to a missing DSN,
#:     because the failure it prevents is a Hosted deployment that looks healthy
#:     while writing every user's bytes to a container's disk.
#: ``OBJECT_STORE_FSYNC``
#:     Whether a committed object is fsynced before its name is published.
#:     Default ``true``. Turning it off trades durability of the *name* for
#:     throughput, which is a reasonable choice for a bulk re-index and a bad one
#:     for the layer holding a user's upload.
OBJECT_STORE_CONFIG_KEYS: dict[str, Any] = {
    "OBJECT_STORE_BACKEND": "filesystem",
    "OBJECT_STORE_FSYNC": True,
}

BACKEND_FILESYSTEM = "filesystem"
OBJECT_STORE_BACKENDS = (BACKEND_FILESYSTEM,)

_BOOL_TRUE = {"1", "true", "yes", "on"}


def default_object_store_root(settings: Any = None) -> str:
    """The root this configuration asks for, without building anything."""
    configured = str(getattr(settings, "OBJECT_STORE_ROOT", "") or "").strip()
    if configured:
        return configured
    context_folder = str(getattr(settings, "CONTEXT_FOLDER", "") or "context").strip() or "context"
    return os.path.join(context_folder, "objects")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _BOOL_TRUE


def build_object_store(settings: Any = None) -> ObjectStore:
    """Build the object store for the configured backend.

    The one place a backend is chosen, mirroring
    :func:`~paperwhisperer.infra.stores.factory.build_stores`: every other caller
    receives an :class:`ObjectStore` and never learns which one it got, which is
    the property that makes "S3 实现留 P4" a scheduling note rather than a
    rewrite.
    """
    backend = str(
        getattr(settings, "OBJECT_STORE_BACKEND", None)
        or OBJECT_STORE_CONFIG_KEYS["OBJECT_STORE_BACKEND"]
    ).strip().lower()
    if backend in ("", "file", "local", BACKEND_FILESYSTEM):
        backend = BACKEND_FILESYSTEM
    if backend not in OBJECT_STORE_BACKENDS:
        raise ValueError(
            f"未知的 OBJECT_STORE_BACKEND：{backend!r}"
            f"（当前可用值：{', '.join(OBJECT_STORE_BACKENDS)}）。"
            "S3/MinIO 实现属于 P4；Hosted 档在 P4 之前同样使用本地文件系统，"
            "并在 capabilities().local 中报告该限制。"
        )
    if settings is None:
        from paperwhisperer.core import config as settings

    root = default_object_store_root(settings)
    fsync = _as_bool(
        getattr(settings, "OBJECT_STORE_FSYNC", None)
        if getattr(settings, "OBJECT_STORE_FSYNC", None) is not None
        else OBJECT_STORE_CONFIG_KEYS["OBJECT_STORE_FSYNC"]
    )
    store = FilesystemObjectStore(root, fsync=fsync)
    store.initialize()
    logger.info("对象存储已就绪：%s", store.describe())
    return store


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _guess_content_type(key: str) -> str:
    """A MIME type from the name, or ``""``.

    From the *name*, never from the bytes. Sniffing content would give the two
    tiers different answers for the same object, and the Hosted tier's S3 will
    store whatever content type it is handed at upload time; a caller that cares
    passes one to ``put`` and the ref records it.
    """
    guessed, _encoding = mimetypes.guess_type(str(key))
    return guessed or ""


def _iso_from_mtime(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")


def copy_object(
    source: ObjectStore,
    destination: ObjectStore,
    ref: ObjectRef,
    *,
    target_layer: str | None = None,
) -> ObjectRef:
    """Copy one object between two stores, without loading it into memory.

    Exists for the one operation P4 will need and no caller should hand-roll: a
    parse result that came from a local disk has to land in S3 without anyone
    reading a 300 MB artifact into memory to write it back out.

    The public protocol's streaming write is :meth:`ObjectStore.put_file`, so a
    caller with a stream stages to a temporary file first. That is the protocol's
    shape, not an accident of the filesystem: an S3 implementation can stream a
    multipart upload from a file exactly as well, whereas an interface built
    around ``bytes`` would force every large write through memory on both tiers.
    """
    descriptor, stage = tempfile.mkstemp(prefix="object-copy-", suffix=".tmp")
    os.close(descriptor)
    try:
        with source.open_stream(ref.layer, ref.key) as reader, open(stage, "wb") as writer:
            while True:
                block = reader.read(1024 * 1024)
                if not block:
                    break
                writer.write(block)
        return destination.put_file(
            target_layer or ref.layer,
            stage,
            key=ref.key,
            content_type=ref.content_type,
        )
    finally:
        _silent_unlink(stage)


__all__ = [
    "BACKEND_FILESYSTEM",
    "DIGEST_FANOUT",
    "LAYERS",
    "LAYER_CANONICAL",
    "LAYER_DERIVED",
    "LAYER_ORIGINAL",
    "LAYER_PARSED",
    "LAYER_SPECS",
    "MAX_OBJECT_BYTES",
    "MAX_SEGMENT_BYTES",
    "OBJECT_STORE_BACKENDS",
    "OBJECT_STORE_CONFIG_KEYS",
    "FilesystemObjectStore",
    "ImmutableObject",
    "LayerMismatch",
    "LayerSpec",
    "LayerUsage",
    "ObjectNotFound",
    "ObjectRef",
    "ObjectStore",
    "ObjectStoreCapabilities",
    "ObjectStoreError",
    "UnsafeKey",
    "UnknownLayer",
    "build_object_store",
    "copy_object",
    "default_object_store_root",
    "digest_from_key",
    "digest_key",
    "validate_object_key",
]
