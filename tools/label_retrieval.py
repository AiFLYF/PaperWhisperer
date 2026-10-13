#!/usr/bin/env python3
"""Labelling harness for the two blocked release gates.

``docs/RELEASE_GATES.md`` records ``citation_precision`` and
``retrieval_recall_at_k`` as ``BLOCKED`` because no labelled set exists. The
harness, the schema and the scorer exist; the truth does not. This module is the
missing instrument: it turns "a person marks which passages answer a question"
from an afternoon of reading into a resumable, append-only log, and it keeps
the line between *what the tool proposed* and *what a person confirmed*
mechanical rather than a matter of good intentions.

Subcommands
-----------

``build``
    Parse the corpus documents named by a seed file, split them into passage
    windows, and pre-rank the candidates for every seeded query. Cached in
    ``eval/labels/worklist.json`` so labelling never re-parses a PDF.

``dump``
    Print the candidate passages for a set of queries. This is the reading
    surface: an annotator (a person, or a draft an assistant hands to a person)
    reads it and writes decisions with ``label`` / ``claims``.

``label``
    Record relevance decisions, interactively from stdin or in bulk from an
    answer sheet. Append-only: a changed mind is a new row, never an edit, so
    the log records how often a decision was re-examined and how often it moved.

``claims``
    Record ``supported`` judgements over claim candidates. This axis cannot be
    self-scored, so its candidates come from the deterministic extractive path
    and the judgement is always the annotator's.

``agreement``
    Compare two annotators over the queries they both labelled: raw agreement,
    Cohen's kappa over per-window relevance decisions, and every disagreement.

``status``
    What is decided, what is skipped, what was revised, per annotator.

``export``
    Write ``eval/labels/retrieval_labels.yaml`` in exactly the schema
    ``eval.gates.load_labels`` reads.

Provenance is the whole point
-----------------------------

The tool ranks, groups and pre-fills. It must never *label*. Three record
sources exist and they mean different things:

``human``
    The annotator typed the relevant set themselves. Strongest.
``human_accepted_prefill``
    The annotator pressed Enter on the tool's ranking. Still a person's
    confirmation -- but a weaker one, recorded separately on purpose.
``assistant_draft``
    Produced by an assistant, not confirmed by a person. ``human_confirmed`` is
    ``False``, ``export`` refuses to put it on the gate's input path without an
    explicit ``--accept-assistant-labels``, and the refusal is the mechanism
    that keeps a self-scored gate from ever being green by accident.

Everything under ``eval/labels/`` is deterministic and local: the legacy parser,
the production lexical form expansion, and ``fastembed`` for the dense leg. No
LLM endpoint is contacted -- the endpoint at :8084 429s under contention and a
labelling tool that depends on it is a labelling tool nobody can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run as a script, ``sys.path[0]`` is ``tools/``. Put the repo on the path
# explicitly so ``python tools/label_retrieval.py`` works from anywhere --
# a labelling tool you have to invoke as ``python -m tools.label_retrieval``
# from the repo root is a tool that gets invoked wrong once.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LABELS_DIR = REPO_ROOT / "eval" / "labels"
WORKLIST_PATH = LABELS_DIR / "worklist.json"
ANNOTATIONS_DIR = LABELS_DIR / "annotations"
EXPORT_PATH = LABELS_DIR / "retrieval_labels.yaml"
PENDING_PATH = LABELS_DIR / "pending_labels.yaml"
AGREEMENT_PATH = LABELS_DIR / "agreement.json"

#: Must match ``eval.gates.LABELS_SCHEMA_PREFIX``; ``tests/test_label_retrieval.py``
#: imports the real loader, so a drift here fails a test rather than CI.
LABELS_SCHEMA = "paperwhisperer-eval-labels/0.1"

WORKLIST_SCHEMA = "paperwhisperer-label-worklist/1"

LANGUAGES = ("en", "zh", "mixed")
KINDS = ("prose", "table", "formula")

SOURCE_HUMAN = "human"
SOURCE_ACCEPTED = "human_accepted_prefill"
SOURCE_ASSISTANT = "assistant_draft"

#: A record source is gate-grade only when a person confirmed it. There is no
#: fourth value and no default, because a default is where "assume human"
#: creeps in.
HUMAN_SOURCES = frozenset({SOURCE_HUMAN, SOURCE_ACCEPTED})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Passage windows
# --------------------------------------------------------------------------

#: Windows are ~900 chars. Big enough that a whole idea survives, small enough
#: that marking "this one, not that one" is a decision rather than a judgement
#: about five pages.
WINDOW_CHARS = 900
WINDOW_OVERLAP = 150

#: Sentence enders for Chinese and English alike. Splitting on whitespace would
#: tear CJK text into whole clauses, because CJK has no inter-word spaces.
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；;.!?])|\n+")


def _is_cjk_dominant(text: str) -> bool:
    cjk = sum(1 for char in text if "一" <= char <= "鿿")
    letters = sum(1 for char in text if char.isalpha())
    return bool(letters) and cjk * 2 > letters


def _join(parts: Sequence[str], sample: str) -> str:
    """Join sentence fragments, inserting spaces only where the script has them."""
    if _is_cjk_dominant(sample):
        return "".join(parts)
    return " ".join(parts)


def split_windows(
    text: str,
    *,
    target: int = WINDOW_CHARS,
    overlap: int = WINDOW_OVERLAP,
) -> list[str]:
    """Split one element's text into overlapping passage windows.

    Splitting on sentence boundaries and then packing sentences into ``target``-
    sized windows keeps a window from ending mid-clause. The overlap matters
    less for correctness than it looks: an answer that straddles a boundary
    would otherwise belong to no window at all, and "this passage does not
    contain the answer" would then be an artefact of where we cut.

    The CJK-aware join is not cosmetic. Packing Chinese fragments with spaces
    puts whitespace where none existed in the source, which changes the text the
    lexical leg tokenises and the text a person reads -- both of them looking at
    a document that does not contain those spaces.
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:
        return [text]

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) > target:
            windows.append(_join(current, sentence))
            tail = windows[-1][-overlap:]
            current = [tail] if tail else []
            current_len = len(tail)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        windows.append(_join(current, current[-1]))
    return [window for window in windows if window.strip()]


# --------------------------------------------------------------------------
# Worklist (parse artifacts + pre-ranked candidates)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    window_id: str
    element_id: str
    element_order: int
    locator: dict[str, Any]
    window_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "element_id": self.element_id,
            "element_order": self.element_order,
            "locator": self.locator,
            "window_index": self.window_index,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Window:
        return cls(
            window_id=str(payload["window_id"]),
            element_id=str(payload["element_id"]),
            element_order=int(payload.get("element_order", 0)),
            locator=dict(payload.get("locator") or {}),
            window_index=int(payload.get("window_index", 0)),
            text=str(payload.get("text") or ""),
        )

    def locator_label(self) -> str:
        locator = self.locator
        kind = str(locator.get("kind") or "?")
        for key in ("page", "slide", "paragraph"):
            if locator.get(key) is not None:
                return f"{kind} {key}={locator[key]}"
        return kind


@dataclass
class Worklist:
    """Cached parse artifacts, seeded queries and pre-ranked candidates."""

    corpus_version: str
    built_at: str
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    queries: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKLIST_SCHEMA,
            "corpus_version": self.corpus_version,
            "built_at": self.built_at,
            "documents": self.documents,
            "queries": self.queries,
            "candidates": self.candidates,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Worklist:
        return cls(
            corpus_version=str(payload.get("corpus_version") or ""),
            built_at=str(payload.get("built_at") or ""),
            documents=dict(payload.get("documents") or {}),
            queries=list(payload.get("queries") or []),
            candidates={str(k): list(v) for k, v in (payload.get("candidates") or {}).items()},
        )

    def save(self, path: Path = WORKLIST_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path = WORKLIST_PATH) -> Worklist:
        if not path.is_file():
            raise SystemExit(
                f"找不到 worklist：{path}。先跑 `python tools/label_retrieval.py build`。"
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # -- lookups -----------------------------------------------------------

    def query(self, query_id: str) -> dict[str, Any]:
        for item in self.queries:
            if str(item.get("id")) == query_id:
                return item
        raise KeyError(query_id)

    def windows_for(self, document: str) -> list[Window]:
        doc = self.documents.get(document) or {}
        return [Window.from_dict(item) for item in doc.get("windows") or ()]

    def window(self, document: str, window_id: str) -> Window | None:
        for candidate in self.windows_for(document):
            if candidate.window_id == window_id:
                return candidate
        return None

    def candidates_for(self, query_id: str) -> list[dict[str, Any]]:
        return list(self.candidates.get(query_id) or ())


def _parse_document(path: Path) -> Any:
    """Parse with the legacy adapter -- the same parser ``eval/gates.py`` uses by default."""
    from paperwhisperer.ingestion.adapters.legacy_loader import (  # noqa: PLC0415
        LegacyDocumentParser,
    )
    from paperwhisperer.ingestion.gateway import ParseOptions  # noqa: PLC0415

    return LegacyDocumentParser().parse(str(path), ParseOptions()).document


def _document_windows(document: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for element in document.elements:
        text = str(element.text or "")
        for index, chunk in enumerate(split_windows(text)):
            rows.append(Window(
                window_id=f"{element.element_id}#w{index}",
                element_id=str(element.element_id),
                element_order=int(getattr(element, "order", 0)),
                locator=dict(element.locator.to_dict()),
                window_index=index,
                text=chunk,
            ).to_dict())
    return rows


# --------------------------------------------------------------------------
# Ranking -- a pre-fill, never a label
# --------------------------------------------------------------------------

#: How many ranked candidates to show. Eight is enough to contain the answer
#: often and few enough to be read.
DEFAULT_TOP_K = 8

#: How many *unranked* windows to append as an audit tail. Without them the
#: annotator can only confirm what the retriever already proposed, and the
#: resulting recall would be a property of the candidate list, not of the
#: retriever the gate is meant to measure. Showing 2 of the tail keeps a gold
#: passage the retriever missed reachable.
DEFAULT_AUDIT = 2


def _lexical_rank(query: str, windows: Sequence[Window]) -> list[str]:
    """Rank by the production retriever's own term-coverage score.

    Reusing ``lexical_query_forms`` + ``query_coverage`` means the pre-fill is
    not a second, private ranking function that will disagree with the retriever
    it is meant to preview.
    """
    from paperwhisperer.rag.retrieve import (  # noqa: PLC0415
        lexical_query_forms,
        query_coverage,
    )

    forms = lexical_query_forms(query) or (query,)
    scored: list[tuple[float, str]] = []
    for window in windows:
        scores = [score for score in (query_coverage(form, window.text) for form in forms)
                  if score is not None]
        best = max(scores) if scores else 0.0
        scored.append((-best, window.window_id))
    scored.sort()
    return [window_id for _, window_id in scored]


def _dense_rank(query: str, windows: Sequence[Window]) -> list[str]:
    """Rank by local cosine similarity from ``fastembed``.

    Optional by design: an annotator without the model still gets a worklist, it
    just has no dense leg. A labelling tool that refuses to start because an
    optional dependency is missing is a tool nobody runs.
    """
    try:
        from paperwhisperer.llm.embeddings_local import (  # noqa: PLC0415
            build_embedder_from_config,
        )
    except Exception:  # noqa: BLE001 - absence must not stop labelling
        return []
    try:
        provider = build_embedder_from_config()
        query_vec = provider.embed_query(query)
        batch = provider.embed_documents([window.text for window in windows])
    except Exception:  # noqa: BLE001 - model load/download failure
        return []
    vectors = list(getattr(batch, "vectors", ()) or ())
    if len(vectors) != len(windows):
        return []

    def cosine(left: Any, right: Any) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    scored = [(-cosine(query_vec, vec), window.window_id)
              for vec, window in zip(vectors, windows, strict=False)]
    scored.sort()
    return [window_id for _, window_id in scored]


def _audit_tail(query_id: str, ranked: Sequence[str], pool: Sequence[str], count: int) -> list[str]:
    """A deterministic sample of windows the ranking did *not* promote.

    Deterministic because the worklist is shared by every annotator: a random
    tail would make two annotators look at different questions.
    """
    tail = [window_id for window_id in pool if window_id not in set(ranked)]
    if not tail or count <= 0:
        return []
    rng = random.Random(f"{WORKLIST_SCHEMA}:{query_id}")
    return rng.sample(tail, min(count, len(tail)))


def rank_candidates(
    query: str,
    query_id: str,
    windows: Sequence[Window],
    *,
    top_k: int = DEFAULT_TOP_K,
    audit: int = DEFAULT_AUDIT,
) -> list[dict[str, Any]]:
    """Fuse the lexical and dense legs, then append an audit tail."""
    from paperwhisperer.rag.retrieve import reciprocal_rank_fusion  # noqa: PLC0415

    lexical = _lexical_rank(query, windows)
    dense = _dense_rank(query, windows)
    legs: dict[str, list[str]] = {"lexical": lexical}
    if dense:
        legs["dense"] = dense

    fused = reciprocal_rank_fusion(legs)
    ranked = [entry.chunk_id for entry in fused][:top_k]
    leg_ranks = {
        entry.chunk_id: {hit.leg: hit.rank for hit in entry.legs} for entry in fused
    }

    chosen = list(ranked)
    chosen += _audit_tail(query_id, chosen, [w.window_id for w in windows], audit)
    by_id = {w.window_id: w for w in windows}
    return [
        {
            "window_id": window_id,
            "element_id": by_id[window_id].element_id,
            "rank": index + 1,
            "legs": leg_ranks.get(window_id, {}),
            "origin": "ranked" if index < len(ranked) else "audit_tail",
            "text": by_id[window_id].text,
        }
        for index, window_id in enumerate(chosen)
    ]


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _load_seeds(path: Path) -> list[dict[str, Any]]:
    import yaml  # noqa: PLC0415

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seeds = list(payload.get("queries") or [])
    if not seeds:
        raise SystemExit(f"{path} 里没有 queries。")
    for index, seed in enumerate(seeds):
        missing = [key for key in ("id", "document", "query") if not seed.get(key)]
        if missing:
            raise SystemExit(f"{path} queries[{index}] 缺少 {missing}。")
    return seeds


def _corpus_version() -> str:
    manifest = REPO_ROOT / "eval" / "manifest.yaml"
    if not manifest.is_file():
        return ""
    import yaml  # noqa: PLC0415

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return str(payload.get("corpus_version") or "")


def cmd_build(args: argparse.Namespace) -> int:
    seeds = _load_seeds(Path(args.seeds))
    corpus_root = Path(args.corpus_root) if args.corpus_root else REPO_ROOT / "eval" / "corpus"

    worklist = Worklist(corpus_version=_corpus_version(), built_at=_now())
    for seed in seeds:
        document = str(seed["document"])
        if document not in worklist.documents:
            path = corpus_root / document
            if not path.is_file():
                print(f"!! 语料缺失，跳过：{path}", file=sys.stderr)
                continue
            started = time.monotonic()
            parsed = _parse_document(path)
            windows = _document_windows(parsed)
            worklist.documents[document] = {
                "sha256": str(getattr(parsed, "sha256", "")),
                "parser": str(getattr(parsed, "parser_name", "legacy")),
                "elements": len(getattr(parsed, "elements", ())),
                "windows": windows,
            }
            print(
                f"parsed {document}: {len(windows)} windows from "
                f"{len(getattr(parsed, 'elements', ()))} elements "
                f"({time.monotonic() - started:.1f}s)"
            )

    skipped = 0
    for seed in seeds:
        query_id = str(seed["id"])
        document = str(seed["document"])
        if document not in worklist.documents:
            skipped += 1
            continue
        windows = [Window.from_dict(item) for item in worklist.documents[document]["windows"]]
        worklist.queries.append({
            "id": query_id,
            "document": document,
            "query": str(seed["query"]),
            "language": str(seed.get("language") or ""),
            "kind": str(seed.get("kind") or ""),
            "query_source": str(seed.get("query_source") or "seed"),
            "note": str(seed.get("note") or ""),
        })
        worklist.candidates[query_id] = rank_candidates(
            str(seed["query"]),
            query_id,
            windows,
            top_k=args.top_k,
            audit=args.audit,
        )

    path = worklist.save(Path(args.out) if args.out else WORKLIST_PATH)
    print(
        f"\n{len(worklist.queries)} queries, {len(worklist.documents)} documents"
        + (f", {skipped} skipped (corpus file missing)" if skipped else "")
        + f"\nworklist -> {path}"
    )
    return 0


# --------------------------------------------------------------------------
# dump -- the reading surface
# --------------------------------------------------------------------------


def cmd_dump(args: argparse.Namespace) -> int:
    worklist = Worklist.load(Path(args.worklist) if args.worklist else WORKLIST_PATH)
    wanted = set(args.query or ())
    for query in worklist.queries:
        query_id = str(query["id"])
        if wanted and query_id not in wanted:
            continue
        if args.language and query.get("language") not in args.language:
            continue
        if args.kind and query.get("kind") not in args.kind:
            continue
        print("=" * 100)
        print(f"[{query_id}] {query['document']}")
        print(f"Q: {query['query']}")
        print(
            f"seed classification: language={query.get('language') or '?'} "
            f"kind={query.get('kind') or '?'} (query_source={query.get('query_source')})"
        )
        for candidate in worklist.candidates_for(query_id):
            locator = worklist.window(str(query["document"]), str(candidate["window_id"]))
            where = locator.locator_label() if locator else "?"
            legs = ",".join(f"{leg}#{rank}" for leg, rank in sorted(candidate["legs"].items()))
            print("-" * 100)
            print(
                f"  {candidate['rank']:>2}. {candidate['window_id']}"
                f"  element={candidate['element_id']}  {where}"
                f"  [{candidate['origin']}]  legs={legs}"
            )
            print(f"      {_preview(str(candidate['text']))}")
        print()
    return 0


def _preview(text: str, limit: int = 320) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + " …"


# --------------------------------------------------------------------------
# Decision log -- append-only
# --------------------------------------------------------------------------

#: Anything a filesystem may refuse, folded to ``_``. The colon is here for a
#: concrete reason rather than defensively: an annotator id is namespaced
#: (``agent:a``, ``human:ann`` -- see docs/LABELLING.md) and the separator is a
#: colon, while the log's *name* was the id verbatim. So every annotator the
#: documented commands produce had a filename Windows refuses, which meant the
#: decision logs could not be checked out there at all, and ``export`` then
#: quietly produced an empty gate input -- a tool that reports "no decisions"
#: where a Linux checkout reports 53. The id is a fact inside the log; the
#: filename is only where it is written, and it is allowed to be a rendering.
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _filename_for(annotator: str) -> str:
    """The annotator's id as a filename component.

    Lossy on purpose: ``agent:a`` and ``agent_a`` collide. That is acceptable
    because the *name* is never read back as the id -- :func:`_annotators_on_disk`
    recovers it from the log's own records, where it cannot be a rendering.
    """
    stem = _FILENAME_UNSAFE.sub("_", str(annotator or "")).strip("._-")
    return stem or "annotator"


def _log_path(annotator: str, kind: str = "relevance") -> Path:
    return ANNOTATIONS_DIR / f"{_filename_for(annotator)}.{kind}.jsonl"


def _annotators_on_disk() -> list[str]:
    """Every annotator with a relevance log, by the id the log itself records.

    Not derived from the filename. ``_log_path`` folds a colon to an underscore,
    so reversing it from the name would report ``agent_a`` for a log whose every
    row says ``agent:a`` -- and the id is what ``collect_export`` writes into
    ``annotated_by`` and what :func:`_annotator_kind` classifies, so a wrong one
    here is wrong in the gate's input file.

    The id is a field of every record, so it is read from there. A log whose rows
    are all unreadable falls back to the sanitised stem rather than being
    dropped, so a truncated file still shows up in ``status``.
    """
    if not ANNOTATIONS_DIR.is_dir():
        return []
    found: set[str] = set()
    for path in sorted(ANNOTATIONS_DIR.glob("*.relevance.jsonl")):
        annotator = ""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            annotator = str(row.get("annotator") or "")
            if annotator:
                break
        found.add(annotator or path.name.split(".")[0])
    return sorted(name for name in found if name)


def read_decisions(annotator: str, kind: str = "relevance") -> list[dict[str, Any]]:
    path = _log_path(annotator, kind)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A half-written line (an interrupted session) must not make the log
            # unreadable. The decision it held is lost, which is exactly what
            # "resumable" means -- it can be entered again.
            continue
    return rows


def append_decision(annotator: str, record: dict[str, Any], kind: str = "relevance") -> None:
    path = _log_path(annotator, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def latest_decisions(annotator: str, kind: str = "relevance") -> dict[str, dict[str, Any]]:
    """Latest decision per subject id. The log keeps history; this is its head."""
    head: dict[str, dict[str, Any]] = {}
    for row in read_decisions(annotator, kind):
        key = str(row.get("subject_id") or row.get("query_id") or row.get("claim_id") or "")
        head[key] = row
    return head


def _revision_index(annotator: str, subject_id: str, kind: str = "relevance") -> int:
    return sum(1 for row in read_decisions(annotator, kind)
               if str(row.get("subject_id") or row.get("query_id")) == subject_id)


def make_relevance_record(
    *,
    annotator: str,
    query: dict[str, Any],
    window_ids: Sequence[str],
    element_ids: Sequence[str],
    language: str,
    kind: str,
    source: str,
    outside_candidates: Sequence[str] = (),
    seconds: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    query_id = str(query["id"])
    index = _revision_index(annotator, query_id)
    previous = latest_decisions(annotator).get(query_id)
    action = "first" if previous is None else (
        "reaffirm" if list(previous.get("window_ids") or []) == list(window_ids)
        else "revise"
    )
    return {
        "subject_id": query_id,
        "query_id": query_id,
        "document": str(query.get("document") or ""),
        "annotator": annotator,
        "recorded_at": _now(),
        "action": action,
        "revision": index,
        "window_ids": list(window_ids),
        "element_ids": list(element_ids),
        "outside_candidates": list(outside_candidates),
        "language": language,
        "kind": kind,
        "source": source,
        "human_confirmed": source in HUMAN_SOURCES,
        "seconds": None if seconds is None else round(float(seconds), 1),
        "note": note,
    }


def make_claim_record(
    *,
    annotator: str,
    claim: dict[str, Any],
    supported: bool | None,
    source: str,
    seconds: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    claim_id = str(claim["id"])
    index = _revision_index(annotator, claim_id, "claims")
    previous = latest_decisions(annotator, "claims").get(claim_id)
    action = "first" if previous is None else (
        "reaffirm" if previous.get("supported") is supported else "revise"
    )
    return {
        "subject_id": claim_id,
        "claim_id": claim_id,
        "query_id": str(claim.get("query_id") or ""),
        "element_id": str(claim.get("element_id") or ""),
        "quote": str(claim.get("quote") or ""),
        "annotator": annotator,
        "recorded_at": _now(),
        "action": action,
        "revision": index,
        "supported": supported,
        "source": source,
        "human_confirmed": source in HUMAN_SOURCES,
        "seconds": None if seconds is None else round(float(seconds), 1),
        "note": note,
    }


# --------------------------------------------------------------------------
# label -- interactive and answer-sheet
# --------------------------------------------------------------------------

HELP_TEXT = """\
输入直接回车 = 接受预填（记为 human_accepted_prefill）
  1,3-5   选中这些候选段落作为"能回答这一问"的段落
  none    这些候选里没有能回答的（0 / n 也行）
  w 3     打印第 3 段全文
  s 词    在全文里搜一个词，并把命中段落加到列表末尾
  l zh    改这一问的 language（en|zh|mixed）
  k table 改这一问的 kind（prose|table|formula）
  b       回到上一问（会带着上次的答案，用来改主意）
  x       跳过（不决定，status 里会看到，resumable）
  q       保存并退出
  ?       这段帮助
"""


def _parse_selection(text: str, count: int) -> list[int] | None:
    """``"1,3-5"`` -> ``[1, 3, 4, 5]``. ``None`` means "not a selection"."""
    cleaned = text.strip()
    if not cleaned:
        return None
    picked: list[int] = []
    for part in cleaned.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start, _, end = part.partition("-")
            try:
                lo, hi = int(start), int(end)
            except ValueError:
                return None
            picked.extend(range(lo, hi + 1))
        else:
            try:
                picked.append(int(part))
            except ValueError:
                return None
    if not picked or any(index < 1 or index > count for index in picked):
        return None
    seen: set[int] = set()
    return [index for index in picked if not (index in seen or seen.add(index))]


def _classify(text: str, allowed: Sequence[str], current: str) -> str:
    value = text.strip().lower()
    return value if value in allowed else current


def _element_ids_for(worklist: Worklist, document: str, window_ids: Sequence[str]) -> list[str]:
    """Window ids -> the element ids the scorer reads.

    The label schema is element-level while a person judges a window, and the
    collapse is lossy in one direction only: several windows of one element
    become one ground-truth element. That is fine for Recall@K, which asks
    whether the top-K contained *a* relevant element. What is not fine would be
    hiding it, which is why the export records the window ids too.
    """
    element_ids: list[str] = []
    for window_id in window_ids:
        window = worklist.window(document, window_id)
        if window and window.element_id not in element_ids:
            element_ids.append(window.element_id)
    return element_ids


def _match_windows(worklist: Worklist, document: str, needle: str) -> list[str]:
    low = needle.lower()
    return [w.window_id for w in worklist.windows_for(document) if low in w.text.lower()]


def _pending_queries(worklist: Worklist, annotator: str, *, review: bool) -> list[dict[str, Any]]:
    """What this run should show: the untouched ones, or only the decided ones.

    ``review`` is the revision path. Cheap revision is the requirement, so the
    tool does not make it a separate mode of thinking: ``review`` is the same
    screen with the previous answer already filled in, and a re-examination that
    changes nothing is recorded as ``reaffirm`` rather than discarded. Knowing
    how often a label survived re-reading is part of trusting it.
    """
    head = latest_decisions(annotator)
    out: list[dict[str, Any]] = []
    for query in worklist.queries:
        record = head.get(str(query["id"]))
        skipped = record is not None and record.get("action") == "skip"
        if review:
            if record is not None and not skipped:
                out.append(query)
        elif record is None or skipped:
            out.append(query)
    return out


def cmd_label(args: argparse.Namespace) -> int:
    worklist = Worklist.load(Path(args.worklist) if args.worklist else WORKLIST_PATH)
    annotator = args.annotator
    if args.answers:
        return _label_from_sheet(worklist, annotator, Path(args.answers), args)

    decided = latest_decisions(annotator)
    queue = _pending_queries(worklist, annotator, review=args.review)
    if args.limit:
        queue = queue[: args.limit]
    if not queue:
        print(f"{annotator}: 没有待标注的查询。")
        return 0
    print(f"{annotator}: {len(queue)} 个查询待标注"
          + ("（复核模式：只显示已决定过的）" if args.review else ""))
    print(HELP_TEXT)

    history: list[dict[str, Any]] = []
    index = 0
    while 0 <= index < len(queue):
        query = queue[index]
        query_id = str(query["id"])
        document = str(query["document"])
        candidates = worklist.candidates_for(query_id)
        shown: list[str] = [str(item["window_id"]) for item in candidates]
        candidate_texts = {str(item["window_id"]): str(item["text"]) for item in candidates}
        previous = decided.get(query_id)
        language = str(query.get("language") or "")
        kind = str(query.get("kind") or "")

        print("=" * 100)
        print(f"[{query_id}] {document}")
        print(f"Q: {query['query']}")
        print(f"language={language or '?'} kind={kind or '?'}")
        for position, window_id in enumerate(shown, start=1):
            locator = worklist.window(document, window_id)
            where = locator.locator_label() if locator else "?"
            print(f"  {position:>2}. {where:<18} {_preview(candidate_texts[window_id], 240)}")
        if previous is not None and previous.get("action") != "skip":
            last = ", ".join(str(item) for item in previous.get("window_ids") or []) or "none"
            print(f"  [上次] {last}（{previous.get('action')}，{previous.get('recorded_at')}）")

        started = time.monotonic()
        decided_now = False
        while True:
            raw = args.stream.readline()
            if not raw:
                print("\n(输入结束) 已保存的决定保持不变。")
                return 0
            line = raw.strip()
            low = line.lower()
            if low in ("q", "quit", "exit"):
                print("\n保存并退出。")
                return 0
            if low in ("?", "h", "help"):
                print(HELP_TEXT)
                continue
            if low == "b":
                if index == 0:
                    print("已经是第一问了。")
                    continue
                index -= 1
                break
            if low == "x":
                record = {
                    "subject_id": query_id,
                    "query_id": query_id,
                    "document": document,
                    "annotator": annotator,
                    "recorded_at": _now(),
                    "action": "skip",
                    "window_ids": [],
                    "element_ids": [],
                    "language": language,
                    "kind": kind,
                    "source": "skip",
                    "human_confirmed": False,
                    "seconds": round(time.monotonic() - started, 1),
                    "note": "",
                }
                append_decision(annotator, record)
                decided[query_id] = record
                decided_now = True
                break
            if low.startswith("w "):
                target = _parse_selection(line[2:], len(shown))
                if not target:
                    print("用法：w 3")
                    continue
                for position in target:
                    print(f"--- {shown[position - 1]} ---")
                    print(candidate_texts[shown[position - 1]])
                continue
            if low.startswith("s "):
                needle = line[2:].strip()
                hits = _match_windows(worklist, document, needle)
                added = 0
                for window_id in hits:
                    if window_id in shown:
                        continue
                    window = worklist.window(document, window_id)
                    if window is None:
                        continue
                    shown.append(window_id)
                    candidate_texts[window_id] = window.text
                    added += 1
                    print(f"  {len(shown):>2}. {window_id} {window.locator_label():<18}"
                          f" {_preview(window.text, 240)}")
                if not added:
                    print(f"没有新增命中（全文里共 {len(hits)} 处命中）。")
                continue
            if low.startswith("l "):
                language = _classify(line[2:], LANGUAGES, language)
                print(f"language={language}")
                continue
            if low.startswith("k "):
                kind = _classify(line[2:], KINDS, kind)
                print(f"kind={kind}")
                continue
            if low in ("none", "n", "0"):
                picked: list[int] = []
                source = SOURCE_HUMAN
            elif line == "":
                # Enter accepts the pre-fill verbatim. It is a confirmation, but
                # a weaker one than typing the set out, and the two are counted
                # separately: "a person pressed enter" and "a person decided"
                # are different evidence about the same label.
                picked = list(range(1, len(shown) + 1))
                source = SOURCE_ACCEPTED
            else:
                picked = _parse_selection(line, len(shown))
                if picked is None:
                    print(f"看不懂 {line!r}（候选是 1..{len(shown)}，输入 ? 看帮助）。")
                    continue
                source = SOURCE_HUMAN

            selected_ids = [shown[position - 1] for position in picked]
            # A selection the ranking never proposed is a retriever miss, found by
            # a person. It is the most valuable thing this tool can produce, so it
            # is surfaced immediately rather than buried in the log.
            candidate_ids = {str(item["window_id"])
                             for item in worklist.candidates_for(query_id)}
            outside = [window_id for window_id in selected_ids
                       if window_id not in candidate_ids]
            record = make_relevance_record(
                annotator=annotator,
                query=query,
                window_ids=selected_ids,
                element_ids=_element_ids_for(worklist, document, selected_ids),
                language=language,
                kind=kind,
                source=source,
                outside_candidates=outside,
                seconds=time.monotonic() - started,
            )
            append_decision(annotator, record)
            decided[query_id] = record
            decided_now = True
            if outside:
                print(f"  记下了排序之外的答案段落：{', '.join(outside)}"
                      "（记为 retriever miss）")
            break

        if decided_now:
            history.append(decided[query_id])
        index += 1

    _print_summary(annotator, history)
    return 0


def _label_from_sheet(
    worklist: Worklist, annotator: str, path: Path, args: argparse.Namespace
) -> int:
    """Bulk mode: the same decisions, written down in one pass.

    An answer sheet is the only sane way to annotate from an editor, and it is
    how an assistant hands a person a first pass they can edit. Every value is
    validated against the worklist, so a typo fails loudly here rather than
    becoming a label that quietly measures nothing.
    """
    import yaml  # noqa: PLC0415

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sheet_source = str(payload.get("source") or (SOURCE_ASSISTANT if args.agent else SOURCE_HUMAN))
    if sheet_source not in HUMAN_SOURCES | {SOURCE_ASSISTANT}:
        raise SystemExit(f"{path} 的 source 必须是 human / human_accepted_prefill / "
                         "assistant_draft 之一。")
    rows = list(payload.get("decisions") or [])
    if not rows:
        raise SystemExit(f"{path} 没有 decisions。")

    written = 0
    for index, row in enumerate(rows):
        query_id = str(row.get("query_id") or "")
        try:
            query = worklist.query(query_id)
        except KeyError:
            raise SystemExit(
                f"{path} decisions[{index}] 的 query_id {query_id!r} 不在 worklist 里。"
            ) from None
        document = str(query["document"])
        candidates = [str(item["window_id"]) for item in worklist.candidates_for(query_id)]
        chosen = row.get("window_ids")
        if chosen is None:
            chosen = row.get("relevant_window_ids")
        picked: list[str] = []
        if chosen is not None:
            for window_id in chosen:
                window_id = str(window_id)
                if worklist.window(document, window_id) is None:
                    raise SystemExit(
                        f"{path} decisions[{index}] 选的 {window_id} 不在文档 {document} 里。"
                    )
                if window_id not in picked:
                    picked.append(window_id)
        explicit_none = bool(row.get("none_relevant"))
        if not picked and not explicit_none and chosen is not None:
            raise SystemExit(
                f"{path} decisions[{index}] 给了空列表但没有 none_relevant: true。"
                "空列表和'这些都不对'是两件事，前者是没填。"
            )
        outside = [w for w in picked if w not in candidates]
        source = str(row.get("source") or sheet_source)
        if source not in HUMAN_SOURCES | {SOURCE_ASSISTANT}:
            raise SystemExit(f"{path} decisions[{index}] 的 source 非法：{source!r}")
        language = str(row.get("language") or query.get("language") or "")
        kind = str(row.get("kind") or query.get("kind") or "")
        if language not in LANGUAGES:
            raise SystemExit(f"{path} decisions[{index}] language={language!r} 非法。")
        if kind not in KINDS:
            raise SystemExit(f"{path} decisions[{index}] kind={kind!r} 非法。")
        record = make_relevance_record(
            annotator=annotator,
            query=query,
            window_ids=picked,
            element_ids=_element_ids_for(worklist, document, picked),
            language=language,
            kind=kind,
            source=source,
            outside_candidates=outside,
            seconds=row.get("seconds"),
            note=str(row.get("note") or ""),
        )
        append_decision(annotator, record)
        written += 1

    print(f"{annotator}: 写入 {written} 条决定（source={sheet_source}）-> {_log_path(annotator)}")
    if sheet_source == SOURCE_ASSISTANT:
        print(
            "  注意：assistant_draft 的记录 human_confirmed=false，"
            "export 默认不会把它们写进门禁的输入文件。"
        )
    return 0


def _print_summary(annotator: str, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    actions = Counter(str(record.get("action")) for record in records)
    sources = Counter(str(record.get("source")) for record in records)
    elapsed = [float(record["seconds"]) for record in records if record.get("seconds")]
    print(
        f"\n本轮 {len(records)} 条：{dict(actions)}；来源 {dict(sources)}"
        + (f"；平均 {sum(elapsed) / len(elapsed):.1f}s/条" if elapsed else "")
    )
    print(f"日志 -> {_log_path(annotator)}")


# --------------------------------------------------------------------------
# claims -- the axis that cannot be self-scored
# --------------------------------------------------------------------------


def build_claim_candidates(
    worklist: Worklist,
    head: dict[str, dict[str, Any]],
    *,
    per_query: int = 2,
) -> list[dict[str, Any]]:
    """Claims to be judged, built from the ranked evidence -- never from a label.

    Each candidate pairs the question with a passage the retriever ranked and
    quotes that passage's opening sentences. The annotator then says whether the
    quote actually supports an answer to the question. This is the extractive
    path's citation precision, not an LLM's: the claim set is a local,
    deterministic proxy for what the product would emit, and it is honest about
    being that. What matters for the gate is that ``supported`` is never
    derivable from the passage -- it is the one field a system cannot grade
    itself.
    """
    claims: list[dict[str, Any]] = []
    for query in worklist.queries:
        query_id = str(query["id"])
        record = head.get(query_id)
        if record is None or record.get("action") == "skip":
            continue
        document = str(query["document"])
        selected = [str(w) for w in (record.get("window_ids") or [])]
        ranked = [str(item["window_id"]) for item in worklist.candidates_for(query_id)]
        pool = [w for w in ranked if w in selected] or ranked
        for position, window_id in enumerate(pool[:per_query], start=1):
            window = worklist.window(document, window_id)
            if window is None:
                continue
            quote = _quote_of(window.text)
            claims.append({
                "id": f"{query_id}-c{position}",
                "query_id": query_id,
                "document": document,
                "query": str(query["query"]),
                "element_id": window.element_id,
                "window_id": window_id,
                "quote": quote,
                "claim_text": str(query["query"]),
                "locator": window.locator_label(),
                "generation": "extractive_top_rank",
            })
    return claims


def _quote_of(text: str, sentences: int = 2, limit: int = 400) -> str:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    quote = " ".join(parts[:sentences]) if parts else text
    return quote[:limit]


def cmd_claims(args: argparse.Namespace) -> int:
    worklist = Worklist.load(Path(args.worklist) if args.worklist else WORKLIST_PATH)
    annotator = args.annotator
    head = latest_decisions(annotator)
    claims = build_claim_candidates(worklist, head, per_query=args.per_query)

    if args.candidates_out:
        Path(args.candidates_out).write_text(
            json.dumps({"claims": claims}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{len(claims)} 条 claim 候选 -> {args.candidates_out}")

    if args.answers:
        return _claims_from_sheet(annotator, claims, Path(args.answers), args)

    judged = latest_decisions(annotator, "claims")
    pending = [claim for claim in claims if str(claim["id"]) not in judged]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"{annotator}: 没有待判的 claim。")
        return 0
    print(f"{annotator}: {len(pending)} 条 claim 待判。输入 y/n/x（跳过）/q（退出）。")
    done: list[dict[str, Any]] = []
    for claim in pending:
        window = worklist.window(str(claim.get("document") or ""), str(claim["window_id"]))
        print("=" * 100)
        print(f"[{claim['id']}] {claim.get('locator', '')} element={claim['element_id']}")
        print(f"Q: {claim['query']}")
        print(f"引用：{claim['quote']}")
        if args.full and window is not None:
            print(f"--- {window.window_id} 全文 ---\n{window.text}")
        started = time.monotonic()
        while True:
            raw = args.stream.readline()
            if not raw:
                print("\n(输入结束)")
                return 0
            low = raw.strip().lower()
            if low == "q":
                print("保存并退出。")
                return 0
            if low == "x":
                append_decision(annotator, {
                    "subject_id": str(claim["id"]),
                    "claim_id": str(claim["id"]),
                    "query_id": str(claim["query_id"]),
                    "element_id": str(claim["element_id"]),
                    "annotator": annotator,
                    "recorded_at": _now(),
                    "action": "skip",
                    "supported": None,
                    "source": "skip",
                    "human_confirmed": False,
                    "seconds": round(time.monotonic() - started, 1),
                    "note": "",
                }, "claims")
                break
            if low in ("y", "yes", "n", "no"):
                record = make_claim_record(
                    annotator=annotator,
                    claim=claim,
                    supported=low in ("y", "yes"),
                    source=SOURCE_HUMAN,
                    seconds=time.monotonic() - started,
                )
                append_decision(annotator, record, "claims")
                judged[str(claim["id"])] = record
                done.append(record)
                break
            print("只接受 y / n / x / q。")
    _print_claims_summary(annotator, done)
    return 0


def _claims_from_sheet(
    annotator: str, claims: Sequence[dict[str, Any]], path: Path, args: argparse.Namespace
) -> int:
    import yaml  # noqa: PLC0415

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sheet_source = str(payload.get("source") or (SOURCE_ASSISTANT if args.agent else SOURCE_HUMAN))
    by_id = {str(claim["id"]): claim for claim in claims}
    written = 0
    for index, row in enumerate(payload.get("decisions") or []):
        claim_id = str(row.get("claim_id") or "")
        if claim_id not in by_id:
            raise SystemExit(f"{path} decisions[{index}] 的 claim_id {claim_id!r} 不在候选里。")
        supported = row.get("supported")
        if not isinstance(supported, bool):
            raise SystemExit(
                f"{path} decisions[{index}] 的 supported 必须是 true/false，不能留空："
                "引用是否有据只有人能判。"
            )
        source = str(row.get("source") or sheet_source)
        if source not in HUMAN_SOURCES | {SOURCE_ASSISTANT}:
            raise SystemExit(f"{path} decisions[{index}] 的 source 非法：{source!r}")
        record = make_claim_record(
            annotator=annotator,
            claim=by_id[claim_id],
            supported=supported,
            source=source,
            seconds=row.get("seconds"),
            note=str(row.get("note") or ""),
        )
        append_decision(annotator, record, "claims")
        written += 1
    print(f"{annotator}: 写入 {written} 条 claim 判定 -> {_log_path(annotator, 'claims')}")
    return 0


def _print_claims_summary(annotator: str, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    supported = sum(1 for record in records if record.get("supported") is True)
    elapsed = [float(record["seconds"]) for record in records if record.get("seconds")]
    print(
        f"\n本轮 {len(records)} 条：{supported} 条判为有据"
        + (f"；平均 {sum(elapsed) / len(elapsed):.1f}s/条" if elapsed else "")
    )


# --------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------


def cohens_kappa(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Cohen's kappa over paired binary judgements, with the degenerate cases named.

    A kappa reported without its confusion matrix is a number nobody can check,
    and both degenerate directions matter here: perfect agreement on a single
    category (``p_e == 1``) makes kappa undefined rather than 1.0, and a tiny
    ``p_e`` makes it wildly unstable.
    """
    total = len(pairs)
    if total == 0:
        return {"n": 0, "observed": None, "expected": None, "kappa": None,
                "note": "没有重叠的标注对"}
    both = sum(1 for left, right in pairs if left and right)
    neither = sum(1 for left, right in pairs if not left and not right)
    left_only = sum(1 for left, right in pairs if left and not right)
    right_only = sum(1 for left, right in pairs if not left and right)
    observed = (both + neither) / total
    p_left = (both + left_only) / total
    p_right = (both + right_only) / total
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    matrix = {
        "both_relevant": both,
        "only_left": left_only,
        "only_right": right_only,
        "both_irrelevant": neither,
    }
    if expected >= 1.0:
        return {
            "n": total, "observed": observed, "expected": expected, "kappa": None,
            "matrix": matrix,
            "note": "两人在同一批项目上都给了同一个类别，kappa 无定义（分母为 0）",
        }
    return {
        "n": total,
        "observed": observed,
        "expected": expected,
        "kappa": (observed - expected) / (1 - expected),
        "matrix": matrix,
        "note": "",
    }


def compare_annotators(
    worklist: Worklist, left: str, right: str
) -> dict[str, Any]:
    """Agreement over the queries both annotators decided.

    Two units, because they answer different questions:

    * **per-window** -- each candidate window is a binary item ("does this
      passage answer the question?"). Kappa over that is the standard number.
    * **per-query** -- did they end up pointing at the same passage *set*, and
      did each of them find anything at all. This is what a label consumer
      cares about, and a high per-window kappa with a low per-query Jaccard is
      a real and diagnosable state (both marking the same few windows in
      passing, disagreeing about the actual answer).
    """
    left_head = latest_decisions(left)
    right_head = latest_decisions(right)

    query_ids = [
        str(query["id"])
        for query in worklist.queries
        if str(query["id"]) in left_head and str(query["id"]) in right_head
        and left_head[str(query["id"])].get("action") != "skip"
        and right_head[str(query["id"])].get("action") != "skip"
    ]

    pairs: list[tuple[bool, bool]] = []
    rows: list[dict[str, Any]] = []
    jaccards: list[float] = []
    set_agreements: list[bool] = []
    for query_id in query_ids:
        left_windows = set(left_head[query_id].get("window_ids") or [])
        right_windows = set(right_head[query_id].get("window_ids") or [])
        for window_id in sorted(left_windows | right_windows):
            pairs.append((window_id in left_windows, window_id in right_windows))
        union = left_windows | right_windows
        jaccard = (len(left_windows & right_windows) / len(union)) if union else 1.0
        jaccards.append(jaccard)
        same_set = left_windows == right_windows
        set_agreements.append(same_set)
        if not same_set:
            rows.append({
                "query_id": query_id,
                "left_windows": sorted(left_windows),
                "right_windows": sorted(right_windows),
                "jaccard": round(jaccard, 3),
                "only_left": sorted(left_windows - right_windows),
                "only_right": sorted(right_windows - left_windows),
            })

    exact_set_rate = (sum(1 for value in set_agreements if value) / len(set_agreements)
                      if set_agreements else None)
    mean_jaccard = (sum(jaccards) / len(jaccards)) if jaccards else None
    classification_disagreements = [
        {
            "query_id": query_id,
            "left": [left_head[query_id].get("language"), left_head[query_id].get("kind")],
            "right": [right_head[query_id].get("language"), right_head[query_id].get("kind")],
        }
        for query_id in query_ids
        if (left_head[query_id].get("language"), left_head[query_id].get("kind"))
        != (right_head[query_id].get("language"), right_head[query_id].get("kind"))
    ]

    return {
        "annotators": [left, right],
        "queries_overlap": len(query_ids),
        "queries_left": len(left_head),
        "queries_right": len(right_head),
        "window_level": cohens_kappa(pairs),
        "exact_set_agreement": exact_set_rate,
        "mean_window_jaccard": mean_jaccard,
        "classification_disagreements": classification_disagreements,
        "disagreements": sorted(rows, key=lambda item: (item["jaccard"], item["query_id"])),
    }


def cmd_agreement(args: argparse.Namespace) -> int:
    worklist = Worklist.load(Path(args.worklist) if args.worklist else WORKLIST_PATH)
    report = compare_annotators(worklist, args.left, args.right)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"-> {args.out}")
    print(text)
    return 0


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def annotator_status(worklist: Worklist, annotator: str) -> dict[str, Any]:
    rows = read_decisions(annotator)
    claims = read_decisions(annotator, "claims")
    head = latest_decisions(annotator)
    claim_head = latest_decisions(annotator, "claims")
    actions = Counter(str(row.get("action")) for row in rows)
    sources = Counter(str(row.get("source")) for row in rows)
    elapsed = [float(row["seconds"]) for row in rows if row.get("seconds")]
    claim_elapsed = [float(row["seconds"]) for row in claims if row.get("seconds")]
    decided = [qid for qid, row in head.items() if row.get("action") != "skip"]
    return {
        "annotator": annotator,
        "queries_total": len(worklist.queries),
        "queries_decided": len(decided),
        "queries_skipped": sum(1 for row in head.values() if row.get("action") == "skip"),
        "queries_remaining": len(worklist.queries) - len(head),
        "rows_written": len(rows),
        "revisions": actions.get("revise", 0),
        "reaffirmations": actions.get("reaffirm", 0),
        "sources": dict(sources),
        "human_confirmed_rows": sum(1 for row in head.values()
                                    if row.get("human_confirmed")),
        "assistant_rows": sum(1 for row in head.values()
                              if row.get("source") == SOURCE_ASSISTANT),
        "claims_decided": sum(1 for row in claim_head.values()
                              if row.get("supported") is not None),
        "claim_rows_written": len(claims),
        "claim_revisions": sum(1 for row in claims if row.get("action") == "revise"),
        "seconds_per_query_mean": (round(sum(elapsed) / len(elapsed), 1) if elapsed else None),
        "seconds_per_claim_mean": (round(sum(claim_elapsed) / len(claim_elapsed), 1)
                                   if claim_elapsed else None),
        "relevance_pick_rate": (
            round(sum(len(row.get("window_ids") or []) for row in head.values())
                  / len(decided), 2) if decided else None
        ),
    }


def cmd_status(args: argparse.Namespace) -> int:
    worklist = Worklist.load(Path(args.worklist) if args.worklist else WORKLIST_PATH)
    names = list(args.annotator or [])
    if not names:
        names = _annotators_on_disk()
    if not names:
        print("还没有任何标注者日志。")
        return 0
    for name in names:
        payload = annotator_status(worklist, name)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------
# export -- exactly what eval.gates.load_labels reads
# --------------------------------------------------------------------------


def _yaml_dump(payload: dict[str, Any], header: Sequence[str] = ()) -> str:
    import yaml  # noqa: PLC0415

    lines = [f"# {line}" for line in header]
    body = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    return "\n".join(lines + [body]) if lines else body


def collect_export(
    worklist: Worklist,
    annotators: Sequence[str],
    *,
    accept_assistant: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build the gate payload and the pending payload from the decision logs.

    Two files, always. The gate's input file carries what may be scored; the
    pending file carries what the tool has but no person confirmed. Splitting
    them is what keeps "the tool produced it" and "a person confirmed it" from
    being the same row in the same file.

    **One row per id, always.** Two annotators labelling the same 20 questions is
    the normal case, not a mistake, and the scorer keys both axes by id
    (``retrieved[query_id]`` for recall, one ``supported`` boolean per claim).
    Emitting both annotators' rows would therefore double every question in the
    recall denominator and let two opposite judgements sit in one file as if
    they were two facts. So duplicate ids are *consolidated* by
    :func:`_consolidate_query` / :func:`_consolidate_claim`, which merge the
    evidence and record the disagreement on the row instead of hiding it.
    """
    by_query: dict[str, list[dict[str, Any]]] = {}
    by_claim: dict[str, list[dict[str, Any]]] = {}
    pending_queries: list[dict[str, Any]] = []
    pending_claims: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for annotator in annotators:
        head = latest_decisions(annotator)
        for query in worklist.queries:
            query_id = str(query["id"])
            record = head.get(query_id)
            if record is None or record.get("action") == "skip":
                continue
            element_ids = [str(item) for item in (record.get("element_ids") or [])]
            if not element_ids:
                # "None of the candidates answers this" is a real judgement and
                # it stays in the log, but the label schema cannot express it:
                # load_labels rejects an empty relevant_element_ids, because an
                # unanswerable row would otherwise be a free hit for every
                # retriever. Excluded from the denominator, counted in the notes.
                warnings.append({
                    "id": f"{annotator}/{query_id}",
                    "reason": "none_relevant：答案不在任何候选段落里。"
                              "标签格式无法表达（load_labels 拒绝空 relevant_element_ids），"
                              "因此不计入召回率的分母——这是必须知道的一个向上偏差。",
                    "exported": False,
                })
                continue
            human = bool(record.get("human_confirmed"))
            row = {
                "id": query_id,
                "document": str(query.get("document") or ""),
                "query": str(query.get("query") or ""),
                "language": str(record.get("language") or ""),
                "kind": str(record.get("kind") or ""),
                "relevant_element_ids": element_ids,
                "relevant_window_ids": [str(item) for item in (record.get("window_ids") or [])],
                # -- provenance: read by a person reviewing the file, ignored by
                #    load_labels. Present so nobody can mistake the origin.
                "annotated_by": annotator,
                "annotated_by_kind": _annotator_kind(annotator),
                "provenance": "human_confirmed" if human else "assistant_draft",
                "human_confirmed": human,
                "decision_source": str(record.get("source") or ""),
                "revisions": int(record.get("revision") or 0),
                "action": str(record.get("action") or ""),
                "selected_outside_ranked_candidates": list(record.get("outside_candidates") or []),
                "recorded_at": str(record.get("recorded_at") or ""),
            }
            if human:
                by_query.setdefault(query_id, []).append(row)
                continue
            if accept_assistant:
                by_query.setdefault(query_id, []).append(row)
                warnings.append({
                    "id": f"{annotator}/{query_id}",
                    "reason": "assistant_draft：没有人确认；已按 --accept-assistant-labels 导出",
                    "exported": True,
                })
            else:
                pending_queries.append(row)

        claim_head = latest_decisions(annotator, "claims")
        for claim_id, record in sorted(claim_head.items()):
            if record.get("supported") is None:
                continue
            human = bool(record.get("human_confirmed"))
            row = {
                "id": claim_id,
                "query_id": str(record.get("query_id") or ""),
                "quote": str(record.get("quote") or ""),
                "element_id": str(record.get("element_id") or ""),
                "supported": bool(record["supported"]),
                "annotated_by": annotator,
                "annotated_by_kind": _annotator_kind(annotator),
                "provenance": "human_confirmed" if human else "assistant_draft",
                "human_confirmed": human,
                "decision_source": str(record.get("source") or ""),
                "revisions": int(record.get("revision") or 0),
                "action": str(record.get("action") or ""),
                "recorded_at": str(record.get("recorded_at") or ""),
                "generation": "extractive_top_rank",
            }
            if human:
                by_claim.setdefault(claim_id, []).append(row)
                continue
            if accept_assistant:
                by_claim.setdefault(claim_id, []).append(row)
                warnings.append({
                    "id": f"{annotator}/{claim_id}",
                    "reason": "assistant_draft：没有人确认；已按 --accept-assistant-labels 导出",
                    "exported": True,
                })
            else:
                pending_claims.append(row)

    queries = [_consolidate_query(rows, warnings) for _id, rows in sorted(by_query.items())]
    claims = [_consolidate_claim(rows, warnings) for _id, rows in sorted(by_claim.items())]

    payload = {
        "schema": LABELS_SCHEMA,
        "corpus_version": worklist.corpus_version,
        "annotators": list(annotators),
        "human_confirmed": not any(not row["human_confirmed"] for row in queries + claims),
        "queries": queries,
        "claims": claims,
    }
    pending = {
        "schema": "paperwhisperer-eval-labels-pending/0.1",
        "corpus_version": worklist.corpus_version,
        "note": (
            "工具产出但没有人确认的记录。这些不在门禁的输入文件里。"
            "删掉它们比把它们并进去更安全：并进去就是让系统给自己打分。"
        ),
        "queries": pending_queries,
        "claims": pending_claims,
    }
    return payload, pending, warnings


def _annotator_kind(annotator: str) -> str:
    """Annotator names are prefixed with their kind: ``human:`` or ``agent:``."""
    if annotator.startswith("agent:"):
        return "assistant"
    if annotator.startswith("human:"):
        return "human"
    return "assistant"


def _merge_element_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    merged: list[str] = []
    for row in rows:
        for element_id in row.get("relevant_element_ids") or ():
            if element_id not in merged:
                merged.append(str(element_id))
    return merged


def _consolidate_query(rows: Sequence[dict[str, Any]],
                       warnings: list[dict[str, Any]]) -> dict[str, Any]:
    """One row per query id, from however many annotators labelled it.

    The union of the relevant elements is the ground truth when two annotators
    disagree, and that choice is stated rather than made silently. Union is the
    right direction for *recall*: recall asks whether the top-K contained a
    passage that answers the question, and a passage one annotator accepted
    really does answer it. The alternative — intersection — would throw away a
    correct answer because two people were uneven, and it would flatter nothing.
    The cost is that the merged ground truth is a little easier to hit, so the
    row carries the disagreement for whoever adjudicates it.
    """
    if len(rows) == 1:
        return dict(rows[0])
    annotators = sorted(str(row.get("annotated_by")) for row in rows)
    element_sets = {tuple(row.get("relevant_element_ids") or ()) for row in rows}
    agreed = len(element_sets) == 1
    merged = dict(rows[0])
    merged["relevant_element_ids"] = _merge_element_ids(rows)
    merged["annotated_by"] = ",".join(annotators)
    merged["annotated_by_kind"] = ("human" if all(
        str(row.get("annotated_by_kind")) == "human" for row in rows) else "mixed")
    merged["human_confirmed"] = all(bool(row.get("human_confirmed")) for row in rows)
    merged["provenance"] = "human_confirmed" if merged["human_confirmed"] else "assistant_draft"
    merged["n_annotators"] = len(rows)
    merged["annotators_agreed"] = agreed
    if not agreed:
        merged["annotator_element_sets"] = {
            str(row.get("annotated_by")): [str(item) for item in
                                           (row.get("relevant_element_ids") or ())]
            for row in rows
        }
        # A disagreement on the language/kind bucket is a disagreement about
        # which denominator a row lands in, so it is reported on the row too.
        classifications = {
            str(row.get("annotated_by")): [str(row.get("language")), str(row.get("kind"))]
            for row in rows
        }
        if len(set(map(tuple, classifications.values()))) > 1:
            merged["annotator_classifications"] = classifications
            merged["classification_agreed"] = False
    else:
        merged["classification_agreed"] = True
    warnings.append({
        "id": str(merged.get("id")),
        "reason": (
            f"{len(rows)} 个标注者标了同一问，"
            + ("结论一致" if agreed else "结论不一致：已取 relevant_element_ids 的并集，"
                                  "分歧记录在 annotator_element_sets")
        ),
        "exported": True,
    })
    return merged


def _consolidate_claim(rows: Sequence[dict[str, Any]],
                       warnings: list[dict[str, Any]]) -> dict[str, Any]:
    """One row per claim id.

    ``supported`` is a single boolean by construction, so a contested claim
    cannot be averaged into a probability and stay loadable. When annotators
    disagree the tie is broken toward ``supported: false`` -- the conservative
    reading for a precision gate is "not established", not "probably fine" --
    and both votes are written onto the row. A disputed row is exactly the kind
    of row a person should adjudicate before the gate is trusted.
    """
    if len(rows) == 1:
        return dict(rows[0])
    annotators = sorted(str(row.get("annotated_by")) for row in rows)
    votes = {str(row.get("annotated_by")): bool(row.get("supported")) for row in rows}
    merged = dict(rows[0])
    merged["supported"] = all(votes.values())
    merged["annotated_by"] = ",".join(annotators)
    merged["annotated_by_kind"] = ("human" if all(
        str(row.get("annotated_by_kind")) == "human" for row in rows) else "mixed")
    merged["human_confirmed"] = all(bool(row.get("human_confirmed")) for row in rows)
    merged["provenance"] = "human_confirmed" if merged["human_confirmed"] else "assistant_draft"
    merged["n_annotators"] = len(rows)
    merged["annotator_votes"] = votes
    merged["annotators_agreed"] = len(set(votes.values())) == 1
    warnings.append({
        "id": str(merged.get("id")),
        "reason": (
            f"{len(rows)} 个标注者判了同一条 claim，"
            + ("结论一致" if merged["annotators_agreed"]
               else "结论不一致：取保守值 supported=false，两人的票记录在 annotator_votes")
        ),
        "exported": True,
    })
    return merged


def cmd_export(args: argparse.Namespace) -> int:
    worklist = Worklist.load(Path(args.worklist) if args.worklist else WORKLIST_PATH)
    annotators = list(args.annotator or [])
    if not annotators:
        annotators = _annotators_on_disk()
        if not annotators:
            raise SystemExit("还没有任何标注者日志。")

    payload, pending, warnings = collect_export(
        worklist, annotators, accept_assistant=args.accept_assistant_labels
    )
    if not payload["queries"]:
        raise SystemExit("没有可导出的 query 决定。")

    assistant_rows = [row for row in payload["queries"] + payload["claims"]
                      if not row["human_confirmed"]]
    header = [
        "Generated by tools/label_retrieval.py. Do not hand-edit.",
        "Loader: eval.gates.load_labels (paperwhisperer-eval-labels/*).",
    ]
    if assistant_rows:
        header += [
            "",
            "!! NOT GATE-GRADE: this file contains rows no person confirmed.",
            f"!! {len(assistant_rows)} of {len(payload['queries']) + len(payload['claims'])}"
            " rows are assistant_draft (human_confirmed: false).",
            "!! A gate scored against these is the system grading itself.",
            "!! Re-export after a person confirms, or delete these rows.",
        ]
    else:
        header += [
            "",
            "Every row below carries human_confirmed: true.",
        ]

    out_path = Path(args.out) if args.out else EXPORT_PATH
    out_path.write_text(_yaml_dump(payload, header), encoding="utf-8")

    pending_path = Path(args.pending_out) if args.pending_out else PENDING_PATH
    pending_path.write_text(_yaml_dump(pending), encoding="utf-8")

    print(f"export -> {out_path}  ({len(payload['queries'])} queries, "
          f"{len(payload['claims'])} claims, "
          f"human_confirmed={payload['human_confirmed']})")
    print(f"pending -> {pending_path}  ({len(pending['queries'])} queries, "
          f"{len(pending['claims'])} claims)")
    if assistant_rows:
        print(
            f"\n{len(assistant_rows)} 条记录没有被任何人确认。它们在门禁输入文件里，"
            "因此**这不是可用于门禁的真值**，见 docs/LABELLING.md。"
        )
    if warnings:
        print("\n需要知道：")
        for line in warnings:
            mark = "已导出" if line["exported"] else "未导出"
            print(f"  - [{mark}] {line['id']}: {line['reason']}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label_retrieval",
        description="标注检索相关性（Recall@K 真值）与引用支持（引用精度真值）。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="解析语料、切窗口、给每条 query 预排候选")
    build.add_argument("--seeds", required=True, help="query 种子文件（YAML）")
    build.add_argument("--corpus-root", default=None)
    build.add_argument("--out", default=None)
    build.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    build.add_argument("--audit", type=int, default=DEFAULT_AUDIT,
                       help="额外展示多少条排序之外的候选（审计尾）")
    build.set_defaults(func=cmd_build)

    dump = sub.add_parser("dump", help="打印候选段落（标注者的阅读面）")
    dump.add_argument("--worklist", default=None)
    dump.add_argument("--query", action="append", help="只看这些 query id（可重复）")
    dump.add_argument("--language", action="append", choices=LANGUAGES)
    dump.add_argument("--kind", action="append", choices=KINDS)
    dump.set_defaults(func=cmd_dump)

    label = sub.add_parser("label", help="记录相关性决定（可交互、可批量、可复核）")
    label.add_argument("--annotator", required=True,
                       help="标注者名，human:xxx 表示人，agent:xxx 表示助手草稿")
    label.add_argument("--worklist", default=None)
    label.add_argument("--answers", default=None, help="answer sheet（YAML），批量模式")
    label.add_argument("--agent", action="store_true",
                       help="answer sheet 的 source 默认记为 assistant_draft")
    label.add_argument("--review", action="store_true",
                       help="复核模式：只显示已决定过的查询（改主意用这个）")
    label.add_argument("--limit", type=int, default=0)
    label.set_defaults(func=cmd_label, stream=sys.stdin)

    claims = sub.add_parser("claims", help="记录 claim 的 supported 判断")
    claims.add_argument("--annotator", required=True)
    claims.add_argument("--worklist", default=None)
    claims.add_argument("--per-query", type=int, default=2)
    claims.add_argument("--candidates-out", default=None,
                        help="把 claim 候选写成 JSON（给 answer sheet 用）")
    claims.add_argument("--answers", default=None)
    claims.add_argument("--agent", action="store_true")
    claims.add_argument("--limit", type=int, default=0)
    claims.add_argument("--full", action="store_true",
                        help="同时打印被引段落全文（判断 supported 时常需要）")
    claims.set_defaults(func=cmd_claims, stream=sys.stdin)

    agree = sub.add_parser("agreement", help="两个标注者的一致性")
    agree.add_argument("left")
    agree.add_argument("right")
    agree.add_argument("--worklist", default=None)
    agree.add_argument("--out", default=None)
    agree.set_defaults(func=cmd_agreement)

    status = sub.add_parser("status", help="标注进度")
    status.add_argument("--worklist", default=None)
    status.add_argument("--annotator", action="append")
    status.set_defaults(func=cmd_status)

    export = sub.add_parser("export", help="导出 retrieval_labels.yaml")
    export.add_argument("--worklist", default=None)
    export.add_argument("--annotator", action="append")
    export.add_argument("--out", default=None)
    export.add_argument("--pending-out", default=None)
    export.add_argument("--accept-assistant-labels", action="store_true",
                        help="把没有人确认的 assistant_draft 也写进门禁输入文件")
    export.set_defaults(func=cmd_export)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())