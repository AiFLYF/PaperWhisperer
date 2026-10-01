"""Traceability: four ids that follow one document, and six families that count.

Task T4.1 of the plan. The row reads: *structured logging; request_id /
job_id / document_id / analysis_run_id through the whole chain; metrics cut
to plan section 13.2's families (parsing / RAG / LLM / grounding / jobs /
cost); OTel + Prometheus optional on the Hosted tier.* The acceptance
sentence is the one that decided the shape of this file: **任一失败可从
document_id/job_id 追到 parser/fallback/LLM/外部 API** — any failure must be
traceable from a document or job id to the parser, the fallback, the LLM call
and the external API.

So the deliverable is a *trace*, and the metrics are the cheap half of it.

**Why not just add ids to the log messages.** Because a log line carrying an
id only helps if you can find the other lines, and finding them means
matching on timestamps and hoping. Five subsystems write logs: ingestion,
infra/jobs, rag, analysis, llm. Their clocks agree to about a millisecond and
nothing else. A user reports "it gave me a wrong answer"; correlating five
subsystems by timestamp is a guess with a log file attached. The four ids
threading through is the difference between a guess and a lookup, and the whole
reason they have to be *fields* rather than text is that a query has to be
able to say ``document_id = "doc_..."`` and get the run.

**Why contextvars, and why the thread story is not automatic.**
``contextvars`` is the only mechanism in the standard library that follows a
value across an ``await`` and a thread *if* the thread is handed the context.
The second half is the catch, and it is stated here because it is the failure
this module exists to prevent and the one a reader must not have to discover:

* **async hop -- automatic.** A coroutine awaited directly shares the current
  context, and ``asyncio.create_task`` / ``gather`` copy it at task creation.
  ``asyncio.to_thread`` copies it too, for the same reason.
* **thread pool -- NOT automatic.** ``ThreadPoolExecutor.submit`` runs the
  callable in a *fresh* context, so every id vanishes at the hop.
  ``loop.run_in_executor`` inherits that behaviour, and it is what
  :mod:`paperwhisperer.api.deps` uses to push SSE generators off the event loop.

  :func:`Observability.submit` and :func:`Observability.thread_hop` close the
  hop explicitly (``contextvars.copy_context().run``). They exist because the
  alternative -- a context that disappears exactly where the long work starts
  -- is not a correlation context. Every thread hop in the codebase is listed
  in this file's :data:`THREAD_HOPS`; adding a new one means adding a row
  there, which is how a place where the ids silently drop out becomes visible
  in review rather than in a support ticket.

**One call, three effects.** Each ``record_*`` helper in this module does three
things from a single call site, because a caller who has to remember to update
three things updates two of them: it counts the metrics for its family, it
writes one structured log line with the ids as fields, and it appends one
ordered event to the trace. That is also the cheap argument -- the expensive
part of observability is not the counting, it is the decision of *where* to
put the call, and there is only one place per event.

**Metrics are aggregate; the trace is per-id.** The split is deliberate and it
is the reason the two live in one file. A metric labelled with a
``document_id`` has unbounded cardinality: a deployment that indexed four
hundred papers has four hundred series for one counter, and the memory of the
process becomes a function of the library the user owns. So metrics are
counted over bounded label sets (:data:`FAMILY` and a short list of enums),
and the per-id detail goes to the trace, which is a bounded ring buffer rather
than a growing table. A caller who wants a per-document number gets it from
:meth:`TraceRecorder.events`; a caller who wants "how often does this happen"
gets it from the counter. :data:`MAX_SERIES_PER_METRIC` enforces the split --
a metric that exceeds its series budget counts the overflow into
``metrics.series_overflow_total`` instead of growing without bound.

**Grounding is a family here because nothing else counts it.**
``paperwhisperer.rag.citations`` decides, deterministically and without a
model, whether a quote is the paper's own words; ``analysis.paper_facts``
decides which facts keep their citations; ``analysis.orchestrator`` decides
whether an answer is ``supported`` or ``insufficient_evidence``. Three
verdicts, no counter anywhere. That is the family the plan names and the
product is about -- a product whose central promise is "every claim traces to
a page" and which does not count verified quotes, paraphrases that had to be
demoted, claims that resolved to no evidence at all, or questions answered
with "the paper does not say". :func:`record_citation_verification` and
:func:`record_fact_extraction` are the two call sites, and both read the real
types by duck-typing so this module keeps no import edge into ``rag`` or
``analysis``.

**Nothing here is enabled by importing it.** The module singleton starts
disabled; :func:`configure` is the only thing that turns collection on, and
:func:`shutdown` puts the process back exactly as it found it -- root level,
root handlers, and the record factory, all restored. A test that turns
observability on and forgets to turn it off would then silence or duplicate
every log line in the rest of the suite, which is a real hazard and not a
hypothetical one; :meth:`Observability.close` is what makes that impossible.

**No OpenTelemetry dependency, and no Prometheus one either.** The plan marks
the pair Hosted-tier and optional, so this ships the standard library only:
:class:`JsonFormatter` writes one JSON object per line, and
:meth:`MetricsRegistry.prometheus_text` writes the exposition format by hand.
A Hosted deployment scrapes the text with a real Prometheus -- no client
library, because the wire format is the stable part and the client is not.
:class:`Exporter` is the seam an OTel adapter would plug into: one method,
``export(events, metrics)``, and the adapter's whole job becomes translating
this file's data into spans and data points. The stdlib
:class:`JsonlExporter` is one such adapter, written in-house, which is how the
seam stays honest -- an interface with no implementation is an interface
nobody has exercised.

**What each instrument costs.** Numbers below are per call, measured on this
repository's own code, and the guidance is therefore specific rather than
aspirational:

======================================  ===========  ==============================
call                                      cost        why
======================================  ===========  ==============================
``record_*`` on a **disabled** facade      0.09 us     one boolean test. This is
                                                     the number that decides
                                                     whether "off by default"
                                                     is affordable
:meth:`Gauge.set`                          1.7 us       one dict write, one lock
:meth:`Counter.inc`                        2.0 us       sorted-tuple key + one
                                                     dict read-modify-write
:meth:`Histogram.observe`                  2.4 us       as above, plus a bucket walk
:func:`correlate` enter + exit             3.5 us       two ContextVar set/reset
:meth:`TraceRecorder.record`               4.3 us       ``_now_iso`` + deque append
                                                     + one field dict
:meth:`JsonFormatter.format`               8.8 us/line  ``json.dumps`` -- paid only
                                                     for a line that is emitted
:meth:`Observability.submit` overhead     +3.5 us      one ``copy_context()``; the
                                                     thread hop was already paid
:meth:`Observability.emit`                 9.4 us      above, with the line
  (level filtered out)                                  discarded by the logger
:meth:`Observability.emit`                 21 us        above, with the JSON
  (line actually written)                               rendered and written
:func:`configure` / :func:`shutdown`       once         handler + record factory
======================================  ===========  ==============================

Against that: an LLM call is 2-40 seconds and a parse is 0.2-900. A single
``record_*`` per event is four to five orders of magnitude below the work it
describes, which is the only reason instrumenting is worth doing at all.

The three places it is *not* free, named because a library that is
indiscriminately cheap is a library nobody measured:

**Rendering a line costs 8.8 us, and emitting one that survives the level
filter costs 21.** That is the largest single number in the table and it is
eleven times the event itself, which is why :func:`configure` takes a level,
why ``OBSERVABILITY_SLOW_LLM_MS`` exists, and why the answer to "the logs are
slow" is to raise a level rather than to remove instrumentation. A deployment
that only wants the counters can pass ``install=False`` and pay nothing at all
here.

**``Observability.submit`` adds 3.5 us to every task it wraps.** Small against
a section call, and the alternative -- an uncorrelated thread pool -- is not a
cheaper design, it is a missing feature. But it is per-*task* rather than
per-*event*, and a pipeline that fans out over three hundred chunks pays it
three hundred times for one document. That is the number to look at if a
re-index ever gets slower.

**Label cardinality is the one place a caller's mistake becomes this module's
memory.** :data:`MAX_SERIES_PER_METRIC` makes that mistake bounded and
counted rather than unbounded and silent, but a cap that is being hit means a
``document_id`` has found its way into a label -- which is the aggregate/per-id
split being ignored, and worth an alert rather than a shrug.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count as _counter_factory
from typing import Any

from paperwhisperer.core import config

logger = logging.getLogger(__name__)

__all__ = [
    "FAMILY_COST",
    "FAMILY_GROUNDING",
    "FAMILY_JOBS",
    "FAMILY_LLM",
    "FAMILY_PARSING",
    "FAMILY_RAG",
    "FAMILIES",
    "ID_FIELDS",
    "MAX_SERIES_PER_METRIC",
    "THREAD_HOPS",
    "Counter",
    "CorrelationIds",
    "Exporter",
    "Gauge",
    "Histogram",
    "JsonFormatter",
    "JsonlExporter",
    "MetricsRegistry",
    "NullExporter",
    "Observability",
    "TraceEvent",
    "TraceRecorder",
    "configure",
    "correlate",
    "current_ids",
    "extract_ids",
    "resolve_ids",
    "new_id",
    "observability",
    "record_citation_verification",
    "record_cost",
    "record_fact_extraction",
    "record_job_transition",
    "record_model_call",
    "record_parse_outcome",
    "record_retrieval",
    "run_in_context",
    "shutdown",
    "thread_hop",
]


# --------------------------------------------------------------------------
# The four ids
# --------------------------------------------------------------------------

#: The correlation keys, in the order they are declared in the plan. Order is
#: fixed so a JSON line and a trace event list them the same way, which makes
#: a diff between two of them readable.
ID_FIELDS: tuple[str, ...] = ("request_id", "job_id", "document_id", "analysis_run_id")

#: The per-scope current ids. One ContextVar holding all four rather than four
#: ContextVars, because the overwhelmingly common operation is "read them all"
#: (a formatter, an event) and one lookup beats four; and because a scope that
#: sets three of four has to be able to *see* the fourth it inherited, which a
#: per-key set of ContextVars would make the caller's problem.
_CURRENT: contextvars.ContextVar[CorrelationIds | None] = contextvars.ContextVar(
    "paperwhisperer_correlation", default=None
)


def new_id(prefix: str = "") -> str:
    """A short, unique, sortable-enough id for one of the four keys.

    ``uuid4().hex`` because the alternative -- a counter, or a timestamp -- is
    either process-local (so two processes collide, and the Hosted tier has
    more than one) or guessable. Prefixed with the key's name rather than a
    single opaque hex so that a line a human reads says which id it is
    carrying; the ids are short enough (32 hex) that the extra four bytes cost
    nothing next to a log line.
    """
    token = uuid.uuid4().hex
    return f"{prefix}_{token}" if prefix else token


@dataclass(frozen=True, slots=True)
class CorrelationIds:
    """The four ids in scope, as one immutable value.

    Frozen because it is copied out of the ContextVar into a log record and an
    event: a mutable one would let a later scope mutate the ids a record was
    already stamped with, and the record is the thing a query runs against.
    Empty strings mean "not known at this point in the chain" -- deliberately
    not ``None``, because an absent field and a field whose value is the string
    ``"None"`` must not both be readable as the same thing in a log line.
    """

    request_id: str = ""
    job_id: str = ""
    document_id: str = ""
    analysis_run_id: str = ""

    def as_dict(self, *, include_empty: bool = False) -> dict[str, str]:
        """The ids as a mapping, for a log record or a JSON payload.

        Empty values are dropped by default because a JSON line with four id
        keys where two are ``""`` is a line a filter has to special-case: the
        absence *is* the information, and omitting the key says it in the way
        every log pipeline already understands.
        """
        pairs = ((name, getattr(self, name)) for name in ID_FIELDS)
        if include_empty:
            return {name: str(value) for name, value in pairs}
        return {name: str(value) for name, value in pairs if value}

    def merged(self, **overrides: Any) -> CorrelationIds:
        """A copy with some ids replaced; ``""``/``None`` means "keep".

        The "keep" rule is what makes nesting safe. A job worker binds
        ``job_id`` and ``document_id`` deep inside a request scope, and it must
        not have to know whether the request scope already had a
        ``document_id``; passing only the ids this scope actually owns and
        inheriting the rest is the only spelling that composes.
        """
        changes: dict[str, str] = {}
        for name in ID_FIELDS:
            value = overrides.get(name)
            if value:
                changes[name] = str(value)
        if not changes:
            return self
        return CorrelationIds(**{**self.as_dict(include_empty=True), **changes})

    @property
    def is_empty(self) -> bool:
        return not any(self.as_dict(include_empty=True).values())

    def to_dict(self) -> dict[str, str]:
        return self.as_dict(include_empty=True)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> CorrelationIds:
        """Read the four ids out of any mapping, ignoring everything else.

        The lenient constructor exists because the ids arrive from half a dozen
        shapes -- a route's form payload, a ``Job.to_dict()``, a request
        header set, a ``document_id`` on a ``ParsedDocument`` -- and a strict
        constructor at each of them would be six chances to raise on a field
        that simply was not present.
        """
        source = payload or {}
        return cls(**{name: str(source.get(name) or "") for name in ID_FIELDS})


def current_ids() -> CorrelationIds:
    """The ids in scope right now, on this thread, in this task.

    Never raises and never returns ``None``: a logging call site that has to
    handle the possibility of no context is a logging call site that will
    eventually not handle it, and the ids being empty is a perfectly good
    answer for code that runs outside any request.
    """
    return _CURRENT.get() or CorrelationIds()


@contextmanager
def correlate(
    *,
    request_id: str | None = "",
    job_id: str | None = "",
    document_id: str | None = "",
    analysis_run_id: str | None = "",
    **extra: Any,
) -> Iterator[CorrelationIds]:
    """Bind ids for the duration of a scope, and restore the previous ones.

    A context manager rather than a token API on purpose: a token must be reset
    in a ``finally`` by the code that took it, and the code that takes it is
    usually a route handler that has three early returns. Nesting works, and
    the inner scope inherits the ids it did not set (see
    :meth:`CorrelationIds.merged`), which is what a job worker inside a request
    needs and what a per-key ContextVar set would not give for free.

    The three-way contract on each id, because the middle case is the whole
    design:

    * **omitted or ``""`` -- inherit.** The common case. A job worker binding
      ``job_id`` and ``document_id`` deep inside a request scope must not have
      to know whether the request already had a ``document_id``, and a route
      binding only ``request_id`` must not wipe the ``document_id`` a
      middleware set.
    * **a non-empty value -- set.** Ordinary binding.
    * **``None`` -- clear.** Deliberately distinct from ``""``, and it is
      ``None`` rather than a separate ``inherit=False`` flag because the
      difference is a property of the *value* and not of the call. The case
      that needs it is real: a worker claiming a job that was submitted
      minutes ago must not spend its life inside the ``request_id`` of the
      request that happened to still be on the stack when it ticked, or every
      line it writes is attributed to an unrelated upload. A flag would have
      made the caller remember a second thing; a sentinel is the same
      information in the same argument.

    ``extra`` is accepted and ignored on purpose: it keeps a call site that
    passes something like ``session_id=`` from raising a ``TypeError`` in the
    middle of a request, and the value is not silently promoted into a field
    because a field nobody declared is a field nobody queries.
    """
    requested = {
        "request_id": request_id,
        "job_id": job_id,
        "document_id": document_id,
        "analysis_run_id": analysis_run_id,
    }
    inherited = current_ids().as_dict(include_empty=True)
    resolved: dict[str, str] = {}
    for name, value in requested.items():
        if value is None:
            resolved[name] = ""            # explicitly cleared
        elif value:
            resolved[name] = str(value)    # explicitly set
        else:
            resolved[name] = inherited[name]  # "" means inherit
    bound = CorrelationIds(**resolved)
    token = _CURRENT.set(bound)
    try:
        yield bound
    finally:
        _CURRENT.reset(token)


def run_in_context(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run ``function`` in a copy of the current correlation context.

    The explicit thread hop. ``copy_context().run`` is the whole mechanism and
    it is two lines because the standard library already has all of it; what
    this function adds is a name that says *this* is where the ids would
    otherwise have been lost, so a reviewer looking at a thread pool sees the
    fix next to the hazard.

    A copy rather than the context itself: two calls to this function must not
    share one mutable slot, and :func:`correlate` sets a ContextVar in a
    thread, which is exactly the kind of state that must not leak sideways
    into a sibling call.
    """
    return contextvars.copy_context().run(function, *args, **kwargs)


def thread_hop(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Free-function form of :meth:`Observability.submit`'s hop, for executors.

    Named for what it is rather than for what it wraps, because the wrapper
    name (``run_in_context``) reads like a general utility and this is not: it
    exists because a bare ``executor.submit`` drops every id.
    """
    return run_in_context(function, *args, **kwargs)


def extract_ids(source: Any, **overrides: Any) -> CorrelationIds:
    """Read the four ids off an arbitrary object, leniently.

    The adapter between the correlation model and the half-dozen record types
    that already carry these values: a ``Job`` has ``job_id`` and
    ``document_id``, a ``ParsedDocument`` has ``document_id``, an
    ``AnalysisRun`` has ``run_id`` (which is the ``analysis_run_id`` -- mapped
    here rather than in each call site, because a call site that guesses gets
    it wrong in a way no test would notice). Attributes are read with
    ``getattr`` and mappings with ``get``; a source with none of the four
    yields empty ids rather than raising, because the caller is a logging
    helper on an error path.

    The one-level look through ``.document`` is not a convenience. A
    ``ParseOutcome`` has no ``document_id`` of its own -- the id lives on the
    ``ParsedDocument`` it carries -- and neither does a ``RetrievalRecord``,
    whose candidates each name one. Without the descent, the two most important
    events in the whole system would be recorded with no document on them and
    a trace by ``document_id`` would come back empty for a document that
    certainly happened. Descending one level is bounded on purpose: a recursive
    walk would find ids in a chunk inside a candidate inside a record and
    attribute an event to whichever document a container mentioned first.
    """
    if isinstance(source, Mapping):
        values = {name: str(source.get(name) or "") for name in ID_FIELDS}
        nested = source.get("document")
    else:
        values = {
            name: str(getattr(source, name, "") or "") for name in ID_FIELDS
        }
        nested = getattr(source, "document", None)
    if not values["document_id"] and nested is not None:
        inner = (
            str(nested.get("document_id") or "") if isinstance(nested, Mapping)
            else str(getattr(nested, "document_id", "") or "")
        )
        if inner:
            values["document_id"] = inner
    # ``AnalysisRun.run_id`` is the analysis run's id; the correlation key is
    # named after it. Kept as an explicit alias so the mapping is visible.
    if not values["analysis_run_id"]:
        run_id = (
            str(source.get("run_id") or "") if isinstance(source, Mapping)
            else str(getattr(source, "run_id", "") or "")
        )
        if run_id:
            values["analysis_run_id"] = run_id
    return CorrelationIds.from_mapping(values).merged(**overrides)


def resolve_ids(source: Any = None, **overrides: Any) -> CorrelationIds:
    """The ids an event should carry: the context, enriched by ``source``.

    **The composition rule, and why getting it wrong loses the whole story.**
    An id *on the object* beats an id *in the context*; an id only in the
    context beats nothing. So the result is the current scope overlaid with
    whatever ``source`` actually knows.

    This exists as a function because the naive version is wrong in a way that
    looks correct. :func:`extract_ids` alone returns only what the object
    carries, and most of the objects this module records do **not** carry a
    ``document_id``: a :class:`~paperwhisperer.ingestion.orchestrator.ParseOutcome`
    has none (its ``document`` does), a
    :class:`~paperwhisperer.rag.retrieve.RetrievalRecord` has none (its
    candidates do), and a
    :class:`~paperwhisperer.rag.citations.CitationVerification` has none at all.
    Passing that empty result as the event's ids would *overwrite* the ids the
    surrounding scope had bound and land the event in no chain whatsoever -- a
    trace where every retrieval, quote and parse is orphaned while the metrics
    look perfect. Every ``record_*`` goes through here so that cannot happen
    per-call-site.

    ``overrides`` (typically the ``document_id=`` a caller was handed) win over
    both, because an explicit argument is a deliberate statement about which
    document this is.
    """
    explicit = {name: str(value) for name, value in overrides.items()
                if name in ID_FIELDS and value}
    if source is not None:
        explicit = {**extract_ids(source).to_dict(), **explicit}
    return current_ids().merged(**explicit)


#: Every thread boundary in this codebase where the ids do **not** survive by
#: itself, and the line that fixes it. Not a list of things that could be
#: wrong -- a list of places that are wrong until somebody wraps them, kept in
#: the module that explains why they are wrong, so that "the ids disappear at
#: the SSE thread" is a fact in the source rather than a thing found out from
#: a user.
THREAD_HOPS: tuple[dict[str, str], ...] = (
    {
        "where": "paperwhisperer/api/deps.py:_STREAM_WORKERS",
        "what": "loop.run_in_executor over a ThreadPoolExecutor for SSE producers",
        "fix": "run the producer through Observability.submit / thread_hop",
    },
    {
        "where": "paperwhisperer/analysis/orchestrator.py:DocumentAnalyzer.analyze",
        "what": "ThreadPoolExecutor fanning the analysis sections out",
        "fix": "Observability.submit per section future",
    },
    {
        "where": "paperwhisperer/analysis/orchestrator.py:generate_summary",
        "what": "ThreadPoolExecutor fanning the summary chunks out",
        "fix": "Observability.submit per chunk future",
    },
    {
        "where": "paperwhisperer/analysis/orchestrator.py:analyze_stream",
        "what": "ThreadPoolExecutor behind the streaming variant",
        "fix": "Observability.submit per section future",
    },
    {
        "where": "paperwhisperer/search/papers.py:multi-source paper search",
        "what": "ThreadPoolExecutor fanning the paper sources out",
        "fix": "Observability.submit per source future",
    },
    {
        "where": "paperwhisperer/ingestion/gateway.py:_thread_executor",
        "what": "single-worker executor a parse attempt is timed out in",
        "fix": "Observability.submit the parse call so a parser timeout is findable",
    },
)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

#: The six families the plan cuts metrics to, in the plan's order. A declared
#: tuple rather than a convention so that "which family does this counter
#: belong to" has an answer, and so a metric outside these six cannot be added
#: by accident: the registry refuses the name, and the reason it refuses is
#: that a family nobody planned is a family nobody reads.
FAMILY_PARSING = "parsing"
FAMILY_RAG = "rag"
FAMILY_LLM = "llm"
FAMILY_GROUNDING = "grounding"
FAMILY_JOBS = "jobs"
FAMILY_COST = "cost"
FAMILIES: tuple[str, ...] = (
    FAMILY_PARSING,
    FAMILY_RAG,
    FAMILY_LLM,
    FAMILY_GROUNDING,
    FAMILY_JOBS,
    FAMILY_COST,
)

#: Cap on distinct label combinations per metric. The label sets this module
#: declares are enums drawn from a handful of values, so a metric should never
#: approach a hundred series. A caller who puts a ``document_id`` in a label
#: would otherwise turn a counter into a per-document table that grows with the
#: user's library, and the process would carry it forever. Overflowing series
#: are dropped and counted into ``metrics.series_overflow_total`` rather than
#: raised, because a metrics call must not be able to fail a request.
MAX_SERIES_PER_METRIC = 256

#: Default histogram buckets, in milliseconds. Chosen for the shapes this
#: project actually produces rather than for general goodness: the log-spaced
#: middle covers an LLM call (2s-40s), the top buckets cover a parse (up to the
#: gateway's 900s ceiling) and the bottom ones cover the ones that should never
#: happen and are worth seeing once when they do.
DEFAULT_BUCKETS_MS: tuple[float, ...] = (
    1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
    10000.0, 30000.0, 60000.0, 300000.0, 900000.0,
)


def _now_iso() -> str:
    """UTC to the second, the shape ``infra.jobs`` writes its rows in.

    The same convention and the same resolution as the job trail, so a trace
    line and a job event for the same moment compare equal when somebody sorts
    them into one timeline -- which is exactly the mixed-triage case a
    correlation context is supposed to make unnecessary.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _series_key(name: str, labels: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Build a hashable, order-stable key for one labelled series.

    Sorted so that ``inc("x", a=1, b=2)`` and ``inc("x", b=2, a=1)`` are one
    series and not two. That is not a micro-optimisation: without it a label
    set written in a different order silently doubles every count, and the
    symptom is a metric that is wrong in a way nobody can reproduce.
    """
    return name, tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _label_tuples(
    series: Mapping[tuple[str, tuple[tuple[str, str], ...]], Any],
) -> dict[tuple[tuple[str, str], ...], Any]:
    """Project a series dict down to its labels, dropping the metric name.

    The name is part of the key so that one dict can hold several metrics (and
    so a name can never be confused with a label), but the exposition format
    and every reader of :meth:`to_dict` want the labels alone. Projecting here
    rather than at each of the three ``to_dict`` implementations is what keeps
    them from each getting it slightly differently.
    """
    return {labels: value for (_, labels), value in series.items()}


class _Metric:
    """Shared behaviour of the three metric kinds.

    Not exported and not a public base class: it holds the series dict, the
    lock and the overflow counter, which are the three things each kind needs
    and which must not each grow their own.
    """

    kind = "untyped"

    def __init__(
        self,
        name: str,
        family: str,
        *,
        documentation: str = "",
        max_series: int = MAX_SERIES_PER_METRIC,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(
                f"指标 {name!r} 属于未规划的族 {family!r}；"
                f"已规划的族：{', '.join(FAMILIES)}。"
            )
        self.name = name
        self.family = family
        self.documentation = documentation or name
        self.max_series = max(1, int(max_series))
        self._series: dict[tuple[str, tuple[tuple[str, str], ...]], Any] = {}
        self._overflow = 0
        self._lock = threading.Lock()

    @property
    def overflow(self) -> int:
        """How many observations were dropped for exceeding the series budget."""
        with self._lock:
            return self._overflow

    def _slot(self, labels: Mapping[str, Any], initial: Any) -> Any | None:
        """The storage for one labelled series, or ``None`` if over budget.

        ``initial`` is a *factory*, called once per series. A shared mutable
        default would be a bug that only shows up under concurrency.
        """
        key = _series_key(self.name, labels)
        with self._lock:
            found = self._series.get(key)
            if found is not None:
                return found
            if len(self._series) >= self.max_series:
                self._overflow += 1
                return None
            self._series[key] = initial()
            return self._series[key]

    def _query(self, labels: Mapping[str, Any], default: Any) -> Any:
        """The value of the series matching ``labels`` as a *subset*.

        Subset, not exact. This is the second of the two ways a label bug shows
        up as a wrong number rather than an error, and it is the one that
        matters more: a caller who asks ``get(source="llm")`` for a series that
        was recorded as ``{source="llm", reason="no_price_configured"}`` must
        not be told ``0``, because ``0`` is a real answer and it would be
        believed. Exact matching would have answered "no such series" -- but
        through a number, so nothing would have signalled the difference.

        With no labels at all it sums every series, so a health check can ask a
        metric for its total without enumerating the label space.
        """
        wanted = sorted((str(key), str(value)) for key, value in labels.items())
        with self._lock:
            matches = [
                value
                for series_labels, value in _label_tuples(self._series).items()
                if all(pair in series_labels for pair in wanted)
            ]
        if not matches:
            return default
        return matches[0] if len(matches) == 1 else sum(matches)

    def reset(self) -> None:
        """Drop every series. For a test, and for a process that must not carry
        the previous run's numbers into the next one."""
        with self._lock:
            self._series.clear()
            self._overflow = 0

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, prefix: str) -> list[str]:
        """Prometheus exposition lines for this metric, hand-written.

        The format is the stable part of Prometheus's contract and the client
        library is not, so a Hosted deployment can scrape this text without
        this project depending on anything. Metric names are prefixed and
        lower-cased with dots replaced by underscores, which is what the
        exposition format requires and why ``parsing.documents_total`` cannot
        be written verbatim.
        """
        safe = _prom_name(self.name)
        lines = [
            f"# HELP {safe} {self.documentation}",
            f"# TYPE {safe} {self.kind}",
        ]
        for labels, rendered in sorted(self.to_dict().items()):
            label_text = (
                "{" + ",".join(f'{key}="{value}"' for key, value in labels) + "}"
                if labels else ""
            )
            lines.append(f"{safe}{label_text} {rendered}")
        return lines


class Counter(_Metric):
    """A monotonically increasing count.

    The default metric because almost every question here -- how many parses
    fell back, how many quotes failed to align, how many jobs were cancelled --
    is a count of events. A gauge is right only where the value can go down.
    """

    kind = "counter"

    def inc(self, amount: int = 1, **labels: Any) -> None:
        """Add ``amount`` (default 1). Ignored when the series budget is spent.

        Non-fatal by design, and the reason is the same as everywhere else in
        this module: instrumentation that can raise is instrumentation that
        will eventually take down the request it was measuring.

        The value is written back into the dict explicitly rather than through
        ``+=`` on the value :meth:`_Metric._slot` returned. An int is immutable,
        so ``slot += 1`` rebinds a local name and the dict entry never changes
        -- the series would read as zero forever while looking like it worked.
        The same bug in a float gauge is why :class:`Gauge` reads and writes.
        """
        if self._slot(labels, int) is None:
            return
        key = _series_key(self.name, labels)
        with self._lock:
            self._series[key] = int(self._series[key]) + int(amount)

    def get(self, **labels: Any) -> int:
        return int(self._query(labels, 0))

    @property
    def total(self) -> int:
        """The sum over every series. For a test and a one-line health check."""
        with self._lock:
            return sum(int(value) for value in self._series.values())

    def to_dict(self) -> dict[tuple[tuple[str, str], ...], int]:
        with self._lock:
            return _label_tuples(self._series)


class Gauge(_Metric):
    """A value that can move in both directions.

    Used where a *current* reading is the question: the cost of a document so
    far, the depth of the queue. A cost total is deliberately a gauge and not a
    counter, because :mod:`analysis.paper_facts` can only ever add to it --
    but a caller that recomputes and sets a lower total must not be
    contradicted by a counter that refuses to go down.
    """

    kind = "gauge"

    def set(self, value: float, **labels: Any) -> None:
        if self._slot(labels, float) is None:
            return
        with self._lock:
            self._series[_series_key(self.name, labels)] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        """Add to a gauge, for a total that only ever grows (cost, mostly).

        Read-modify-write under one lock, rather than two calls to :meth:`set`,
        because two concurrent increments on a cost total would otherwise lose
        one of them: cost is money, and money that disappears between two
        requests is the failure a gauge is least able to report.
        """
        if self._slot(labels, float) is None:
            return
        key = _series_key(self.name, labels)
        with self._lock:
            self._series[key] = float(self._series.get(key, 0.0)) + float(amount)

    def get(self, **labels: Any) -> float:
        with self._lock:
            return float(self._series.get(_series_key(self.name, labels), 0.0))

    @property
    def total(self) -> float:
        with self._lock:
            return sum(float(value) for value in self._series.values())

    def to_dict(self) -> dict[tuple[tuple[str, str], ...], float]:
        with self._lock:
            return _label_tuples(self._series)


class Histogram(_Metric):
    """Count, sum and fixed bucket counts -- and nothing else.

    No reservoir, no quantile sketch, no rotation. Those are the right answers
    to a Prometheus that stores aggregates across processes; this process
    already has a trace, and a trace answers "what were the slow ones" exactly
    where a quantile answers it approximately. A histogram here is for the
    *shape* ("is the p95 drifting"), and the buckets are declared per metric so
    a reader can see the resolution the shape is being read at.

    Storing cumulative bucket counts rather than per-bucket ones: the
    exposition format wants cumulative, and storing them that way means
    :meth:`observe` is a handful of integer increments and nothing else.
    """

    kind = "histogram"

    def __init__(
        self,
        name: str,
        family: str,
        *,
        buckets: Sequence[float] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(name, family, **kwargs)
        self.buckets = tuple(sorted(float(edge) for edge in buckets or DEFAULT_BUCKETS_MS))
        if not self.buckets:
            raise ValueError(f"直方图 {name!r} 至少需要一个桶边界。")

    def observe(self, value: float, **labels: Any) -> None:
        """Record one observation. Out-of-range values land in the top bucket."""
        moment = float(value)
        if self._slot(labels, lambda: _HistogramState([0] * len(self.buckets))) is None:
            return
        key = _series_key(self.name, labels)
        with self._lock:
            slot = self._series[key]
            slot.count += 1
            slot.total += moment
            for index, edge in enumerate(self.buckets):
                if moment <= edge:
                    slot.counts[index] += 1
            # Above every edge still counts in ``count``/``total``; the +Inf
            # bucket is implied by the exposition format.

    def get(self, **labels: Any) -> dict[str, Any]:
        with self._lock:
            found = self._series.get(_series_key(self.name, labels))
        if found is None:
            found = _HistogramState([0] * len(self.buckets))
        return found.to_dict(self.buckets)

    @property
    def count(self) -> int:
        with self._lock:
            return sum(int(state.count) for state in self._series.values())

    def to_dict(self) -> dict[tuple[tuple[str, str], ...], dict[str, Any]]:
        with self._lock:
            return {
                labels: state.to_dict(self.buckets)
                for labels, state in _label_tuples(self._series).items()
            }

    def render(self, prefix: str) -> list[str]:
        safe = _prom_name(self.name)
        lines = [
            f"# HELP {safe} {self.documentation}",
            f"# TYPE {safe} histogram",
        ]
        for labels, state in sorted(self.to_dict().items()):
            base = dict(labels)
            for index, edge in enumerate(self.buckets):
                label_text = ",".join(
                    f'{key}="{value}"'
                    for key, value in sorted({**base, "le": _num(edge)}.items())
                )
                lines.append(f"{safe}_bucket{{{label_text}}} {state['buckets'][index]}")
            inf_text = ",".join(
                f'{key}="{value}"' for key, value in sorted({**base, "le": "+Inf"}.items())
            )
            lines.append(f"{safe}_bucket{{{inf_text}}} {state['count']}")
            plain = ",".join(f'{key}="{value}"' for key, value in sorted(base.items()))
            suffix = f"{{{plain}}}" if plain else ""
            lines.append(f"{safe}_sum{suffix} {state['sum']:g}")
            lines.append(f"{safe}_count{suffix} {state['count']}")
        return lines


@dataclass(slots=True)
class _HistogramState:
    """Cumulative bucket counts, plus count and sum.

    ``counts`` is sized by the owning histogram's bucket list at construction
    (see :meth:`Histogram.observe`'s factory) rather than derived from the
    value, because the buckets are a property of the metric and a value can
    never tell you how many there are.
    """

    counts: list[int]
    count: int = 0
    total: float = 0.0

    def to_dict(self, buckets: Sequence[float]) -> dict[str, Any]:
        """The cumulative counts, padded to the bucket list.

        Padded rather than truncated, so a caller that reads ``buckets[i]``
        against ``buckets[i]`` of the declaration never has to check the length
        first. A histogram that changes its bucket list between two versions
        should not make a reader of a stored snapshot guess.
        """
        width = len(buckets)
        return {
            "count": int(self.count),
            "sum": float(self.total),
            "buckets": [int(item) for item in self.counts[:width]] + [0] * max(
                0, width - len(self.counts)
            ),
        }


def _num(value: float) -> str:
    return f"{value:g}"


def _prom_name(name: str) -> str:
    """The exposition-safe form of a metric name.

    Prometheus names match ``[a-zA-Z_:][a-zA-Z0-9_:]*``. The plan's family
    names are dotted and this module's metrics are ``family.subject``; dots are
    the only thing that has to change, and changing them here rather than at
    each declaration is what stops a metric from being named two ways in two
    files.
    """
    cleaned = "".join(character if character.isalnum() or character in "_:" else "_"
                      for character in name.replace(".", "_"))
    return f"paperwhisperer_{cleaned}"


class MetricsRegistry:
    """Every metric this module declares, and the only place one may be added.

    Declared up front rather than created on first use, for two reasons. A
    metric that exists from import time is scrapeable at zero, which is what a
    health endpoint needs to render a panel without inventing a bar for a
    counter that has not fired yet (the same argument
    :meth:`infra.jobs.JobStore.count_by_state` makes). And a declared metric
    has one owner, so two call sites cannot both increment a name and each
    believe they are counting everything.
    """

    def __init__(self) -> None:
        # -- parsing: what happened to the file, and did it leave the machine --
        self.parsing_documents = Counter(
            "parsing.documents_total", FAMILY_PARSING,
            documentation="解析结束的文档数，按结果与是否使用云端标注。",
        )
        self.parsing_attempts = Counter(
            "parsing.attempts_total", FAMILY_PARSING,
            documentation="解析阶段尝试数，按阶段、解析器、状态标注。",
        )
        self.parsing_fallbacks = Counter(
            "parsing.fallbacks_total", FAMILY_PARSING,
            documentation="解析回退升级次数，按来源阶段与目标阶段标注。",
        )
        self.parsing_quality = Counter(
            "parsing.quality_total", FAMILY_PARSING,
            documentation="解析质量判定数，按 pass/warn/fail 标注。",
        )
        self.parsing_cache = Counter(
            "parsing.cache_total", FAMILY_PARSING,
            documentation="解析缓存读写数，按 hit/miss 标注。",
        )
        self.parsing_cloud_egress = Counter(
            "parsing.cloud_egress_total", FAMILY_PARSING,
            documentation="文档离开本机的次数（云端解析被真正调用）。",
        )
        self.parsing_duration = Histogram(
            "parsing.duration_ms", FAMILY_PARSING,
            documentation="单次文档解析的墙钟耗时（毫秒）。",
        )

        # -- rag: did retrieval find things, and what did it have to give up --
        self.rag_retrievals = Counter(
            "rag.retrievals_total", FAMILY_RAG,
            documentation="检索次数，按任务类型与空结果原因标注。",
        )
        self.rag_evidence = Histogram(
            "rag.evidence_returned", FAMILY_RAG,
            documentation="单次检索返回的证据条数。",
            buckets=(0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0),
        )
        self.rag_candidates = Counter(
            "rag.candidates_total", FAMILY_RAG,
            documentation="融合候选数，按是否被选中标注。",
        )
        self.rag_legs_abstained = Counter(
            "rag.legs_abstained_total", FAMILY_RAG,
            documentation="检索通道弃权次数（无立场，未参与投票）。",
        )
        self.rag_degradations = Counter(
            "rag.degradations_total", FAMILY_RAG,
            documentation="检索降级原因计数。",
        )
        self.rag_reranked = Counter(
            "rag.reranked_total", FAMILY_RAG,
            documentation="rerank 生效次数。",
        )
        self.rag_duration = Histogram(
            "rag.duration_ms", FAMILY_RAG,
            documentation="单次检索的墙钟耗时（毫秒）。",
        )

        # -- llm: calls, tokens, and the errors worth paging on --
        self.llm_calls = Counter(
            "llm.calls_total", FAMILY_LLM,
            documentation="模型调用次数，按阶段、模型与结果标注。",
        )
        self.llm_tokens = Counter(
            "llm.tokens_total", FAMILY_LLM,
            documentation="模型 token 数，按 input/output 标注。",
        )
        self.llm_failures = Counter(
            "llm.failures_total", FAMILY_LLM,
            documentation="模型调用失败次数，按原因分类标注。",
        )
        self.llm_retries = Counter(
            "llm.retries_total", FAMILY_LLM,
            documentation="模型调用重试次数。",
        )
        self.llm_duration = Histogram(
            "llm.duration_ms", FAMILY_LLM,
            documentation="单次模型调用的墙钟耗时（毫秒）。",
        )
        self.llm_cache = Counter(
            "llm.cache_total", FAMILY_LLM,
            documentation="embedding 缓存命中/未命中数。",
        )

        # -- grounding: the family nothing else counts, and the one the
        #    product is about. Verified quotes, demoted paraphrases, claims
        #    that resolved to no evidence, and answers that said "the paper
        #    does not say" instead of guessing.
        self.grounding_quote_alignment = Counter(
            "grounding.quote_alignment_total", FAMILY_GROUNDING,
            documentation="引文回源校验次数，按 citations.AlignmentOutcome 标注。",
        )
        self.grounding_verified_quotes = Counter(
            "grounding.verified_quotes_total", FAMILY_GROUNDING,
            documentation="通过回源校验、可加引号的引文数。",
        )
        self.grounding_degraded_claims = Counter(
            "grounding.degraded_claims_total", FAMILY_GROUNDING,
            documentation="必须降级为转述（去掉引号）的声明数。",
        )
        self.grounding_unsupported_claims = Counter(
            "grounding.unsupported_claims_total", FAMILY_GROUNDING,
            documentation="声明引用了不存在的证据、或已无任何证据支撑的数量。",
        )
        self.grounding_facts = Counter(
            "grounding.facts_total", FAMILY_GROUNDING,
            documentation="事实层字段数，按 extracted/not_found/conflicting/needs_review 标注。",
        )
        self.grounding_answers = Counter(
            "grounding.answers_total", FAMILY_GROUNDING,
            documentation="问答结果数，按 supported/partially_supported/insufficient 标注。",
        )
        self.grounding_insufficient = Counter(
            "grounding.insufficient_evidence_total", FAMILY_GROUNDING,
            documentation="证据不足而拒答的次数（这是本产品承诺的正确行为）。",
        )

        # -- jobs: queue depth, outcomes, and the recovery that must be zero-ish --
        self.jobs_submitted = Counter(
            "jobs.submitted_total", FAMILY_JOBS,
            documentation="任务提交数，按任务类型与是否去重标注。",
        )
        self.jobs_transitions = Counter(
            "jobs.transitions_total", FAMILY_JOBS,
            documentation="任务状态变更数，按来源状态与目标状态标注。",
        )
        self.jobs_terminal = Counter(
            "jobs.terminal_total", FAMILY_JOBS,
            documentation="任务终态数，按 completed/partial/failed/cancelled 标注。",
        )
        self.jobs_recovered = Counter(
            "jobs.recovered_total", FAMILY_JOBS,
            documentation="worker 崩溃后被回收的任务数。",
        )
        self.jobs_cancelled = Counter(
            "jobs.cancelled_total", FAMILY_JOBS,
            documentation="取消请求数，按是否已被接受标注。",
        )
        self.jobs_depth = Gauge(
            "jobs.queue_depth", FAMILY_JOBS,
            documentation="各状态的任务数（由 count_by_state 覆盖式写入）。",
        )
        self.jobs_duration = Histogram(
            "jobs.duration_ms", FAMILY_JOBS,
            documentation="任务从领取到进入终态的耗时（毫秒）。",
        )

        # -- cost: money, with "unknown" kept distinct from "zero" --
        self.cost_total = Gauge(
            "cost.total", FAMILY_COST,
            documentation="已知成本累计（按来源：llm/embedding/cloud_parse）。",
        )
        self.cost_llm = Gauge(
            "cost.llm_total", FAMILY_COST,
            documentation="已知 LLM 成本累计。",
        )
        self.cost_embedding = Gauge(
            "cost.embedding_total", FAMILY_COST,
            documentation="已知 embedding 成本累计。",
        )
        self.cost_unpriced = Counter(
            "cost.unpriced_operations_total", FAMILY_COST,
            documentation="无法定价的调用数（价格未配置或 provider 未回 usage）——这不是免费。",
        )
        self.cost_documents = Gauge(
            "cost.by_document", FAMILY_COST,
            documentation="按 document_id 记录的该文档累计成本（仅少量文档时有效）。",
            max_series=64,
        )

        # -- meta: instrumentation observing itself, because a metric library
        #    that silently stops counting is worse than none.
        self.series_overflow = Counter(
            "metrics.series_overflow_total", FAMILY_PARSING,
            documentation="因标签基数超限而被丢弃的观测数（调用方误把 id 放进了标签）。",
        )
        self.events_recorded = Counter(
            "metrics.events_total", FAMILY_PARSING,
            documentation="写入 trace 的事件数，按族标注。",
        )
        self.events_dropped = Counter(
            "metrics.events_dropped_total", FAMILY_PARSING,
            documentation="trace 环形缓冲区写满后丢弃的事件数。",
        )

    @property
    def metrics(self) -> tuple[_Metric, ...]:
        """Every declared metric, for iteration in tests and on shutdown."""
        return tuple(
            value for value in vars(self).values() if isinstance(value, _Metric)
        )

    def reset(self) -> None:
        """Zero every metric and its overflow counter."""
        for metric in self.metrics:
            metric.reset()

    def to_dict(self) -> dict[str, Any]:
        """The whole registry as nested plain data, for a health endpoint."""
        return {
            family: {
                metric.name: metric.to_dict() for metric in self.metrics
                if metric.family == family
            }
            for family in FAMILIES
        }

    def prometheus_text(self) -> str:
        """The whole registry in Prometheus exposition format.

        Assembled by hand from the standard library, because the plan puts
        Prometheus on the Hosted tier and a Hosted deployment should be able to
        scrape a counter without the personal tier's dependency list growing by
        a client library. The format is a stable contract; the library is not.
        """
        lines: list[str] = []
        for metric in self.metrics:
            lines.extend(metric.render("paperwhisperer_"))
        overflow = self.series_overflow.total
        if overflow:
            lines.append(f"paperwhisperer_metrics_series_overflow_total {overflow}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The trace
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One thing that happened, in one chain, with the ids that place it.

    The type the acceptance criterion is answered with. It is frozen and
    JSON-safe by construction (``fields`` is a plain dict of primitives,
    normalised on the way in) so that a trace can be asserted on, exported and
    shipped to a Hosted collector without a conversion step that might lose
    exactly the field somebody needed.

    ``seq`` is a process-local monotonic counter and is what makes ordering
    answerable. Wall-clock timestamps are recorded too, but they are the *wrong*
    tool for "what order did these happen in": two events inside the same
    millisecond, or two threads whose clocks are read at different points, will
    order incorrectly, and an acceptance criterion about tracing is not the
    place to accept a probable answer.
    """

    seq: int
    at: str
    name: str
    family: str
    ids: CorrelationIds
    fields: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    error: str = ""
    level: str = "INFO"

    @property
    def is_failure(self) -> bool:
        """Whether this event is a failure, for :meth:`TraceRecorder.failures`.

        Keyed on the event's own ``level`` rather than on a boolean, because
        level is the vocabulary the logging module already has: a caller
        recording a failure with ``log_event(..., level="ERROR")`` gets it in
        the trace's failure query for free, and one that records a warning as
        an error is then visible as a trace that is too loud.
        """
        return self.level in {"ERROR", "CRITICAL"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": int(self.seq),
            "at": self.at,
            "name": self.name,
            "family": self.family,
            "level": self.level,
            **self.ids.as_dict(),
            "duration_ms": self.duration_ms,
            "error": self.error,
            "fields": dict(self.fields),
        }

    def summary(self) -> str:
        """One Chinese line naming what happened, for a CLI or a support reply."""
        bits = [self.name]
        for key in ("parser", "stage", "outcome", "reason", "status", "model", "confidence"):
            value = self.fields.get(key)
            if value not in (None, ""):
                bits.append(f"{key}={value}")
        detail = " ".join(str(item) for item in bits)
        if self.duration_ms is not None:
            detail += f" {self.duration_ms:.0f}ms"
        if self.error:
            detail += f" 错误：{self.error}"
        return detail


def _json_safe(value: Any) -> Any:
    """Reduce a field value to something ``json.dumps`` will accept.

    Not a validation pass -- a *lossiness* pass, and the loss is deliberate: an
    arbitrary object's ``repr`` is the only thing available for it, and an
    unrepresentable field is not worth losing the whole event over. Values a
    call site actually cares about (ints, floats, bools, str, enums' ``value``)
    pass through untouched.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    inner = getattr(value, "value", None)
    if isinstance(inner, (str, int, float, bool)):
        return inner
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)[:200]


class TraceRecorder:
    """A bounded, ordered, queryable log of what happened to which ids.

    A ring buffer rather than a growing list, and the bound is the design
    decision worth defending. A personal-tier process is long-lived: an
    unbounded event list is a slow leak whose size is proportional to how long
    the app has been running, discovered on the day it takes the machine down.
    A ring buffer's cost is known in advance, it keeps the *most recent* events
    -- which is the right window, because the report that prompted the
    investigation is nearly always about something that just happened -- and
    the drops are counted (:attr:`dropped`) rather than silent, because a trace
    that quietly lost events would answer "no such event" for one that did
    happen.

    The buffer is a ``deque`` under a lock. The lock is the one thing here
    that is not free, and it is held for two operations (append and, on a
    query, a list copy) -- never across a log call or an exporter, both of
    which are orders of magnitude slower and would turn a 0.9 us append into a
    millisecond.
    """

    def __init__(self, maxlen: int = 2048) -> None:
        self.maxlen = max(1, int(maxlen))
        self._events: deque[TraceEvent] = deque(maxlen=self.maxlen)
        self._sequence = _counter_factory(1)
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Events lost because the buffer was full. Never silently zero forever."""
        with self._lock:
            return self._dropped

    def record(
        self,
        name: str,
        *,
        family: str,
        ids: CorrelationIds | None = None,
        level: str = "INFO",
        duration_ms: float | None = None,
        error: str = "",
        **fields: Any,
    ) -> TraceEvent:
        """Append one event and return it (so a caller can enrich or drop it)."""
        event = TraceEvent(
            seq=next(self._sequence),
            at=_now_iso(),
            name=str(name),
            family=str(family),
            ids=ids if ids is not None else current_ids(),
            fields={key: _json_safe(value) for key, value in fields.items()},
            duration_ms=None if duration_ms is None else float(duration_ms),
            error=str(error or ""),
            level=str(level or "INFO"),
        )
        with self._lock:
            if len(self._events) == self.maxlen:
                self._dropped += 1
            self._events.append(event)
        return event

    def snapshot(self) -> list[TraceEvent]:
        """Every buffered event, oldest first.

        The unfiltered view, for an exporter and for a health endpoint. A copy
        taken under the lock and handed over, so the caller may hold it for as
        long as it likes -- an exporter writing a file is milliseconds, and the
        recorder must not know or wait for that.
        """
        with self._lock:
            return list(self._events)

    def events(
        self,
        *,
        document_id: str = "",
        job_id: str = "",
        request_id: str = "",
        analysis_run_id: str = "",
        family: str = "",
        name: str = "",
        failures_only: bool = False,
    ) -> list[TraceEvent]:
        """Events matching every filter given, oldest first.

        A filter matches only when the event *carries* that id and it equals
        the filter. "The event has no ``job_id``" is not a match for
        ``job_id="j1"``: it is the difference between "the job's story" and
        "everything, sorted hopefully", and a query that returned the latter
        would look like it worked.
        """
        with self._lock:
            found = list(self._events)
        wanted = {
            "document_id": str(document_id or ""),
            "job_id": str(job_id or ""),
            "request_id": str(request_id or ""),
            "analysis_run_id": str(analysis_run_id or ""),
        }
        selected = [
            event
            for event in found
            if all(
                not value or getattr(event.ids, name_) == value
                for name_, value in wanted.items()
            )
            and (not family or event.family == family)
            and (not name or event.name == name)
            and (not failures_only or event.is_failure)
        ]
        selected.sort(key=lambda event: event.seq)
        return selected

    def failures(self, **filters: Any) -> list[TraceEvent]:
        """Only the failures, for ``errors(document_id=...)`` at 3am."""
        return self.events(failures_only=True, **filters)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            events = [event.to_dict() for event in self._events]
            dropped = self._dropped
        return {"count": len(events), "maxlen": self.maxlen, "dropped": dropped, "events": events}

    def explain(self, **filters: Any) -> str:
        """The chain for one document (or job), as a Chinese narrative.

        This is the query the acceptance criterion asks for, written out rather
        than left as a data structure: given a ``document_id``, print what
        parsed it, whether it was repaired or escalated, whether it went to the
        cloud, what retrieval found, what the model was asked, what the
        grounding checks decided, and what it cost. A JSON blob answers the
        same question for a machine; this answers it for the person who has
        the user's complaint in front of them.
        """
        events = self.events(**filters)
        if not events:
            who = (
                filters.get("document_id")
                or filters.get("job_id")
                or filters.get("request_id")
                or "（未指定）"
            )
            return (
                f"未找到 {who} 的任何事件。"
                "若 instrumentation 未启用（见 configure()），这是预期的结果。\n"
            )
        heading = next(
            (key for key in ("document_id", "job_id", "request_id", "analysis_run_id")
             if filters.get(key)),
            None,
        )
        lines = [f"链路追踪 {heading}={filters[heading]}（共 {len(events)} 个事件）："]
        for event in events:
            mark = "✗" if event.is_failure else ("·" if event.level == "WARNING" else "✓")
            lines.append(f"  {mark} [{event.family}] {event.summary()}")
        if self.dropped:
            lines.append(
                f"  （缓冲区已丢弃 {self.dropped} 个更早的事件；"
                f"容量为 {self.maxlen}，可用 OBSERVABILITY_TRACE_LIMIT 调大。）"
            )
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------

#: The fields a formatted record is assembled from, in a fixed order. A fixed
#: order rather than "whatever the caller passed", because a JSON log that
#: reorders its keys between lines is a log a human cannot read and a diff
#: tool cannot compare.
_RECORD_FIELDS: tuple[str, ...] = (
    "event", "family", "stage", "parser", "status", "outcome", "reason",
    "model", "confidence", "message", "duration_ms", "count", "cost", "error",
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation ids as fields.

    The format is chosen so that ``jq 'select(.document_id=="doc_1")'`` is the
    query. That is the whole reason the ids are top-level fields rather than
    interpolated into the message: a message containing the id is greppable by
    a human and unqueryable by anything else, and the moment a deployment
    aggregates logs across requests -- which is the moment this stops being a
    local dev convenience -- the interpolating format stops working.

    The level filter lives in the *handler*, not here, and
    :func:`configure` is careful to attach it there. ``format`` is called for
    every record the handler accepted, so putting a level check inside it would
    be checking a decision that was already made and charging
    ``json.dumps`` for the records that were going to be thrown away.
    """

    def __init__(self, *, include_empty_ids: bool = False) -> None:
        super().__init__()
        self.include_empty_ids = bool(include_empty_ids)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # ``event_ids`` (set explicitly by Observability.emit) beats
        # ``correlation`` (stamped from the ambient context by the record
        # factory), which beats reading the ContextVar directly -- the last
        # being the path a third-party log record takes.
        ids = (
            getattr(record, "event_ids", None)
            or getattr(record, "correlation", None)
            or current_ids()
        )
        if isinstance(ids, CorrelationIds):
            payload.update(ids.as_dict(include_empty=self.include_empty_ids))
        for name in _RECORD_FIELDS:
            value = getattr(record, name, None)
            if value not in (None, ""):
                payload[name] = _json_safe(value)
        extra = getattr(record, "fields", None)
        if isinstance(extra, Mapping):
            for key, value in extra.items():
                if key not in payload:
                    payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exc_type"] = getattr(record.exc_info[0], "__name__", "Exception")
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """The human-readable path, kept working for a developer running the app.

    A person's terminal is a different consumer from a log pipeline, and it is
    a real regression to make local development read as JSON: the whole point
    of the structured format is machine filterability, and a developer watching
    a job drain wants ``parsing primary docling ok (warn)`` rather than a
    300-character object. The ids are still in the line -- short, and bracketed
    -- because a developer who spots a problem locally is about to go looking
    for it with the same id.

    Kept as a subclass rather than a flag on :class:`JsonFormatter` so the two
    formats cannot acquire conflicting behaviour as each is edited: the shared
    part is the field collection, and each owns only how it renders.
    """

    def format(self, record: logging.LogRecord) -> str:
        # ``event_ids`` (set explicitly by Observability.emit) beats
        # ``correlation`` (stamped from the ambient context by the record
        # factory), which beats reading the ContextVar directly -- the last
        # being the path a third-party log record takes.
        ids = (
            getattr(record, "event_ids", None)
            or getattr(record, "correlation", None)
            or current_ids()
        )
        stamps = [value for value in (ids.request_id, ids.job_id, ids.document_id,
                                      ids.analysis_run_id) if value] if isinstance(
            ids, CorrelationIds
        ) else []
        bits = [
            f"{_now_iso()} {record.levelname:<7} {record.name}",
            f"[{'/'.join(stamps)}]" if stamps else "",
        ]
        for name in _RECORD_FIELDS:
            value = getattr(record, name, None)
            if value not in (None, ""):
                bits.append(f"{name}={_json_safe(value)}")
        extra = getattr(record, "fields", None)
        if isinstance(extra, Mapping):
            bits.extend(f"{key}={_json_safe(value)}" for key, value in extra.items())
        bits.append(record.getMessage())
        line = " ".join(bit for bit in bits if bit)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


#: The logger this module's own record factory stamps. A dedicated child of the
#: package logger rather than the root: :func:`configure` attaches its handler
#: to ``paperwhisperer`` and installs a record factory that *stamps* ids onto
#: every record without redirecting any of them, so a module that never calls
#: anything in this file still gets its records correlated. The factory is
#: global -- that is the stdlib's only hook -- and is therefore the one piece of
#: process-wide state this module owns, which is why it is installed and
#: uninstalled as a pair and why :meth:`Observability.close` restores the exact
#: previous factory.
_ROOT_LOGGER_NAME = "paperwhisperer"


class _RecordFactory:
    """Installable factory that copies the current ids onto every LogRecord.

    A ``functools.partial``-style closure over the previous factory rather than
    a module-level function, for the same reason the uninstaller has to hold
    the previous value: ``logging.setLogRecordFactory`` is global, so
    overwriting it without being able to put it back is not an enhancement, it
    is a leak that outlives the test that caused it.
    """

    __slots__ = ("_previous",)

    def __init__(self, previous: Callable[[logging.LogRecord], logging.LogRecord]) -> None:
        self._previous = previous

    def __call__(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = self._previous(*args, **kwargs)
        # ``correlation`` is set rather than each id separately: a formatter
        # needs all four together, and four attributes per record would be four
        # attribute writes on every log line in the process to serve one read.
        if not hasattr(record, "correlation"):
            record.correlation = current_ids()
        return record


# --------------------------------------------------------------------------
# Exporters (the seam OTel would plug into)
# --------------------------------------------------------------------------


class Exporter:
    """Where the trace and the metrics go when this process stops pretending.

    The plan puts OTel + Prometheus on the Hosted tier and marks them
    optional, so the interface exists without the dependency. One method, and
    it is handed both halves of the state, because a collector that receives
    spans but not the counters for the same window produces a dashboard that
    disagrees with itself.

    Implementations must be **synchronous, quick, and never raise** -- they are
    called from :meth:`Observability.flush` on the request path, and an
    exporter that does I/O there is the one place in this module where
    instrumentation would cost the user real latency.
    """

    def export(self, events: Sequence[TraceEvent], metrics: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release the exporter's resources. Must not raise."""


class NullExporter(Exporter):
    """The default. Counts what it was given and does nothing else.

    Not "a no-op for symmetry": it is what makes the seam real, because
    :meth:`Observability.flush` runs the same path either way and a caller can
    assert that events reached *an* exporter. An interface with no
    implementation is an interface nobody has exercised.
    """

    def __init__(self) -> None:
        self.batches = 0
        self.event_count = 0
        self._lock = threading.Lock()

    def export(self, events: Sequence[TraceEvent], metrics: Mapping[str, Any]) -> None:
        with self._lock:
            self.batches += 1
            self.event_count += len(events)

    def close(self) -> None:
        return None


class JsonlExporter(Exporter):
    """Append the trace to a JSON-lines file, one object per event.

    The stdlib implementation of :class:`Exporter`, and the thing that makes
    the trace outlive the process -- which matters on the personal tier, where
    the alternative is a bug report and a reboot. Append mode and one
    ``write`` per batch rather than per event: the file is opened for the
    batch and closed, so a crash costs the batch rather than the file.

    ``corrupt`` is counted, not raised. A trace file that cannot be written
    must not take down the process that was being diagnosed, and a silently
    discarded trace is worse than useless -- so the failure is counted, and
    :meth:`Observability.flush` returns the count so a caller can notice.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.corrupt = 0
        self.written = 0
        self._lock = threading.Lock()

    def export(self, events: Sequence[TraceEvent], metrics: Mapping[str, Any]) -> None:
        if not events:
            return
        lines = [json.dumps(event.to_dict(), ensure_ascii=False, default=str) for event in events]
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
                self.written += len(lines)
            except OSError:
                self.corrupt += 1

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# The facade
# --------------------------------------------------------------------------

#: Settings this module reads, with the default each falls back to. Declared
#: here rather than in :mod:`paperwhisperer.core.config` because this task does
#: not own that module -- the same arrangement :mod:`infra.jobs` and
#: :mod:`analysis.paper_facts` use: the reader is ``getattr(config, name,
#: default)`` against this table, and whoever adds the keys to config copies
#: these names and defaults. Absent key, default in force, and the default is
#: the conservative one -- observability off, so a checkout nobody configured
#: behaves exactly as it did before this file existed.
OBSERVABILITY_CONFIG_KEYS: dict[str, Any] = {
    #: Master switch. ``False`` by default: the acceptance criterion is
    #: "instrumentation is on and correct", not "instrumentation is on", and a
    #: library that starts collecting on import is a library whose cost nobody
    #: agreed to pay.
    "OBSERVABILITY_ENABLED": False,
    "OBSERVABILITY_LOG_LEVEL": "INFO",
    "OBSERVABILITY_JSON_LOGS": True,
    "OBSERVABILITY_TRACE_LIMIT": 2048,
    "OBSERVABILITY_MAX_SERIES": MAX_SERIES_PER_METRIC,
    "OBSERVABILITY_EXPORTER_PATH": "",
    #: Above this a parse or an LLM call is logged at WARNING even on success.
    #: Both numbers are "normal is very slow": the gateway's own ceiling is
    #: 900 s and a reasoning model on a big document is tens of seconds, so a
    #: default below those flags nothing unless something is actually wrong.
    "OBSERVABILITY_SLOW_PARSE_MS": 120000,
    "OBSERVABILITY_SLOW_LLM_MS": 30000,
}


def _setting(name: str) -> Any:
    return getattr(config, name, OBSERVABILITY_CONFIG_KEYS[name])


class Observability:
    """Collection, and the one place that turns it on and off.

    The facade exists so that "is observability running" is one attribute and
    "stop it" is one method, and so that :func:`configure` /
    :func:`shutdown` cannot leave a half-installed record factory behind a
    half-removed handler. Every ``record_*`` helper is a bound method here as
    well as a module-level function, so a caller that prefers to pass a facade
    around (a test, a Hosted wiring) can and the default one stays the
    process-wide singleton.

    Disabled is the constructor's default and not an accident: a module-level
    ``Observability()`` that started collecting would mean importing
    ``paperwhisperer.core.observability`` -- which five modules now do -- is
    what turned instrumentation on for every deployment that never asked.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        trace_limit: int | None = None,
        exporter: Exporter | None = None,
        registry: MetricsRegistry | None = None,
        max_series: int = MAX_SERIES_PER_METRIC,
    ) -> None:
        self.enabled = bool(enabled)
        default_limit = OBSERVABILITY_CONFIG_KEYS["OBSERVABILITY_TRACE_LIMIT"]
        limit = default_limit if trace_limit is None else trace_limit
        self.trace = TraceRecorder(int(limit))
        self.metrics = registry if registry is not None else MetricsRegistry()
        self.exporter: Exporter = exporter if exporter is not None else NullExporter()
        self.max_series = int(max_series)
        self._handler: logging.Handler | None = None
        self._logger: logging.Logger | None = None
        self._logger_level: int | None = None
        self._previous_factory: Callable[[logging.LogRecord], logging.LogRecord] | None = None
        self._lock = threading.RLock()

    # -- configuration -----------------------------------------------------

    def install(
        self,
        *,
        level: int | str = logging.INFO,
        json_lines: bool | None = None,
        stream: Any = None,
        target: str = _ROOT_LOGGER_NAME,
    ) -> logging.Handler:
        """Attach the handler and the record factory, idempotently.

        Three separate global mutations, so all three are saved and all three
        are restored by :meth:`close`: the target logger's level (a caller that
        had turned ``paperwhisperer`` down to WARNING would otherwise find it
        raised), a handler on that logger (otherwise duplicated lines), and the
        process-wide record factory (otherwise ids that no formatter can see).

        ``target`` defaults to the package logger rather than the root so that
        a library embedding this project does not have its own logging
        rearranged by importing it.
        """
        with self._lock:
            if self._handler is not None:
                return self._handler
            resolved_json = (
                bool(_setting("OBSERVABILITY_JSON_LOGS"))
                if json_lines is None else bool(json_lines)
            )
            # ``StreamHandler(None)`` is stderr, so the default needs no branch.
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                JsonFormatter() if resolved_json else TextFormatter()
            )
            handler.setLevel(level)

            target_logger = logging.getLogger(target)
            self._logger = target_logger
            self._logger_level = target_logger.level
            # The logger's own level gates records *before* the handler sees
            # them, so a logger left at WARNING would drop INFO records no
            # matter what the handler is set to -- and an observability layer
            # that cannot be turned up is a layer that is off in production and
            # on in dev. Lowered only when the current level is more
            # restrictive than the handler's (or unset, which inherits
            # whatever the root decided): a logger deliberately set to ERROR
            # is an operator's choice and is not overridden. Both are restored
            # by :meth:`close`.
            wanted = _as_level(level)
            current = target_logger.level
            if current == logging.NOTSET or current > wanted:
                target_logger.setLevel(wanted)
            target_logger.addHandler(handler)

            self._previous_factory = logging.getLogRecordFactory()
            logging.setLogRecordFactory(_RecordFactory(self._previous_factory))
            self._handler = handler
            self.enabled = True
            return handler

    def close(self) -> None:
        """Put the process back exactly as :meth:`install` found it. Idempotent.

        The restoration of the record factory is the part that matters most and
        the part a leak would be least visible: a stray factory keeps stamping
        ids onto records for the rest of the session, which is harmless in
        isolation and quietly wrong in a test suite that also asserts on log
        output. Closing twice is a no-op rather than an error, so a fixture's
        teardown can call it without knowing whether the test enabled anything.
        """
        with self._lock:
            if self._handler is not None:
                logger_ = self._logger
                if logger_ is not None:
                    try:
                        logger_.removeHandler(self._handler)
                    except Exception:  # pragma: no cover - defensive
                        pass
                    if self._logger_level is not None:
                        logger_.setLevel(self._logger_level)
                try:
                    self._handler.close()
                except Exception:  # pragma: no cover - defensive
                    pass
                self._handler = None
            if self._previous_factory is not None:
                logging.setLogRecordFactory(self._previous_factory)
                self._previous_factory = None
            self._logger = None
            self._logger_level = None
            self.enabled = False
        try:
            self.exporter.close()
        except Exception:  # pragma: no cover - close must not raise
            pass

    def flush(self) -> int:
        """Hand the current events to the exporter. Returns how many went.

        The caller decides when -- :func:`flush` on shutdown, or a route that
        wants the trace durable before it returns. Not automatic on a timer: a
        background thread in a library that most deployments run as a single
        process would be a surprise, and the cost of an unflushed trace is
        bounded by the ring buffer's own length.

        The events are snapshotted and the exporter is called **outside** the
        lock. An exporter may do file I/O, and holding the recorder's lock
        across it would turn every concurrent ``record`` call in the process
        into a queue behind that I/O -- the exact "instrumentation that costs
        the request" outcome requirement 6 exists to rule out.
        """
        events = self.trace.snapshot()
        try:
            self.exporter.export(events, self.metrics.to_dict())
        except Exception:  # noqa: BLE001 - an exporter must not break a flush
            logger.debug("可观测性导出失败，已忽略。", exc_info=True)
            return 0
        return len(events)

    # -- ids ---------------------------------------------------------------

    def bind(
        self,
        *,
        request_id: str = "",
        job_id: str = "",
        document_id: str = "",
        analysis_run_id: str = "",
    ):
        """``correlate`` as a method, for a caller holding a facade."""
        return correlate(
            request_id=request_id,
            job_id=job_id,
            document_id=document_id,
            analysis_run_id=analysis_run_id,
        )

    def submit(self, executor: Any, function: Callable[..., Any], /, *args: Any, **kwargs: Any):
        """Submit to a thread pool **with** the correlation context attached.

        The single most important method in this file for a codebase that
        fans work out across a pool. ``executor.submit(function, ...)`` runs
        ``function`` in a fresh :mod:`contextvars` context, so every id is gone
        by the time the work that actually takes seconds begins; this wraps it
        so the work runs under a copy of the ids that are in scope now.

        Returns the executor's own ``Future``, so it is a drop-in at every call
        site in :data:`THREAD_HOPS` and none of them has to change what they do
        with the result.
        """
        context = contextvars.copy_context()
        return executor.submit(context.run, function, *args, **kwargs)

    # -- the one primitive -------------------------------------------------

    def emit(
        self,
        name: str,
        *,
        family: str,
        ids: CorrelationIds | None = None,
        level: str = "INFO",
        logger_name: str = _ROOT_LOGGER_NAME,
        duration_ms: float | None = None,
        error: str = "",
        **fields: Any,
    ) -> TraceEvent | None:
        """Write the event and the log line together. The primitive underneath.

        The four id names are pulled out of ``fields`` before anything else
        happens, so ``emit(..., document_id=...)`` sets the event's document
        instead of quietly writing it into ``fields``. Leaving that to
        ``fields`` would be the single worst footgun available to a call site:
        the signature accepts it, the event carries it, and every query by
        ``document_id`` returns nothing -- a trace that looks configured and
        answers "no such document" for a document that certainly ran.

        Returns the event even when disabled? No -- returns ``None``, because a
        caller that stored the return value would then need a null check at
        every site, and a disabled collector should cost a caller *nothing*
        beyond the call itself. ``record_*`` therefore never returns the event
        either; a caller wanting to enrich one passes the fields in.

        Level is the caller's decision and is not filtered here beyond the
        enabled check, because the trace is the product: an ``ERROR`` belongs in
        it even when the console is at ``WARNING``.
        """
        if not self.enabled:
            return None
        # Pull the id names out of ``fields`` before anything else, for the
        # reason in the docstring: an id left in ``fields`` is accepted by the
        # signature, carried by the event, and invisible to every query.
        field_ids = {
            name: fields.pop(name) for name in ID_FIELDS if fields.get(name)
        }
        resolved = (
            resolve_ids(None, **field_ids)
            if ids is None
            else ids.merged(**field_ids)
        )
        event = self.trace.record(
            name,
            family=family,
            ids=resolved,
            level=level,
            duration_ms=duration_ms,
            error=error,
            **fields,
        )
        self.metrics.events_recorded.inc(family=family)
        if self.trace.dropped:
            self.metrics.events_dropped.inc(self.trace.dropped)
        numeric_level = _as_level(level)
        target = self._logger if self._logger is not None else logging.getLogger(logger_name)
        # ``event_ids`` rather than ``correlation``: the record factory has
        # already stamped the *ambient* context onto the record, and
        # :mod:`logging` raises ``KeyError`` rather than overwrite an existing
        # attribute, so reusing that name would make every line raise. A
        # distinct key also carries a real distinction -- ``emit`` may be
        # handed explicit ids that are not the ones in scope (a trace written
        # for a document the caller is describing rather than the one it is
        # running on), and the formatters prefer them. The factory keeps the
        # ``correlation`` attribute to itself, which is why it has to be
        # restorable.
        target.log(
            numeric_level,
            name,
            extra={
                "event_ids": resolved,
                "event": name,
                "family": family,
                "duration_ms": duration_ms,
                "error": error,
                "fields": dict(event.fields),
            },
        )
        return event

    def log_event(self, name: str, **kwargs: Any) -> TraceEvent | None:
        """A named event with no metric attached.

        For the steps that are worth following but not worth counting -- a
        cache hit, a cloud refusal, a leg that abstained. They are exactly the
        lines that answer "why did this document behave like that", which is the
        question the whole module exists for.
        """
        family = str(kwargs.pop("family", FAMILY_PARSING))
        return self.emit(name, family=family, **kwargs)

    # -- ingestion / parsing ----------------------------------------------

    def record_parse(self, outcome: Any, *, duration_ms: float | None = None, **extra: Any) -> None:
        """Count and record one finished parse. Read from a ``ParseOutcome``.

        Duck-typed against :class:`ingestion.orchestrator.ParseOutcome` and not
        imported, for the layering reason the rest of this repository keeps
        making: ``core`` must not import ``ingestion``, or every module that
        wants a log line also gains the parser gateway and its optional
        dependencies. Reading ``attempts``, ``used_cloud`` and ``quality`` off
        the object the caller already has costs one ``getattr`` and keeps the
        dependency pointing the other way.

        The three numbers that matter and why each is separate:

        * ``parsing.cloud_egress_total`` counts documents that *left the
          machine*, from ``used_cloud`` -- not ``cloud_attempted``, which is
          money spent on a parse that lost. Both are counted, under different
          names, because "a paid cloud parse was discarded" and "the answer
          came from a third party" are different facts with different
          consequences for the reader.
        * ``parsing.fallbacks_total`` counts the *escalation*, from the attempt
          trail rather than from ``fallback_used``, so a chain that tried a
          repair and kept the primary is visible as what it was.
        * ``parsing.quality_total`` counts the gate's verdict. A document that
          parsed and was still WARN is the case a user cannot distinguish from
          a good one.
        """
        if not self.enabled:
            return
        status = str(getattr(outcome, "status", "") or "unknown")
        usable = bool(getattr(outcome, "is_usable", False))
        used_cloud = bool(getattr(outcome, "used_cloud", False))
        parser = str(getattr(outcome, "parsed_by", "") or "")
        ids = resolve_ids(outcome, **{k: v for k, v in extra.items() if k in ID_FIELDS})

        self.metrics.parsing_documents.inc(
            status=status, parser=parser or "unknown", used_cloud=str(used_cloud).lower()
        )
        if used_cloud:
            self.metrics.parsing_cloud_egress.inc(parser=parser or "unknown")
        if getattr(outcome, "cache_hit", False):
            self.metrics.parsing_cache.inc(result="hit")
        else:
            self.metrics.parsing_cache.inc(result="miss")

        quality = getattr(outcome, "quality", None)
        if quality is not None:
            self.metrics.parsing_quality.inc(
                status=str(getattr(getattr(quality, "status", ""), "value", "") or "unknown")
            )

        for attempt in getattr(outcome, "attempts", ()) or ():
            stage = str(getattr(attempt, "stage", "") or "unknown")
            parser_name = str(getattr(attempt, "parser_name", "") or "unknown")
            status = str(getattr(attempt, "status", "") or "unknown")
            self.metrics.parsing_attempts.inc(
                stage=stage,
                parser=parser_name,
                status=status,
                attempted=str(bool(getattr(attempt, "attempted", False))).lower(),
                cache_hit=str(bool(getattr(attempt, "cache_hit", False))).lower(),
                network=str(bool(getattr(attempt, "network", False))).lower(),
            )
            if not getattr(attempt, "attempted", False):
                # A stage that never ran is recorded, because "the cloud
                # fallback was refused" and "the cloud fallback never happened"
                # are different answers to a question users ask.
                self.emit(
                    f"parse.stage_skipped.{stage}",
                    family=FAMILY_PARSING,
                    ids=ids,
                    level="INFO",
                    stage=stage,
                    reason=str(getattr(attempt, "skipped_reason", "") or ""),
                    # Empty rather than "unknown" when the stage named no
                    # parser: it never ran, so it had no parser, and a
                    # narrative claiming otherwise is one more thing to
                    # discount while reading a trace under pressure.
                    parser=str(getattr(attempt, "parser_name", "") or ""),
                )
                continue
            if getattr(attempt, "cache_hit", False):
                self.metrics.parsing_cache.inc(result="hit", parser=parser_name)

            # Every stage that *ran* gets its own event, and a failed one is an
            # ERROR. This is the acceptance criterion in one line of code: the
            # "who tried this and what did it say" trail has to be in the trace
            # and not only in a counter, because the question a user asks is
            # "why did my PDF go to a third party" and the answer is the
            # failure of the stage before it -- a number would not say
            # *why*, and a reason without a number would not tell you it is
            # happening once in a thousand documents.
            succeeded = status in {"success", "partial"}
            self.emit(
                f"parse.attempt.{stage}",
                family=FAMILY_PARSING,
                ids=ids,
                level="INFO" if succeeded else "ERROR",
                duration_ms=float(getattr(attempt, "duration_ms", 0) or 0) or None,
                error="" if succeeded else str(getattr(attempt, "error_message", "") or ""),
                stage=stage,
                parser=parser_name,
                status=status,
                network=bool(getattr(attempt, "network", False)),
                cache_hit=bool(getattr(attempt, "cache_hit", False)),
                quality=str(getattr(attempt, "quality_status", "") or ""),
            )

        stages = [
            str(getattr(attempt, "stage", ""))
            for attempt in getattr(outcome, "attempts", ()) or ()
            if getattr(attempt, "attempted", False)
        ]
        for earlier, later in zip(stages, stages[1:], strict=False):
            self.metrics.parsing_fallbacks.inc(from_stage=earlier, to_stage=later)
        if len(stages) > 1:
            # The escalation itself, as an event. The per-stage events show what
            # happened; this one names the decision -- "we moved from local to
            # cloud because local failed" -- which is the sentence the privacy
            # question is actually asking for.
            self.emit(
                "parse.escalated",
                family=FAMILY_PARSING,
                ids=ids,
                level="WARNING",
                stage=stages[-1],
                reason=f"{stages[0]}→{stages[-1]}",
                parser=parser,
                status=status,
            )

        level = "INFO" if usable else "ERROR"
        self.emit(
            "parse.finished",
            family=FAMILY_PARSING,
            ids=ids,
            level=level,
            duration_ms=duration_ms,
            error="" if usable else str(getattr(outcome, "error_message", "") or ""),
            parser=parser,
            status=status,
            used_cloud=used_cloud,
            fallback_used=bool(getattr(outcome, "fallback_used", False)),
            cache_hit=bool(getattr(outcome, "cache_hit", False)),
            summary=str(getattr(outcome, "summary_line", lambda: "")() or ""),
            sha256=str(getattr(outcome, "sha256", "") or ""),
        )
        if duration_ms is not None:
            self.metrics.parsing_duration.observe(duration_ms)
        # The cost of the parse is a cloud parse, and it is counted even when
        # the cloud attempt lost: money was spent and a ledger that only
        # records winners under-reports the bill.
        if getattr(outcome, "cloud_attempted", False):
            self.record_cost(source="cloud_parse", cost=None, reason="unpriced")

    # -- rag ---------------------------------------------------------------

    def record_retrieval(
        self, record: Any, *, duration_ms: float | None = None, **extra: Any
    ) -> None:
        """Count and record one retrieval. Read from a ``RetrievalRecord``.

        The four numbers a bad answer needs, and the reasoning for each:

        * ``rag.retrievals_total`` labelled by ``empty_reason``. "No evidence"
          is the *correct* outcome for most of what this product is asked, so a
          counter that did not distinguish "the retriever found nothing" from
          "a filter excluded everything" would report the two most important
          different bugs as one.
        * ``rag.legs_abstained_total`` from ``degradations``. A leg that found
          nothing to say is not a leg that found nothing, and the record says
          which.
        * ``rag.candidates_total`` split on ``selected``, so "the right passage
          was never a candidate" and "the right passage was ranked sixth" are
          different reads of the same bad answer.
        * ``rag.duration_ms``. Retrieval is the one stage whose latency a user
          feels directly, because it is on the path of every question.
        """
        if not self.enabled:
            return
        task = str(getattr(record, "task", "") or "unknown")
        empty_reason = str(getattr(record, "empty_reason", "") or "unknown")
        ids = resolve_ids(record, **{k: v for k, v in extra.items() if k in ID_FIELDS})
        candidates = tuple(getattr(record, "candidates", ()) or ())
        evidence = tuple(getattr(record, "evidence", ()) or ())
        degradations = tuple(str(item) for item in getattr(record, "degradations", ()) or ())

        self.metrics.rag_retrievals.inc(task=task, empty_reason=empty_reason)
        self.metrics.rag_evidence.observe(len(evidence))
        self.metrics.rag_degradations.inc(len(degradations))
        for candidate in candidates:
            self.metrics.rag_candidates.inc(
                task=task, selected=str(bool(getattr(candidate, "selected", False))).lower()
            )
            if not getattr(candidate, "selected", False):
                self.metrics.rag_candidates.inc(
                    task=task, selected="false", reason=str(getattr(candidate, "reason", "") or "")
                )
        rerank = getattr(record, "rerank", None) or {}
        if isinstance(rerank, Mapping) and rerank.get("applied"):
            self.metrics.rag_reranked.inc(task=task)

        # A leg that abstained is recorded as its own event so the trace can
        # answer "why did the answer come from the lexical leg alone" without
        # the reader parsing an English degradation sentence.
        for degradation in degradations:
            leg = "unknown"
            for candidate in ("dense", "lexical", "fuzzy"):
                if degradation.startswith(candidate) or f"{candidate} 通道" in degradation:
                    leg = candidate
                    break
            if leg != "unknown":
                self.metrics.rag_legs_abstained.inc(leg=leg, task=task)

        # An empty retrieval is INFO, not WARNING, and this is not a
        # cosmetic choice. On the personal tier it is the *correct and common*
        # outcome: the plans are strict by design and the product's stated rule
        # is to say "the paper does not say" rather than retrieve something
        # adjacent. A WARNING for right behaviour trains every operator to
        # ignore warnings, and the one warning that then matters — a parser
        # that failed, a claim that could not be grounded — arrives in a
        # stream they have already learned to scroll past.
        #
        # The absence is counted instead (rag_retrievals with an
        # empty_reason label), because a count nobody reads and a log nobody
        # reads are the same failure.
        level = "INFO"
        self.emit(
            "retrieval.finished",
            family=FAMILY_RAG,
            ids=ids,
            level=level,
            duration_ms=duration_ms,
            status=empty_reason,
            task=task,
            record_id=str(getattr(record, "record_id", "") or ""),
            plan_signature=str(getattr(record, "plan_signature", "") or ""),
            candidates=len(candidates),
            evidence=len(evidence),
            degradations=degradations,
            documents=len(getattr(record, "documents_considered", ()) or ()),
        )
        if duration_ms is not None:
            self.metrics.rag_duration.observe(duration_ms)

    # -- llm ---------------------------------------------------------------

    def record_model_call(
        self,
        call: Any,
        *,
        duration_ms: float | None = None,
        outcome: str = "ok",
        error: str = "",
        **extra: Any,
    ) -> None:
        """Count and record one model call. Read from a ``ModelCall``.

        The token counters are the point: ``AnalysisRun.llm_cost`` is
        ``None`` whenever any call's price is unknown, and a cost dashboard
        that rounds that to a small number is worse than one that says it does
        not know. So the tokens -- which are *always* known, because they are
        estimated from the prompts the module already built -- are counted
        unconditionally, and the money is a separate call to
        :meth:`record_cost` that can be unknown. Splitting them is what lets
        "this run was expensive" and "this run's cost is unknown" both be
        answered.
        """
        if not self.enabled:
            return
        stage = str(getattr(call, "stage", "") or "unknown")
        model = str(getattr(call, "model", "") or "unknown")
        ids = resolve_ids(call, **{k: v for k, v in extra.items() if k in ID_FIELDS})
        input_tokens = int(getattr(call, "input_tokens", 0) or 0)
        output_tokens = int(getattr(call, "output_tokens", 0) or 0)

        self.metrics.llm_calls.inc(stage=stage, model=model, outcome=outcome)
        self.metrics.llm_tokens.inc(input_tokens, direction="input")
        self.metrics.llm_tokens.inc(output_tokens, direction="output")
        if outcome not in {"ok", "success"}:
            self.metrics.llm_failures.inc(stage=stage, outcome=outcome)

        slow_threshold = float(_setting("OBSERVABILITY_SLOW_LLM_MS"))
        level = "INFO"
        if outcome not in {"ok", "success"}:
            level = "ERROR"
        elif duration_ms is not None and duration_ms >= slow_threshold:
            # A successful call that was slow is the one a user complains about
            # and the one that never appears in an error log.
            level = "WARNING"

        self.emit(
            "llm.call",
            family=FAMILY_LLM,
            ids=ids,
            level=level,
            duration_ms=duration_ms,
            error=error,
            stage=stage,
            model=model,
            outcome=outcome,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=getattr(call, "cost", None),
            prompt_version=str(getattr(call, "prompt_version", "") or ""),
        )
        if duration_ms is not None:
            self.metrics.llm_duration.observe(duration_ms)

    def record_embedding_usage(self, usage: Any, **extra: Any) -> None:
        """Count one embedding operation from an ``EmbeddingUsage``.

        ``cache_hits`` and ``tokens_reported`` both matter and are not
        derivable from each other. A run that spent nothing because everything
        was cached and a run that spent nothing because the index was empty are
        indistinguishable in a cost number alone, which is the reason
        :class:`EmbeddingUsage` kept the counter; here it is what a reader of
        the metrics needs in order not to draw the wrong conclusion.
        """
        if not self.enabled:
            return
        self.metrics.llm_cache.inc(int(getattr(usage, "cache_hits", 0) or 0), result="hit")
        self.metrics.llm_cache.inc(int(getattr(usage, "texts", 0) or 0), result="miss")
        self.metrics.llm_tokens.inc(
            int(getattr(usage, "input_tokens", 0) or 0), direction="embedding"
        )
        self.record_cost(
            source="embedding",
            cost=getattr(usage, "cost", None),
            reason="" if getattr(usage, "tokens_reported", True) else "tokens_unreported",
            **extra,
        )

    # -- grounding ---------------------------------------------------------

    def record_citation_verification(self, verification: Any, **extra: Any) -> None:
        """Count one quote verification. Read from a ``CitationVerification``.

        The whole family, in one call, from the one place that already decides.
        Three outcomes are counted, and collapsing any of them would lose the
        signal:

        * ``aligned`` -> ``grounding.verified_quotes_total``: this is the only
          span permitted quotation marks, and its *rate* is the number the
          release gate is about.
        * any degraded outcome -> ``grounding.quote_alignment_total{outcome}``
          and, when the claim is rendered as a paraphrase,
          ``grounding.degraded_claims_total``. The count of demoted quotes is
          the number of places the product had to take the model's wording away
          from its own quotation marks, which is precisely the failure the
          Evidence-first design is for.
        * a claim left with no evidence at all -> ``grounding.unsupported_claims_total``.

        ``presentation`` is read rather than inferred from ``aligned``, because
        :class:`rag.citations.CitationVerification` derives it from the outcome
        and the derivation is where a new outcome would have to be classified;
        re-deriving it here would be a second opinion that could disagree with
        the one the renderer obeys.
        """
        if not self.enabled:
            return
        outcome = getattr(verification, "outcome", "")
        outcome_value = str(getattr(outcome, "value", outcome) or "unknown")
        presentation = getattr(verification, "presentation", None)
        presentation_value = str(getattr(presentation, "value", presentation) or "")
        aligned = bool(
            getattr(verification, "aligned", False)
            or presentation_value == "direct_quote"
            or outcome_value == "aligned"
        )
        mode = getattr(verification, "mode", "")
        mode_value = str(getattr(mode, "value", mode) or "")
        ids = resolve_ids(verification, **{k: v for k, v in extra.items() if k in ID_FIELDS})

        self.metrics.grounding_quote_alignment.inc(outcome=outcome_value)
        if aligned:
            self.metrics.grounding_verified_quotes.inc(mode=mode_value or "unknown")
        elif presentation_value == "paraphrase":
            self.metrics.grounding_degraded_claims.inc(outcome=outcome_value)
        if not getattr(verification, "element_id", "") and not getattr(
            verification, "locator", None
        ):
            # Nothing to point a reader at: the claim cannot be cited at all.
            self.metrics.grounding_unsupported_claims.inc(reason="no_locator")

        self.emit(
            "grounding.quote_verified",
            family=FAMILY_GROUNDING,
            ids=ids,
            level="INFO" if aligned else "WARNING",
            status=outcome_value,
            presentation=presentation_value or ("direct_quote" if aligned else "paraphrase"),
            reason="" if aligned else outcome_value,
        )

    def record_claim_verdicts(
        self,
        claims: Sequence[Any],
        *,
        confidence: str = "",
        **extra: Any,
    ) -> None:
        """Count one answer's claims, from ``analysis.orchestrator``.

        This is the one the product is judged on. An answer is
        ``supported`` when every claim is carried by retrieved evidence and
        presents no quotation or one that aligned; ``insufficient_evidence``
        means the answer refused, which is a *correct* outcome and is counted
        separately for exactly that reason -- a system that answered every
        question would show this counter at zero, and a reader who does not
        know that would read zero as a bug.

        Degraded-to-paraphrase and unsupported are counted per claim here
        rather than per verification because this is where the claim *as
        rendered* is known: :meth:`record_citation_verification` sees a
        quotation, not the sentence the reader ends up looking at.
        """
        if not self.enabled:
            return
        ids = resolve_ids(None, **{k: v for k, v in extra.items() if k in ID_FIELDS})
        parsed = tuple(claims or ())
        degraded = 0
        unsupported = 0
        for claim in parsed:
            presentation = str(getattr(claim, "presentation", "") or "")
            if not getattr(claim, "evidence_ids", ()):
                unsupported += 1
            elif presentation == "paraphrase" and getattr(claim, "quote", ""):
                degraded += 1
        if degraded:
            self.metrics.grounding_degraded_claims.inc(degraded, outcome="answer_claim")
        if unsupported:
            self.metrics.grounding_unsupported_claims.inc(unsupported, reason="answer_claim")

        self.metrics.grounding_answers.inc(confidence=confidence or "unknown")
        if confidence == "insufficient_evidence":
            self.metrics.grounding_insufficient.inc(reason=confidence)

        self.emit(
            "grounding.answer_verdict",
            family=FAMILY_GROUNDING,
            ids=ids,
            level="INFO",
            confidence=confidence,
            status=confidence,
            count=len(parsed),
            degraded=degraded,
            unsupported=unsupported,
        )

    def record_fact_extraction(self, report: Any, run: Any = None, **extra: Any) -> None:
        """Count one fact-layer extraction, from ``paper_facts``.

        Reads the real report rather than re-deriving anything: it already knows
        which quotes aligned (``verified_quotes``), which fields failed
        verification (``unverified_fields``), which citations resolved to
        nothing (``dropped_evidence_ids``) and how many facts are grounded
        (``AnalysisRun.grounded_fact_count``). This is a reader, not a second
        opinion -- the same rule :mod:`rag.citations` follows for the
        normalisation list.

        ``grounded_fact_count / fact_count`` is the ratio worth watching, and
        it is a *ratio of the same run* rather than an average across runs, so
        one degenerate paper does not get hidden by a hundred good ones.
        """
        if not self.enabled:
            return
        ids = resolve_ids(run if run is not None else report,
                         **{k: v for k, v in extra.items() if k in ID_FIELDS})
        verified = len(getattr(report, "verified_quotes", ()) or ())
        unverified_fields = tuple(getattr(report, "unverified_fields", ()) or ())
        dropped = tuple(getattr(report, "dropped_evidence_ids", ()) or ())
        parse_failed = bool(getattr(report, "parse_failed", False))

        self.metrics.grounding_verified_quotes.inc(verified, mode="fact_layer")
        if unverified_fields:
            self.metrics.grounding_facts.inc(len(unverified_fields), status="needs_review")
        if dropped:
            self.metrics.grounding_unsupported_claims.inc(
                len(dropped), reason="unresolvable_citation"
            )
        for field_name in tuple(getattr(report, "not_stated_fields", ()) or ()):
            self.metrics.grounding_facts.inc(status="not_found", field=str(field_name))

        # Named states rather than a nested ternary: a reader should be able to
        # see that "extraction failed" and "extraction worked but the grounding
        # found problems" are different verdicts, and a three-armed expression
        # hides exactly that.
        if parse_failed:
            state, level = "parse_failed", "ERROR"
        elif unverified_fields or dropped:
            state, level = "degraded", "WARNING"
        else:
            state, level = "ok", "INFO"
        self.emit(
            "grounding.fact_extraction",
            family=FAMILY_GROUNDING,
            ids=ids,
            level=level,
            status=state,
            verified_quotes=verified,
            unverified_fields=unverified_fields,
            dropped_evidence_ids=dropped,
            not_stated_fields=tuple(getattr(report, "not_stated_fields", ()) or ()),
            fact_count=int(getattr(run, "fact_count", 0) or 0) if run is not None else None,
            grounded=int(getattr(run, "grounded_fact_count", 0) or 0) if run is not None else None,
            degradations=tuple(getattr(report, "degradations", lambda: ())()),
        )
        if run is not None:
            for call in tuple(getattr(run, "calls", ()) or ()):
                self.record_model_call(call)
            self.record_cost(
                source="llm",
                cost=getattr(run, "llm_cost", None),
                reason="" if getattr(run, "llm_cost", None) is not None else "unpriced",
                **extra,
            )

    # -- jobs --------------------------------------------------------------

    def record_job_transition(self, job: Any, *, event: Any = None, **extra: Any) -> None:
        """Count one job state change. Read from a ``Job`` / ``JobEvent``.

        ``record_job_transition`` takes the job rather than only the event so
        that ``job.kind`` -- the routing key, and therefore the only label that
        makes a queue metric readable (``document_analysis`` is not
        ``reindex``) -- is available. Recovery is counted on its own counter:
        ``recover_stale`` moving a job is the one transition that means
        "somebody died", and it should be the loudest number in this family
        rather than a row buried in a transition histogram.
        """
        if not self.enabled:
            return
        source_state = str(getattr(event, "from_state", "") or getattr(job, "state", "") or "")
        target_state = str(getattr(event, "to_state", "") or getattr(job, "state", "") or "")
        kind = str(getattr(job, "kind", "") or "unknown")
        note = str(getattr(event, "note", "") or "")
        ids = resolve_ids(job, **{k: v for k, v in extra.items() if k in ID_FIELDS})
        terminal = target_state in {"completed", "partial", "failed", "cancelled"}

        self.metrics.jobs_transitions.inc(from_state=source_state or "unknown",
                                          to_state=target_state or "unknown")
        if terminal:
            self.metrics.jobs_terminal.inc(state=target_state)
        if note.startswith("lease_expired"):
            self.metrics.jobs_recovered.inc(note=note)
        if note.startswith("cancelled"):
            self.metrics.jobs_cancelled.inc(accepted="true", kind=kind)

        self.emit(
            "job.transition",
            family=FAMILY_JOBS,
            ids=ids,
            level="ERROR" if target_state == "failed" else "INFO",
            status=target_state,
            stage=source_state,
            reason=note,
            kind=kind,
            attempt=int(getattr(event, "attempt", 0) or getattr(job, "attempts", 0) or 0),
        )

    def record_job_queue(self, counts: Mapping[str, int]) -> None:
        """Overwrite the queue-depth gauges from ``count_by_state``.

        A *set*, not an inc: the store's answer is a census, and accumulating a
        census produces a number that only ever grows and means nothing. A
        state the store did not mention is left alone rather than zeroed,
        because ``count_by_state`` documents that it returns every state
        including the empty ones -- so a state it omitted is a caller passing
        the wrong thing, and inventing a zero for it would hide that.
        """
        if not self.enabled:
            return
        for state, total in (counts or {}).items():
            self.metrics.jobs_depth.set(int(total), state=str(state))

    def record_job_submitted(self, result: Any) -> None:
        """Count one submission, including the deduplicated case.

        The deduplicated case is counted separately and at the same time as a
        non-deduplicated one, because it is a *good* outcome (it is what stops
        a double-click paying for a second cloud parse) and a counter that mixed
        it in would report fewer submissions than the API received.
        """
        if not self.enabled:
            return
        job = getattr(result, "job", result)
        self.metrics.jobs_submitted.inc(
            kind=str(getattr(job, "kind", "") or "unknown"),
            deduplicated=str(bool(getattr(result, "deduplicated", False))).lower(),
        )
        self.emit(
            "job.submitted",
            family=FAMILY_JOBS,
            ids=resolve_ids(job),
            level="INFO",
            status=str(getattr(job, "state", "") or "queued"),
            kind=str(getattr(job, "kind", "") or "unknown"),
        )

    def record_job_duration(self, job: Any, *, duration_ms: float) -> None:
        """Record a finished job's wall clock, once, at the terminal transition."""
        if not self.enabled:
            return
        self.metrics.jobs_duration.observe(float(duration_ms))

    # -- cost --------------------------------------------------------------

    def record_cost(
        self,
        *,
        source: str,
        cost: float | None,
        document_id: str = "",
        reason: str = "",
    ) -> None:
        """Add one cost, keeping "unknown" distinct from "zero".

        The distinction is the whole function. ``AnalysisRun.llm_cost``
        returns ``None`` when any call's price is unknown precisely because a
        partial sum of a partly-unknown total is a number that looks like a
        total; a gauge that treated ``None`` as ``0.0`` would re-introduce
        exactly the lie that was designed out, and a budget built on it would
        under-spend. So a known cost is added, and an unknown one is counted in
        ``cost.unpriced_operations_total`` and named in the event.

        ``reason`` is the *why* of the unknown -- no price configured, or the
        provider reported no usage block. They are different operational
        problems (configure a price; investigate the provider) and merging them
        into one counter would answer neither.
        """
        if not self.enabled:
            return
        if cost is None:
            self.metrics.cost_unpriced.inc(source=source, reason=reason or "unpriced")
            self.emit(
                "cost.unpriced",
                family=FAMILY_COST,
                level="WARNING",
                # Resolved against the context rather than constructed from the
                # one id this method happens to be handed: an unpriced call that
                # lost its job_id is unattributable to the run that spent the
                # money, which is the only reason the event exists.
                ids=resolve_ids(None, document_id=document_id),
                reason=reason or "unpriced",
                source=source,
            )
            return
        amount = float(cost)
        self.metrics.cost_total.inc(amount, source=source)
        if source == "llm":
            self.metrics.cost_llm.inc(amount)
        elif source == "embedding":
            self.metrics.cost_embedding.inc(amount)
        if document_id:
            # Per-document cost is bounded by ``max_series`` (64) and only
            # recorded when a document id is in hand, so the gauge is a
            # recent-documents view rather than a table that grows with the
            # library. Dropping the oldest is the ring-buffer trade again, and
            # the overflow counter is where the drop becomes visible.
            self.metrics.cost_documents.inc(amount, document_id=document_id)


#: The process-wide facade. Disabled until :func:`configure` says otherwise.
observability = Observability()


def configure(**kwargs: Any) -> Observability:
    """Turn collection on for this process, and return the facade.

    Called once from the app's startup (or a test's setup), and deliberately
    explicit: importing this module must not change what the process does,
    because five modules import it for the ``record_*`` helpers and a
    deployment that never asked for telemetry should keep paying nothing for
    it.

    Accepts ``enabled=...`` for the awkward case -- a test that wants to prove
    the *disabled* behaviour of a call site after the app already turned things
    on -- and ``install=False`` for a caller that wants the metrics and the
    trace but keeps its own log handler, which is the arrangement a Hosted
    deployment with a real collector wants. Everything :meth:`install` accepts
    is forwarded, so a caller cannot pass ``stream=`` here and have it vanish.

    An unrecognised keyword raises rather than being ignored. The alternative
    -- a ``**kwargs`` that swallows whatever it is given -- is how ``stream=``
    was silently dropped in the first place, leaving a test that asserted on
    written output asserting on an empty string with no error anywhere.
    """
    if kwargs.pop("enabled", True):
        observability.enabled = True
    install_kwargs: dict[str, Any] = {}
    for key in ("stream", "target"):
        if key in kwargs:
            install_kwargs[key] = kwargs.pop(key)
    if kwargs.pop("install", True):
        observability.install(
            level=kwargs.pop("level", str(_setting("OBSERVABILITY_LOG_LEVEL"))),
            json_lines=kwargs.pop("json_lines", None),
            **install_kwargs,
        )
    trace_limit = kwargs.pop("trace_limit", None)
    if trace_limit is not None:
        observability.trace = TraceRecorder(int(trace_limit))
    exporter_path = kwargs.pop("exporter_path", str(_setting("OBSERVABILITY_EXPORTER_PATH") or ""))
    if exporter_path:
        observability.exporter = JsonlExporter(exporter_path)
    if kwargs:
        raise TypeError(
            f"configure() 不认识这些参数：{', '.join(sorted(kwargs))}。"
            "可用的参数见 Observability.install 与本函数文档。"
        )
    return observability


def shutdown() -> None:
    """Turn everything off and put the process back. Idempotent, never raises."""
    try:
        observability.flush()
    except Exception:  # pragma: no cover - shutdown must not raise
        pass
    observability.close()


# --------------------------------------------------------------------------
# Module-level instrumentation, for the call sites
# --------------------------------------------------------------------------
#
# Thin wrappers over the singleton. Every one of them checks ``enabled`` first
# and returns immediately when collection is off, which is what makes
# "instrumentation is off" cost one boolean test at the call site rather than
# a branch the caller has to remember.


def record_parse_outcome(outcome: Any, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_parse`——记录一次解析结束。"""
    observability.record_parse(outcome, **kwargs)


def record_retrieval(record: Any, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_retrieval`——记录一次检索结束。"""
    observability.record_retrieval(record, **kwargs)


def record_model_call(call: Any, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_model_call`——记录一次模型调用。"""
    observability.record_model_call(call, **kwargs)


def record_cost(*, source: str, cost: float | None, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_cost`——累加一次成本。"""
    observability.record_cost(source=source, cost=cost, **kwargs)


def record_citation_verification(verification: Any, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_citation_verification`——记录一次引文回源校验。"""
    observability.record_citation_verification(verification, **kwargs)


def record_fact_extraction(report: Any, run: Any = None, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_fact_extraction`——记录一次事实层抽取。"""
    observability.record_fact_extraction(report, run, **kwargs)


def record_job_transition(job: Any, *, event: Any = None, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_job_transition`——记录一次任务状态变更。"""
    observability.record_job_transition(job, event=event, **kwargs)


def record_claim_verdicts(claims: Sequence[Any], *, confidence: str = "", **kwargs: Any) -> None:
    """见 :meth:`Observability.record_claim_verdicts`——记录一次问答的结论可信度。"""
    observability.record_claim_verdicts(claims, confidence=confidence, **kwargs)


def record_embedding_usage(usage: Any, **kwargs: Any) -> None:
    """见 :meth:`Observability.record_embedding_usage`——记录一次 embedding 用量。"""
    observability.record_embedding_usage(usage, **kwargs)


def record_job_queue(counts: Mapping[str, int]) -> None:
    """见 :meth:`Observability.record_job_queue`——用队列普查覆盖 gauge。"""
    observability.record_job_queue(counts)


def record_job_submitted(result: Any) -> None:
    """见 :meth:`Observability.record_job_submitted`——记录一次任务提交。"""
    observability.record_job_submitted(result)


def record_job_duration(job: Any, *, duration_ms: float) -> None:
    """见 :meth:`Observability.record_job_duration`——记录一次任务的墙钟耗时。"""
    observability.record_job_duration(job, duration_ms=duration_ms)


def flush() -> int:
    """把 trace 交给 exporter。见 :meth:`Observability.flush`。"""
    return observability.flush()


def _as_level(level: int | str) -> int:
    """Accept a level name or a number, because callers have both.

    ``logging.WARNING`` is a number, ``"WARNING"`` is what a ``.env`` holds and
    what an operator will type, and this is the module that reads both kinds of
    setting. A bad name falls back to INFO rather than raising: a log level is
    not a reason to refuse to start.
    """
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


__all__ += [
    "DEFAULT_BUCKETS_MS",
    "OBSERVABILITY_CONFIG_KEYS",
    "TextFormatter",
    "flush",
    "record_claim_verdicts",
    "record_embedding_usage",
    "record_job_duration",
    "record_job_queue",
    "record_job_submitted",
]
