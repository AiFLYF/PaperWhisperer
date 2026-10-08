"""DOI recovery on the reading-queue save.

"DOI 丢失可回查" is only a product feature if something calls it. The
enrichment library existed, was tested, and was unreachable from production —
so these tests are about the *hook*, not about the lookup: what the save
triggers, what it refuses to trigger, and what it does when the providers are
down.

**No test here reaches the network.** Both provider builders are replaced with
stubs that record every :class:`~paperwhisperer.search.enrich.LookupQuery`
they are handed, so a test can assert on the *absence* of a call (the paper
that already has a DOI must cost nothing) as easily as on its presence. That
asymmetry is the whole reason the seam is the builder rather than the fetcher:
one that only appears on a hit could not be observed not appearing.

**Every assertion is on what a provider could have destroyed.** A wrong DOI is
worse than a missing one — it makes the paper un-deduplicatable against
everything, permanently — so the tests below pin the three ways that damage
could happen: a guess written onto a record that did not need one, a provider's
answer surviving after the provider failed, and a partial write leaving an
``identity_key`` that disagrees with the identifiers beside it.
"""

from __future__ import annotations

import json

import pytest

from paperwhisperer.api import routes_papers
from paperwhisperer.search import enrich
from paperwhisperer.search.enrich import (
    LookupQuery,
    MetadataSource,
    parse_crossref_payload,
    parse_openalex_payload,
)


class StubProviders:
    """Stand-ins for Crossref and OpenAlex that record every question asked.

    ``answer`` receives a :class:`LookupQuery` and returns the raw provider
    payload to return, or raises to simulate a provider that is down. The
    default answer builds a Crossref payload that *verifies* against the query,
    which is what a healthy provider looks like from the enrichment layer's
    point of view.
    """

    def __init__(self, answer=None):
        self.queries: list[tuple[str, LookupQuery]] = []
        self._answer = answer or self._verified_crossref

    # -- the fake answers ------------------------------------------------ #

    @staticmethod
    def _verified_crossref(query: LookupQuery):
        """A Crossref payload that matches the query on title, author and year.

        Mirroring the query back is deliberate: enrichment verifies a candidate
        before writing it, so a stub that ignored the query would return an
        unverified candidate and the "the DOI was recovered" assertion would be
        testing the stub rather than the merge.
        """
        return {
            "message": {
                "items": [
                    {
                        "DOI": f"10.5555/{abs(hash(query.title)) % 10**8:08d}",
                        "title": [query.title],
                        "author": [{"family": query.author_for_query()}],
                        "issued": {"date-parts": [[int(query.year or 2017)]]},
                    }
                ]
            }
        }

    # -- wiring ---------------------------------------------------------- #

    def _source(self, name, parse) -> MetadataSource:
        def fetch(query: LookupQuery):
            self.queries.append((name, query))
            return self._answer(query)

        return MetadataSource(name=name, fetch=fetch, parse=parse)

    def install(self, monkeypatch) -> StubProviders:
        """Replace both production provider builders with these stubs.

        Patched on ``enrich``, not on the route: ``default_metadata_sources``
        resolves ``build_crossref_source`` in its own module globals, so this is
        the seam that stops the network — and it is the same seam a future
        change to the provider list has to keep working.
        """
        monkeypatch.setattr(enrich, "build_crossref_source", self._source_factory("Crossref", parse_crossref_payload))
        monkeypatch.setattr(enrich, "build_openalex_source", self._source_factory("OpenAlex", parse_openalex_payload))
        return self

    def _source_factory(self, name, parse):
        return lambda *args, **kwargs: self._source(name, parse)

    # -- assertions ------------------------------------------------------ #

    @property
    def asked_titles(self) -> list[str]:
        return [query.title for _name, query in self.queries]


@pytest.fixture
def providers(monkeypatch):
    """Stub both metadata providers; returns the recorder."""
    return StubProviders().install(monkeypatch)


def _item(title, author="Ada Lovelace", year="2020", **extra):
    """One client-supplied paper, in the shape a search result is sent in."""
    return {
        "source": "Test Provider",
        "title": title,
        "authors": [author],
        "year": year,
        "url": f"https://example.org/papers/{abs(hash(title)) % 10**8:08d}",
        **extra,
    }


def _save(client, session_factory, items):
    """POST a reading queue the way the UI does, and return the parsed body."""
    session_id, token = session_factory()
    response = client.post(
        "/api/reading-queue",
        json={"session_id": session_id, "session_token": token, "items": items},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _json_request(body: bytes):
    """A ``Request`` over a fixed JSON body, for driving a route without ASGI.

    ``TestClient`` serves every request on a portal thread of its own, which
    makes "did this run on the event loop's thread?" unanswerable — a blocking
    call on that thread would look identical to a correctly offloaded one.
    Calling the route directly pins it to *this* thread's loop, so a heartbeat
    task is real evidence either way.
    """
    from starlette.requests import Request

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/reading-queue",
        "headers": [(b"content-type", b"application/json")],
    }
    return Request(scope, receive)


class TestEnrichmentIsOnlyReachedWhenItCanHelp:
    def test_a_paper_that_already_has_a_doi_costs_no_provider_call(
        self, client, session_factory, providers
    ):
        body = _save(
            client,
            session_factory,
            [_item("A Study of Already Known Things", doi="10.1234/already.known")],
        )

        assert providers.queries == []
        assert body["items"][0]["doi"] == "10.1234/already.known"

    def test_the_record_a_client_sent_is_stored_byte_for_byte(
        self, client, session_factory, providers
    ):
        original = _item("A Study of Already Known Things", doi="10.1234/already.known")

        body = _save(client, session_factory, [original])

        stored = body["items"][0]
        assert stored["doi"] == original["doi"]
        assert stored["title"] == original["title"]
        assert stored["authors"] == original["authors"]
        assert stored["year"] == original["year"]
        assert stored["identity_key"] == "doi:10.1234/already.known"

    def test_a_queue_of_dois_only_is_not_enriched_at_all(
        self, client, session_factory, providers
    ):
        """The cap counts *candidates*, so nothing is looked up when nothing is missing."""
        body = _save(
            client,
            session_factory,
            [
                _item("First Known Paper", doi="10.1234/one"),
                _item("Second Known Paper", doi="10.1234/two"),
            ],
        )

        assert providers.queries == []
        assert [item["doi"] for item in body["items"]] == ["10.1234/one", "10.1234/two"]


class TestAMissingDoiIsRecovered:
    def test_the_recovered_doi_is_returned_and_stored(
        self, client, session_factory, providers
    ):
        body = _save(
            client,
            session_factory,
            [_item("Attention Is All You Need", author="Ashish Vaswani", year="2017")],
        )

        item = body["items"][0]
        assert item["doi"], "the DOI was not recovered"
        assert item["doi"].startswith("10.5555/")
        assert item["identity_key"] == f"doi:{item['doi']}"

    def test_only_the_doi_is_added_and_the_rest_of_the_record_is_untouched(
        self, client, session_factory, providers
    ):
        """One recovered identifier, not a re-import of what a provider happens to know.

        Enrichment also fills ``year`` and ``first_author``, and those are
        deliberately *not* merged: the record's ``authors`` is the list the
        reader is shown, and a recovered surname that appears nowhere in it
        would be a name the user cannot find. So the merge is scoped to the
        four identifier fields and nothing else about the record moves.
        """
        original = _item("Attention Is All You Need", author="Ashish Vaswani", year="2017")

        item = _save(client, session_factory, [original])["items"][0]

        assert item["title"] == original["title"]
        assert item["authors"] == ["Ashish Vaswani"]
        assert item["year"] == "2017"
        assert item["url"] == original["url"]
        assert item["venue"] == ""
        assert item["doi"].startswith("10.5555/")
        assert item["arxiv_id"] == ""
        assert item["s2_paper_id"] == ""
        assert item["openalex_id"] == ""
        # The strongest form of "nothing else moved": the stored record has
        # exactly the key set `normalize_paper_collection` produces, so no
        # enrichment-only field (`first_author`, `errors`, `notes`) has leaked
        # into the record the session store will hand back on the next load.
        assert set(item) == {
            "abstract",
            "arxiv_id",
            "authors",
            "doi",
            "identity_key",
            "openalex_id",
            "paper_id",
            "pdf_url",
            "s2_paper_id",
            "saved_at",
            "source",
            "title",
            "url",
            "venue",
            "year",
        }

    def test_metadata_enrichment_finds_is_not_merged_onto_the_record(
        self, client, session_factory, monkeypatch
    ):
        """The rule that actually needs a record too thin to test it implicitly.

        A record that arrives with an author and a year makes the ``year`` /
        ``first_author`` parts of ``filled_fields`` unreachable: ``_apply_candidate``
        skips any field the record already has, so the narrowing of the merge to
        the identifier fields is never exercised. A bare record — title and
        nothing else, which is exactly what a pasted reference looks like — does
        exercise it, because the provider's year and author both come back as
        genuinely new fields.

        Both are still refused: ``year`` is a display field the record's own
        ``authors`` list does not corroborate, and ``first_author`` is a surname
        that appears nowhere in the list the reader is shown.
        """
        def answered(query):
            return {
                "message": {
                    "items": [
                        {
                            "DOI": "10.5555/recovered.bare",
                            "title": [query.title],
                            "author": [{"family": "Vaswani", "given": "Ashish"}],
                            "issued": {"date-parts": [[2017]]},
                        }
                    ]
                }
            }

        providers = StubProviders(answer=answered).install(monkeypatch)

        item = _save(
            client,
            session_factory,
            [{"source": "Pasted", "title": "Attention Is All You Need"}],
        )["items"][0]

        assert providers.asked_titles == ["Attention Is All You Need"]
        assert item["doi"] == "10.5555/recovered.bare", "the DOI should still be recovered"
        assert item["year"] == "", "a provider's year must not be written onto the record"
        assert item["authors"] == []
        assert "first_author" not in item, "a display field leaked into the stored record"

    def test_an_identifier_the_record_already_carried_is_never_overwritten(
        self, client, session_factory, providers
    ):
        """Crossref's ``has-preprint`` relation is how the arXiv id comes back.

        The record already has one. Enrichment fills only empty fields, so the
        provider's copy must not land on top of it — and if it did, the two would
        disagree about the same paper with no record of which was checked.
        """
        body = _save(
            client,
            session_factory,
            [
                _item(
                    "Attention Is All You Need",
                    author="Ashish Vaswani",
                    year="2017",
                    arxiv_id="1706.03762",
                )
            ],
        )

        item = body["items"][0]
        assert item["arxiv_id"] == "1706.03762"
        assert item["doi"].startswith("10.5555/")

    def test_the_persisted_session_reloads_with_the_recovered_doi(
        self, client, session_factory, providers
    ):
        """The gain has to survive the write, not just the response body."""
        from paperwhisperer.sessions.store import load_session_payload

        session_id, token = session_factory()
        response = client.post(
            "/api/reading-queue",
            json={
                "session_id": session_id,
                "session_token": token,
                "items": [_item("Attention Is All You Need", author="Ashish Vaswani")],
            },
        )
        assert response.status_code == 200

        payload = load_session_payload(session_id)
        stored = payload["paper_search"]["reading_queue"][0]
        assert stored["doi"] == response.json()["items"][0]["doi"]
        assert stored["doi"]


class TestAFailingProviderChangesNothing:
    def test_a_provider_that_is_down_leaves_the_original_record_intact(
        self, client, session_factory, monkeypatch
    ):
        def offline(query):
            raise OSError("connection refused")

        providers = StubProviders(answer=offline).install(monkeypatch)
        original = _item("A Paper Nobody Can Look Up", author="Grace Hopper")

        body = _save(client, session_factory, [original])

        assert providers.queries, "the provider should have been asked"
        item = body["items"][0]
        assert item["doi"] == ""
        assert item["title"] == original["title"]
        assert item["authors"] == original["authors"]
        assert body["count"] == 1

    def test_a_provider_returning_nothing_leaves_the_record_alone(
        self, client, session_factory, monkeypatch
    ):
        providers = StubProviders(answer=lambda query: None).install(monkeypatch)

        body = _save(client, session_factory, [_item("An Unindexed Paper")])

        assert providers.queries
        assert body["items"][0]["doi"] == ""

    def test_a_candidate_that_fails_verification_is_not_written(
        self, client, session_factory, monkeypatch
    ):
        """A verified-looking answer for a *different* paper must not be stored.

        The stub always answers with the right title and a wrong year, which is
        the shape of the failure the verification check exists for: Crossref's
        top hit for a common title, belonging to somebody else's paper.
        """
        def wrong_year(query):
            return {
                "message": {
                    "items": [
                        {
                            "DOI": "10.5555/somebody.elses.paper",
                            "title": [query.title],
                            "author": [{"family": query.author_for_query()}],
                            "issued": {"date-parts": [[1998]]},
                        }
                    ]
                }
            }

        providers = StubProviders(answer=wrong_year).install(monkeypatch)

        body = _save(client, session_factory, [_item("A Common Title", year="2021")])

        assert providers.queries
        assert body["items"][0]["doi"] == ""

    def test_enrichment_itself_raising_does_not_fail_the_save(self, client, session_factory, providers):
        """The guard around the call, not just the one inside it."""
        def explode(identity, sources=()):
            raise RuntimeError("the enrichment module fell over")

        original = routes_papers.enrich_identity
        try:
            routes_papers.enrich_identity = explode
            body = _save(client, session_factory, [_item("A Paper Worth Keeping")])
        finally:
            routes_papers.enrich_identity = original

        assert body["count"] == 1
        assert body["items"][0]["title"] == "A Paper Worth Keeping"
        assert body["items"][0]["doi"] == ""

class TestTheLookupStaysOffTheEventLoop:
    def test_enrichment_runs_on_another_thread_while_the_route_is_suspended(
        self, session_factory, monkeypatch
    ):
        """A blocking HTTP client called from the coroutine blocks every other request.

        The route is ``async`` and the fetchers are blocking ``urllib``, so the
        pass is handed to a worker thread. What that buys is not a thread name
        but *concurrency*: while a lookup is in flight the event loop must stay
        free to serve the requests already behind it.

        The thread the loop runs on is known here — it is this test's thread,
        because the route is driven directly rather than through
        ``TestClient``. (``TestClient`` serves every request on a portal thread
        of its own, which would make "was this on the loop's thread?"
        unanswerable: a blocking call there looks identical to a correct one.)

        The stub blocks until the heartbeat has fired, so the assertion is that
        the loop kept turning during a lookup rather than after it.
        """
        import asyncio
        import threading

        loop_turns = asyncio.Event()
        released = threading.Event()
        lookup_threads: list[int] = []

        async def heartbeat():
            # Yields the loop the way a second live request would.
            await asyncio.sleep(0.05)
            loop_turns.set()

        def blocking(query):
            lookup_threads.append(threading.current_thread().ident)
            released.wait(timeout=5)
            return StubProviders._verified_crossref(query)

        StubProviders(answer=blocking).install(monkeypatch)
        session_id, token = session_factory()
        body = json.dumps(
            {
                "session_id": session_id,
                "session_token": token,
                "items": [_item("A Slow Lookup")],
            }
        ).encode()

        async def drive():
            task = asyncio.create_task(
                routes_papers.save_reading_queue(_json_request(body))
            )
            beat = asyncio.create_task(heartbeat())
            # `to_thread` suspends the route, so the heartbeat completes while
            # the worker is still blocked and this returns on the heartbeat. An
            # inline call would instead stall the loop until the worker's own
            # five-second timeout expired.
            done, _pending = await asyncio.wait(
                {task, beat}, timeout=2, return_when=asyncio.FIRST_COMPLETED
            )
            released.set()
            if beat not in done:
                pytest.fail("the event loop was blocked until the lookup finished")
            return await task

        loop_thread = threading.current_thread().ident
        response = asyncio.run(drive())

        assert response["count"] == 1
        assert lookup_threads, "the provider was never asked"
        assert lookup_threads[0] != loop_thread, "the blocking lookup ran on the event loop thread"


class TestTheCapBoundsOneSave:
    def test_more_doi_less_papers_than_the_cap_are_not_all_enriched(
        self, client, session_factory, providers
    ):
        limit = routes_papers.READING_QUEUE_ENRICHMENT_LIMIT
        titles = [f"A Distinct Unindexed Paper Number {index}" for index in range(limit + 3)]

        body = _save(
            client, session_factory, [_item(title) for title in titles]
        )

        assert body["count"] == len(titles), "the cap must drop nothing, only skip lookups"
        assert len(providers.queries) == limit, "one save fired more lookups than the cap allows"
        recovered = [bool(item["doi"]) for item in body["items"]]
        assert recovered == [True] * limit + [False] * 3
        # Which papers were looked up is the cap's contract (the first `limit`);
        # the order those independent lookups happen in is not — they run
        # concurrently, so the provider sees them in whatever order the threads
        # reach it. Assert the set, not the sequence.
        assert set(providers.asked_titles) == set(titles[:limit])

    def test_the_cap_is_small_enough_to_protect_a_shared_api(self):
        """A guard on the guard: the queue holds 30 items and the providers do not."""
        from paperwhisperer.core import config

        assert 0 < routes_papers.READING_QUEUE_ENRICHMENT_LIMIT <= 10
        assert routes_papers.READING_QUEUE_ENRICHMENT_LIMIT < config.READING_QUEUE_LIMIT
