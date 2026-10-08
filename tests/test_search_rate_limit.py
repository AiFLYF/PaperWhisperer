"""Per-provider proactive rate limiting in the shared metadata client.

These tests pin the *proactive* spacing the providers ask for, not the 429
backoff that already existed: arXiv states its rule as a hard rate (one request
every three seconds) and refuses bursts, so waiting for a refusal is too late.
The shared client therefore spaces requests per host before sending them, and
these tests prove that spacing holds — including under concurrency, where a
plain "sleep the interval then request" would let a burst of callers through
together.

The clock is faked so the tests are instant and assert the exact gaps; the
sleeping a real throttle would do is arithmetic on the reserved slot, and that
arithmetic is what is under test.
"""

from __future__ import annotations

import threading

import pytest

from paperwhisperer.search import http


class FakeClock:
    """A monotonic clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    http.reset_rate_limits()
    fake = FakeClock()
    monkeypatch.setattr(http, "time", fake)
    yield fake
    http.reset_rate_limits()


class TestEachProviderIsSpacedByItsOwnRule:
    def test_arxiv_is_three_seconds_apart(self, clock):
        for _ in range(3):
            http._throttle("http://export.arxiv.org/api/query?search_query=x")
        # The first call reserves its slot and does not wait; each later call
        # waits exactly one interval, which is the rule arXiv publishes.
        assert clock.sleeps == [3.0, 3.0]

    def test_semantic_scholar_is_one_second_apart(self, clock):
        http._throttle("https://api.semanticscholar.org/graph/v1/paper/search")
        http._throttle("https://api.semanticscholar.org/graph/v1/paper/search")
        assert clock.sleeps == [1.0]

    def test_crossref_and_openalex_are_spaced_fast(self, clock):
        # Two consecutive calls to each; each waits one short interval.
        http._throttle("https://api.crossref.org/works")
        http._throttle("https://api.crossref.org/works")
        http._throttle("https://api.openalex.org/works")
        http._throttle("https://api.openalex.org/works")
        assert clock.sleeps == pytest.approx([0.1, 0.05])

    def test_an_unrecognised_host_is_not_guessed_at(self, clock):
        """No rate is invented for a service whose terms were never read."""
        for _ in range(3):
            http._throttle("https://example.org/api")
        assert clock.sleeps == []

    def test_providers_do_not_share_a_gate(self, clock):
        """arXiv's three seconds must not also throttle Crossref."""
        http._throttle("http://export.arxiv.org/api/query")
        http._throttle("https://api.crossref.org/works")
        assert clock.sleeps == []


class TestTheIntervalIsConfiguration:
    def test_a_configured_interval_is_honoured(self, monkeypatch):
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "ARXIV_MIN_INTERVAL_SECONDS", 0.25)
        http.reset_rate_limits()
        fake = FakeClock()
        monkeypatch.setattr(http, "time", fake)
        try:
            http._throttle("http://export.arxiv.org/api/query")
            http._throttle("http://export.arxiv.org/api/query")
            assert fake.sleeps == [0.25]
        finally:
            http.reset_rate_limits()

    def test_zero_disables_the_proactive_spacing(self, monkeypatch):
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "ARXIV_MIN_INTERVAL_SECONDS", 0.0)
        http.reset_rate_limits()
        fake = FakeClock()
        monkeypatch.setattr(http, "time", fake)
        try:
            for _ in range(5):
                http._throttle("http://export.arxiv.org/api/query")
            assert fake.sleeps == []
        finally:
            http.reset_rate_limits()


class TestConcurrentCallersGetDistinctSlots:
    def test_a_burst_is_serialised_not_released_together(self, monkeypatch):
        """Concurrent callers are released one interval apart, not together.

        This is the property a plain "sleep(interval); request" lacks: several
        threads starting together would all sleep the same amount and then all
        fire at once. The assertion is the *spread* of the release times — a
        real clock and a real gate, so if the gate stopped serialising, four
        callers would leave within microseconds of each other and the spread
        would collapse to ~0. Measuring elapsed time (rather than just counting
        releases) is what makes this test able to fail.
        """
        import time as real_time

        from paperwhisperer.core import config

        interval = 0.05
        monkeypatch.setattr(config, "CROSSREF_MIN_INTERVAL_SECONDS", interval)
        http.reset_rate_limits()

        released: list[float] = []
        lock = threading.Lock()
        start_barrier = threading.Barrier(4)

        def worker() -> None:
            start_barrier.wait()
            http._throttle("https://api.crossref.org/works")
            with lock:
                released.append(real_time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        try:
            assert len(released) == 4
            spread = max(released) - min(released)
            # Three gaps between four callers: with a working gate this is at
            # least ~3 intervals; released together it would be ~0. The slack
            # absorbs scheduling jitter without letting a collapsed gate pass.
            assert spread >= interval * 2, (
                f"callers released {spread:.4f}s apart; a burst went through "
                f"together instead of being spaced by ~{interval}s"
            )
        finally:
            http.reset_rate_limits()


class TestTheGateSitsInFrontOfEveryRequest:
    def test_repeated_json_gets_to_one_host_are_spaced(self, monkeypatch):
        """The spacing is in the request path, not something callers opt into."""
        calls: list[str] = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"{}"

            class _Headers:
                def get_content_charset(self):
                    return "utf-8"

            headers = _Headers()

        def fake_urlopen(request, **kwargs):
            calls.append(request.full_url)
            return FakeResponse()

        http.reset_rate_limits()
        fake = FakeClock()
        monkeypatch.setattr(http, "time", fake)
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        try:
            http.http_get_json("https://api.crossref.org/works", retries=1)
            http.http_get_json("https://api.crossref.org/works", retries=1)
            assert len(calls) == 2, "both requests must still be sent"
            assert fake.sleeps == pytest.approx([0.1]), "the second must wait the configured gap"
        finally:
            http.reset_rate_limits()