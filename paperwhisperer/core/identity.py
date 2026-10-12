"""Where a deployment's embedding provider gets its name.

One function, in ``core/`` rather than beside the embedding gateway that first
needed it, because two layers have to agree on the answer and only one of them
is allowed to import the other. ``core/config.py`` is the bottom of the tree --
it imports nothing but the standard library and ``env_loader`` -- so a rule that
config itself must apply cannot live in ``llm/``. Until this module existed the
rule was written twice: config stamped the index with a constant, and the
gateway named itself after its endpoint. Under the shipped default those two
answers were ``openai_compatible`` and ``api.openai.com``, and
:func:`paperwhisperer.analysis.orchestrator._assert_embedder_matches_index`
correctly refused to write vectors from one space into an index labelled as
another -- which made every upload and every registration fail on a default
configuration. See ``docs/OPTIMIZATION_2026.md`` and the ``index_version``
tests for the full account.
"""

from __future__ import annotations

from urllib.parse import urlparse


def provider_identity(base_url: str) -> str:
    """A short, stable name for the vendor behind ``base_url``.

    Derived from the netloc so two deployments pointing at two vendors produce
    two different cache partitions and two different index versions without
    anyone maintaining a provider name table.
    """
    host = urlparse(str(base_url or "")).netloc or str(base_url or "")
    return host.strip().lower().rstrip("/") or "unknown-provider"
