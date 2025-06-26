"""gpe_core2.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~
Expressway **Hash-Lane** (Sprint-2) – query-level caching facade.

Usage
-----
>>> orch = Orchestrator()
>>> payload_1 = orch.process_query("What are popular fruits today?")  # → cache miss
>>> payload_2 = orch.process_query("What are popular fruits today?")  # → cache hit, latency↓

Design
------
* A *normalized query string* → **64-bit xxHash** → Redis key.
* Value stored: **orjson-serialised GpePayload** (dict form).
* TTL default 3600 s (configurable).
* On cache hit, `metadata["cached"] = True` is set before return.
"""
from __future__ import annotations

import os
import xxhash
import orjson
import redis
from typing import Any, Mapping

from gpe_core2.encoder import base as enc
from gpe_core2.decoder import base as dec  # may be unused now, kept for completeness
from gpe_core2.glassbox import reasoning_engine as gb

# Legacy models (will be refactored later)
from gpe_core1_compat.models import GpePayload  # type: ignore

__all__ = ["Orchestrator"]


class Orchestrator:  # noqa: D101 – docstring above
    _REDIS_URL = os.getenv("GPE_REDIS_URL", "redis://localhost:6379/0")
    _TTL = int(os.getenv("GPE_CACHE_TTL", "3600"))  # seconds

    def __init__(self) -> None:
        self.rds = redis.Redis.from_url(self._REDIS_URL, decode_responses=False)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def process_query(self, query: str) -> GpePayload:
        """Return *GpePayload* – cached if possible."""
        key = self._hash(query)

        # 1. Cache lookup --------------------------------------------------
        cached_bytes = self.rds.get(key)
        if cached_bytes is not None:
            payload = self._bytes_to_payload(cached_bytes)
            payload.metadata["cached"] = True
            return payload

        # 2. GlassBox reasoning -------------------------------------------
        result = gb.reason(query)
        data = result["answer"]
        hints: Mapping[str, Any] | None = result.get("gpe_hints")  # type: ignore[arg-type]

        payload = enc.encode(data, gpe_hints=hints)

        # 3. Store in cache ------------------------------------------------
        self.rds.setex(key, self._TTL, self._payload_to_bytes(payload))
        return payload

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hash(query: str) -> str:
        """Return 16-hex-char xxhash64 of *normalized* query."""
        norm = " ".join(query.lower().split())  # whitespace collapse
        return xxhash.xxh64(norm).hexdigest()

    @staticmethod
    def _payload_to_bytes(payload: GpePayload) -> bytes:  # type: ignore[name-defined]
        return orjson.dumps(payload, default=lambda o: o.__dict__)  # naive serialisation

    @staticmethod
    def _bytes_to_payload(blob: bytes) -> GpePayload:  # type: ignore[name-defined]
        d: dict[str, Any] = orjson.loads(blob)
        return GpePayload(**d)  # type: ignore[arg-type]

