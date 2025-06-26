"""gpe_core2.glassbox.reasoning_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stubbed *GlassBox* backend.

In production CGA this module will wrap the actual DL‑ARE/EVG pipeline and return
JSON‑serialisable objects **plus** a lightweight *gpe_hints* block describing
basic statistics of the result so that the encoder can pick smarter rules.

For Sprint‑1 we only need a deterministic toy implementation that produces:
    • `answer`  : the data to be encoded
    • `gpe_hints`: {"cardinality": <int>, "is_vector": <bool>}
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

__all__ = ["reason"]


# ---------------------------------------------------------------------------
# Helper – very small heuristics to extract hint metrics
# ---------------------------------------------------------------------------

def _infer_cardinality(val: Any) -> int | None:
    """Return number of unique elements if *val* is a short iterable."""
    if isinstance(val, (list, tuple)) and len(val) <= 256:
        return len({*val})
    return None


def _is_vector(val: Any) -> bool:
    """Detect 1‑D numeric vector (rough)."""
    if (isinstance(val, list) and val and all(isinstance(x, (int, float)) for x in val)):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason(query: str) -> Dict[str, Any]:
    """Return a toy answer + *gpe_hints* driven by the *query* string.

    Examples
    --------
    >>> reason("fruits")
    {'answer': ['apple', 'apple', 'apple'], 'gpe_hints': {'cardinality': 3}}
    """
    query_low = query.lower().strip()

    # Very naive branching just for demonstration
    if "vector" in query_low:
        # 128‑length numeric embedding
        vec = [math.sin(i) for i in range(128)]
        hints: Mapping[str, Any] = {"is_vector": True}
        return {"answer": vec, "gpe_hints": hints}

    # default case – constant list
    fruits = ["apple", "apple", "apple"]
    hints = {"cardinality": _infer_cardinality(fruits)}
    return {"answer": fruits, "gpe_hints": hints}

