"""tests.test_hint
Smoke-test for Rule-Optimizer + Encoder integration.
"""
from gpe_core2.encoder import base as enc
from gpe_core2.decoder import base as dec
from gpe_core2.glassbox import reasoning_engine as gb


def test_constant_cardinality_hint():
    """Encoder should choose OP_CONST when cardinality is small."""
    # 1. GlassBox produces data + hints
    res = gb.reason("fruits")
    data = res["answer"]
    hints = res["gpe_hints"]

    # 2. Encode with hints
    payload = enc.encode(data, gpe_hints=hints)

    # 3. Basic round-trip check
    assert dec.decode(payload) == data

    # 4. Inspect generative rules – must contain at least one CONST rule
    rules = payload.generative_payload["rules"]
    assert any(r.get("op_code") == "OP_CONST" for r in rules)
