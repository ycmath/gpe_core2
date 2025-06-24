"""
Encoder ↔ Decoder end-to-end round-trip.

• JSON → encode() → payload
• payload → decode() → JSON'
• JSON' must equal original
"""
import json, copy
from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

def _sample():
    return {
        "car": {
            "wheels": [{"tire": "summer", "size": 18}]*4,
            "doors": 4,
            "meta": {"vin": "XYZ123"}
        }
    }

def test_roundtrip_cpu():
    original = _sample()
    enc = GPEEncoder(include_fallback=False)
    payload = enc.encode(copy.deepcopy(original))
    dec = GPEDecoder()
    restored = dec.decode(payload)
    assert restored == original
