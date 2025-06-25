import os, pytest
from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

@pytest.mark.parametrize("numba_flag", ["true", "false"])
def test_v11_roundtrip(numba_flag):
    os.environ["GPE_USE_NUMBA"] = numba_flag
    data_cases = [
        [1]*15,                          # ConstantRule
        list(range(8)),                  # RangeRule
        [0,0,0,9,0,0,0,5,0,0],           # CompactListRule
    ]
    enc = GPEEncoder(enable_optimization=True, include_fallback=False)
    dec = GPEDecoder()

    for dat in data_cases:
        blob = enc.encode(dat)
        assert blob.generative_payload["version"] == "gpe.v1.1"
        assert dec.decode(blob) == dat
