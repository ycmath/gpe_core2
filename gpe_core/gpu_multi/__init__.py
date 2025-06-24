"""Multi-GPU components using Ray."""

try:
    from .multi_decoder import GPEDecoderGPU_Ray
    __all__ = ["GPEDecoderGPU_Ray"]
except ImportError:
    __all__ = []
