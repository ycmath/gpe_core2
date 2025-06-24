"""GPU accelerated components for GPE."""

# GPU 모듈은 cupy가 설치된 경우에만 import
try:
    from .stream_decoder import GPEDecoderGPUStream
    from .stream_decoder_meta import GPEDecoderGPUStreamFull
    __all__ = ["GPEDecoderGPUStream", "GPEDecoderGPUStreamFull"]
except ImportError:
    __all__ = []
