"""GPE Core - Generative Payload Encapsulation protocol implementation."""

__version__ = "0.1.0"
__author__ = "YC Math"

# 주요 클래스들 export
from .models import (
    ASTNode,
    BaseRule,
    InstantiateRule,
    AppendChildRule,
    RepeatRule,
    AttentionSeed,
    GpePayload,
)
from .encoder import GPEEncoder
from .decoder import GPEDecoder, GPEDecodeError
from .decoder_mp import GPEDecoderMP
from .decoder_mp_shm import GPEDecoderMP_ShMem

# 선택적 imports (의존성이 설치된 경우만)
try:
    from .gpu.stream_decoder import GPEDecoderGPUStream
    from .gpu.stream_decoder_meta import GPEDecoderGPUStreamFull
except ImportError:
    pass

try:
    from .gpu_multi.multi_decoder import GPEDecoderGPU_Ray
except ImportError:
    pass

__all__ = [
    "ASTNode",
    "BaseRule",
    "InstantiateRule",
    "AppendChildRule",
    "RepeatRule",
    "AttentionSeed",
    "GpePayload",
    "GPEEncoder",
    "GPEDecoder",
    "GPEDecodeError",
    "GPEDecoderMP",
    "GPEDecoderMP_ShMem",
]
