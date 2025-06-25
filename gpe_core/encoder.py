from dataclasses import asdict
from typing import Any, Dict, Optional

from .models import GpePayload, BaseRule, AttentionSeed
from .ast_builder import ASTBuilder
from .repetition_detector import RepetitionDetector
from .seed_generator import SeedGenerator
from .json_util import dumps

# ⬇️ 추가 : 최적화 모듈과 설정
from .rule_optimizer import RuleOptimizer
from .config import OPT_CONFIG

# 버전 태그 고정
GPE_V1_0 = "gpe.v1.0"
GPE_V1_1 = "gpe.v1.1"
 
class GPEEncoder:
    def __init__(
        self,
        include_fallback: bool = True,
        enable_optimization: bool = False,
        opt_config: Optional[Dict[str, Any]] = None,
    ):
        self.include_fallback = include_fallback
        self.enable_optimization = enable_optimization
        self.opt_config = opt_config or OPT_CONFIG
    
    def encode(self, data):
        builder = ASTBuilder()
        root_id = builder.build(data)
        rep = RepetitionDetector(builder.nodes).detect()
        seeds: list[AttentionSeed] = SeedGenerator(builder.nodes, rep).generate()

        # ── (1) 규칙 최적화 단계 ──────────────────────────
        if self.enable_optimization:
            # 1) AttentionSeed → flat rule 리스트
            flat_rules: list[BaseRule] = [r for s in seeds for r in s.rules]

            # 2) 최적화
            opt = RuleOptimizer(
                constant_threshold=self.opt_config["constant_threshold"],
                template_threshold=self.opt_config["template_threshold"],
                range_min_length=self.opt_config["range_min_length"],
            )
            flat_rules = opt.optimize_rules(flat_rules)
            gen = dict(
                version=GPE_V1_1,
                root_id=root_id,
                seeds=[asdict(r) for r in flat_rules],   # v1.1 = flat
                optimization=opt.get_stats(),
            )
            payload_version = GPE_V1_1
        else:
            gen = dict(
                version=GPE_V1_0,
                root_id=root_id,
                seeds=[asdict(s) for s in seeds],        # v1.0 = [{rules:[…]}]
            )
            payload_version = GPE_V1_0
        fb = dumps(data) if self.include_fallback else None
        return GpePayload(
            payload_type=payload_version,
            generative_payload=gen,
            fallback_payload={"json": fb} if fb else None,
        )
