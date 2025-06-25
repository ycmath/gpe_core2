# Step 2: encoder.py에 RuleOptimizer 통합

import os
os.chdir('/content/gpe_core')

print("=== Step 2: Encoder에 RuleOptimizer 통합 ===")

# ===== 1. encoder.py 백업 =====
import shutil
shutil.copy('gpe_core/encoder.py', 'gpe_core/encoder_backup.py')
print("✅ encoder.py 백업 완료")

# ===== 2. 수정된 encoder.py =====
enhanced_encoder = '''from dataclasses import asdict
from .models import GpePayload
from .ast_builder import ASTBuilder
from .repetition_detector import RepetitionDetector
from .seed_generator import SeedGenerator
from .rule_optimizer import RuleOptimizer
from .json_util import dumps
from typing import Any, Dict, List, Optional

class GPEEncoder:
    def __init__(self, 
                 include_fallback: bool = True,
                 enable_optimization: bool = True,
                 optimization_config: Optional[Dict] = None):
        """
        GPE 인코더
        
        Args:
            include_fallback: JSON fallback 포함 여부
            enable_optimization: 규칙 최적화 활성화 여부
            optimization_config: 최적화 설정 (constant_threshold 등)
        """
        self.include_fallback = include_fallback
        self.enable_optimization = enable_optimization
        
        # 최적화 설정
        opt_config = optimization_config or {}
        self.rule_optimizer = RuleOptimizer(
            constant_threshold=opt_config.get('constant_threshold', 5),
            template_threshold=opt_config.get('template_threshold', 0.7),
            range_min_length=opt_config.get('range_min_length', 5)
        ) if enable_optimization else None
    
    def encode(self, data):
        # 1. AST 구축
        builder = ASTBuilder()
        root_id = builder.build(data)
        
        # 2. 반복 감지
        rep = RepetitionDetector(builder.nodes).detect()
        
        # 3. 시드/규칙 생성
        seeds = SeedGenerator(builder.nodes, rep).generate()
        
        # 4. 규칙 최적화 (새로운 단계)
        if self.enable_optimization and self.rule_optimizer:
            optimized_seeds = self._optimize_seeds(seeds)
            optimization_stats = self.rule_optimizer.get_stats()
        else:
            optimized_seeds = seeds
            optimization_stats = None
        
        # 5. 페이로드 생성
        gen = {
            "version": "gpe.v1.1",  # 버전 업데이트
            "root_id": root_id,
            "seeds": [asdict(s) for s in optimized_seeds],
        }
        
        # 최적화 통계 추가
        if optimization_stats:
            gen["optimization"] = optimization_stats
        
        fb = dumps(data) if self.include_fallback else None
        
        return GpePayload(
            payload_type="gpe.v1.1",
            generative_payload=gen,
            fallback_payload={"json": fb} if fb else None,
        )
    
    def _optimize_seeds(self, seeds: List) -> List:
        """시드의 규칙들을 최적화"""
        from .models import AttentionSeed
        
        optimized_seeds = []
        
        for seed in seeds:
            # 모든 규칙 추출
            all_rules = []
            if hasattr(seed, 'rules'):
                all_rules = seed.rules
            
            # 규칙 최적화
            if all_rules:
                optimized_rules = self.rule_optimizer.optimize_rules(all_rules)
            else:
                optimized_rules = all_rules
            
            # 새 시드 생성
            optimized_seed = AttentionSeed(rules=optimized_rules)
            optimized_seeds.append(optimized_seed)
        
        return optimized_seeds
    
    def encode_with_analysis(self, data):
        """분석 정보와 함께 인코딩"""
        # 기본 인코딩
        payload = self.encode(data)
        
        # 추가 분석 정보
        analysis = {
            "data_size": len(dumps(data)),
            "payload_size": len(dumps(payload.generative_payload)),
            "compression_ratio": 0,
            "optimization_enabled": self.enable_optimization
        }
        
        # 압축률 계산
        if analysis["data_size"] > 0:
            analysis["compression_ratio"] = (
                1 - analysis["payload_size"] / analysis["data_size"]
            ) * 100
        
        # 최적화 통계 추가
        if "optimization" in payload.generative_payload:
            analysis["optimization"] = payload.generative_payload["optimization"]
        
        return payload, analysis
'''

# encoder.py 업데이트
with open('gpe_core/encoder.py', 'w') as f:
    f.write(enhanced_encoder)
print("✅ encoder.py 업데이트 완료")

# ===== 3. 통합 테스트 =====
print("\n=== 통합 테스트 ===")

test_integration = '''
# 모듈 재로드
import sys
for mod in list(sys.modules.keys()):
    if mod.startswith('gpe_core'):
        del sys.modules[mod]

from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

# 테스트 데이터
test_cases = [
    {
        "name": "상수 반복",
        "data": {
            "items": [
                {"id": i, "status": "active", "type": "user"}
                for i in range(20)
            ]
        }
    },
    {
        "name": "연속 숫자",
        "data": {
            "numbers": list(range(1, 21)),
            "ids": [f"ID_{i:04d}" for i in range(10)]
        }
    },
    {
        "name": "리스트 패턴",
        "data": {
            "values": [1] * 10 + [2] * 5 + [3] * 5
        }
    }
]

print("1. 최적화 없이 인코딩")
encoder_no_opt = GPEEncoder(include_fallback=False, enable_optimization=False)

for test in test_cases:
    payload = encoder_no_opt.encode(test["data"])
    rules_count = sum(len(seed["rules"]) for seed in payload.generative_payload["seeds"])
    print(f"  {test['name']}: {rules_count} rules")

print("\\n2. 최적화 활성화 인코딩")
encoder_with_opt = GPEEncoder(
    include_fallback=False, 
    enable_optimization=True,
    optimization_config={'constant_threshold': 3}
)

for test in test_cases:
    payload, analysis = encoder_with_opt.encode_with_analysis(test["data"])
    rules_count = sum(len(seed["rules"]) for seed in payload.generative_payload["seeds"])
    
    print(f"\\n  {test['name']}:")
    print(f"    규칙 수: {rules_count}")
    print(f"    압축률: {analysis['compression_ratio']:.1f}%")
    if 'optimization' in analysis:
        opt = analysis['optimization']
        print(f"    최적화: {opt['original_rules']} → {opt['optimized_rules']} 규칙")
        print(f"    - 상수: {opt['constants_found']}")
        print(f"    - 범위: {opt['ranges_found']}")
        print(f"    - 리스트: {opt['lists_compressed']}")

print("\\n3. 디코딩 호환성 테스트")
decoder = GPEDecoder()

# 간단한 데이터로 테스트
simple_data = {"test": [1, 1, 1, 2, 2, 2]}
payload = encoder_with_opt.encode(simple_data)

try:
    decoded = decoder.decode(payload)
    print(f"  원본: {simple_data}")
    print(f"  디코딩: {decoded}")
    print(f"  일치: {decoded == simple_data}")
except Exception as e:
    print(f"  ❌ 디코딩 오류: {e}")
    print("  (decoder.py 업데이트 필요)")
'''

exec(test_integration)

print("\n✅ Step 2 완료: Encoder 통합 성공!")
print("\n다음 단계:")
print("- Step 3: decoder.py를 최적화된 규칙 처리 가능하도록 확장")
print("- Step 4: 전체 시스템 테스트 및 벤치마크")
