# GPE Rule Optimizer - 원래 설계 + 스마트 규칙 최적화

from typing import Any, Dict, List, Set, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import hashlib

# ===== 1. 확장된 규칙 타입 정의 =====

@dataclass
class BaseRule:
    op_code: str

# 기존 규칙들
@dataclass
class InstantiateRule(BaseRule):
    class_name: str
    instance_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AppendChildRule(BaseRule):
    parent_id: str
    child_id: str
    attribute_name: str = "children"

@dataclass
class RepeatRule(BaseRule):
    count: int
    instruction: Union[BaseRule, List[BaseRule]]

# ===== 새로운 최적화된 규칙들 =====

@dataclass
class ConstantRule(BaseRule):
    """상수 값 규칙 - 반복되는 단순 값"""
    op_code: str = "CONSTANT"
    value_id: str = ""
    value: Any = None
    references: List[str] = field(default_factory=list)  # 이 값을 참조하는 노드들

@dataclass
class TemplateRule(BaseRule):
    """템플릿 규칙 - 구조는 같고 일부 값만 다른 경우"""
    op_code: str = "TEMPLATE"
    template_id: str = ""
    structure: Dict[str, Any] = field(default_factory=dict)  # 구조 템플릿
    instances: List[Dict[str, Any]] = field(default_factory=list)  # 변하는 값들

@dataclass
class RangeRule(BaseRule):
    """범위 규칙 - 연속된 숫자나 패턴"""
    op_code: str = "RANGE"
    start: Any = None
    end: Any = None
    step: Any = 1
    pattern: Optional[str] = None  # "item_{i}" 같은 패턴

@dataclass
class CompactListRule(BaseRule):
    """압축 리스트 규칙 - 동일 요소의 효율적 표현"""
    op_code: str = "COMPACT_LIST"
    length: int = 0
    default_value: Any = None
    exceptions: List[Tuple[int, Any]] = field(default_factory=list)  # (index, value)

# ===== 2. 규칙 최적화기 =====

class RuleOptimizer:
    """생성된 규칙을 분석하여 최적화"""
    
    def __init__(self):
        self.constant_threshold = 5  # 5번 이상 반복되면 상수로
        self.template_threshold = 0.7  # 70% 이상 구조 일치하면 템플릿
        
    def optimize_rules(self, rules: List[BaseRule], nodes: Dict[str, Any]) -> List[BaseRule]:
        """규칙 리스트를 최적화된 규칙으로 변환"""
        
        # 1. 규칙 분석
        analysis = self._analyze_rules(rules, nodes)
        
        # 2. 최적화 적용
        optimized = []
        processed = set()
        
        # 상수 최적화
        for constant in analysis['constants']:
            opt_rule = self._create_constant_rule(constant, rules)
            if opt_rule:
                optimized.append(opt_rule)
                processed.update(constant['rule_indices'])
        
        # 템플릿 최적화
        for template in analysis['templates']:
            opt_rule = self._create_template_rule(template, rules, nodes)
            if opt_rule:
                optimized.append(opt_rule)
                processed.update(template['rule_indices'])
        
        # 범위 최적화
        for range_pattern in analysis['ranges']:
            opt_rule = self._create_range_rule(range_pattern, rules)
            if opt_rule:
                optimized.append(opt_rule)
                processed.update(range_pattern['rule_indices'])
        
        # 리스트 압축
        for list_pattern in analysis['lists']:
            opt_rule = self._create_compact_list_rule(list_pattern, rules)
            if opt_rule:
                optimized.append(opt_rule)
                processed.update(list_pattern['rule_indices'])
        
        # 최적화되지 않은 규칙들 추가
        for i, rule in enumerate(rules):
            if i not in processed:
                optimized.append(rule)
        
        return optimized
    
    def _analyze_rules(self, rules: List[BaseRule], nodes: Dict[str, Any]) -> Dict:
        """규칙 패턴 분석"""
        analysis = {
            'constants': [],
            'templates': [],
            'ranges': [],
            'lists': []
        }
        
        # NEW 규칙들 분석
        new_rules = [(i, r) for i, r in enumerate(rules) if r.op_code == "NEW"]
        
        # 1. 상수 패턴 찾기
        value_groups = defaultdict(list)
        for i, rule in new_rules:
            if 'value' in rule.attributes:
                value_key = json.dumps(rule.attributes['value'], sort_keys=True)
                value_groups[value_key].append((i, rule))
        
        for value_key, group in value_groups.items():
            if len(group) >= self.constant_threshold:
                analysis['constants'].append({
                    'value': json.loads(value_key),
                    'count': len(group),
                    'rule_indices': [i for i, _ in group],
                    'instance_ids': [r.instance_id for _, r in group]
                })
        
        # 2. 템플릿 패턴 찾기
        dict_rules = [(i, r) for i, r in new_rules if r.class_name == "dict"]
        if len(dict_rules) > 3:
            templates = self._find_template_patterns(dict_rules, rules)
            analysis['templates'].extend(templates)
        
        # 3. 범위 패턴 찾기
        number_sequences = self._find_range_patterns(new_rules)
        analysis['ranges'].extend(number_sequences)
        
        # 4. 리스트 압축 패턴
        list_patterns = self._find_list_patterns(rules)
        analysis['lists'].extend(list_patterns)
        
        return analysis
    
    def _find_template_patterns(self, dict_rules: List[Tuple[int, Any]], all_rules: List[BaseRule]) -> List[Dict]:
        """딕셔너리 규칙에서 템플릿 패턴 찾기"""
        templates = []
        
        # 각 딕셔너리의 구조 추출
        structures = []
        for rule_idx, rule in dict_rules:
            # 이 딕셔너리의 자식들 찾기
            children = []
            for i, r in enumerate(all_rules):
                if r.op_code == "APPEND" and r.parent_id == rule.instance_id:
                    children.append((i, r))
            
            # 구조 시그니처 생성
            structure = {
                'rule_idx': rule_idx,
                'instance_id': rule.instance_id,
                'children': children,
                'child_keys': sorted([r.attribute_name for _, r in children if hasattr(r, 'attribute_name')])
            }
            structures.append(structure)
        
        # 유사한 구조 그룹화
        structure_groups = defaultdict(list)
        for struct in structures:
            key = json.dumps(struct['child_keys'])
            structure_groups[key].append(struct)
        
        # 충분히 반복되는 구조를 템플릿으로
        for key, group in structure_groups.items():
            if len(group) >= 3:
                templates.append({
                    'structure_key': key,
                    'count': len(group),
                    'rule_indices': [s['rule_idx'] for s in group],
                    'instances': group
                })
        
        return templates
    
    def _find_range_patterns(self, new_rules: List[Tuple[int, Any]]) -> List[Dict]:
        """연속된 숫자나 패턴 찾기"""
        ranges = []
        
        # 숫자 값들 추출
        number_rules = []
        for i, rule in new_rules:
            if 'value' in rule.attributes and isinstance(rule.attributes['value'], (int, float)):
                number_rules.append((i, rule, rule.attributes['value']))
        
        if len(number_rules) < 3:
            return ranges
        
        # 연속성 체크
        sorted_numbers = sorted(number_rules, key=lambda x: x[2])
        
        current_range = [sorted_numbers[0]]
        for i in range(1, len(sorted_numbers)):
            prev_val = sorted_numbers[i-1][2]
            curr_val = sorted_numbers[i][2]
            
            if curr_val - prev_val == 1:  # 연속
                current_range.append(sorted_numbers[i])
            else:
                if len(current_range) >= 3:
                    ranges.append({
                        'type': 'numeric',
                        'start': current_range[0][2],
                        'end': current_range[-1][2],
                        'count': len(current_range),
                        'rule_indices': [r[0] for r in current_range]
                    })
                current_range = [sorted_numbers[i]]
        
        # 마지막 범위 체크
        if len(current_range) >= 3:
            ranges.append({
                'type': 'numeric',
                'start': current_range[0][2],
                'end': current_range[-1][2],
                'count': len(current_range),
                'rule_indices': [r[0] for r in current_range]
            })
        
        return ranges
    
    def _find_list_patterns(self, rules: List[BaseRule]) -> List[Dict]:
        """리스트에서 압축 가능한 패턴 찾기"""
        patterns = []
        
        # APPEND 규칙들을 부모별로 그룹화
        append_groups = defaultdict(list)
        for i, rule in enumerate(rules):
            if rule.op_code == "APPEND" and hasattr(rule, 'parent_id'):
                append_groups[rule.parent_id].append((i, rule))
        
        # 같은 부모에 많은 자식이 추가되는 경우
        for parent_id, appends in append_groups.items():
            if len(appends) >= 5:
                # 자식들의 패턴 분석
                child_values = []
                for _, append_rule in appends:
                    # 자식 노드의 값 찾기
                    for r in rules:
                        if r.op_code == "NEW" and r.instance_id == append_rule.child_id:
                            if 'value' in r.attributes:
                                child_values.append(r.attributes['value'])
                            break
                
                # 가장 흔한 값 찾기
                if child_values:
                    value_counts = defaultdict(int)
                    for v in child_values:
                        value_counts[json.dumps(v, sort_keys=True)] += 1
                    
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    if most_common[1] > len(child_values) * 0.5:  # 50% 이상
                        patterns.append({
                            'parent_id': parent_id,
                            'total': len(appends),
                            'default_value': json.loads(most_common[0]),
                            'default_count': most_common[1],
                            'rule_indices': [i for i, _ in appends]
                        })
        
        return patterns
    
    def _create_constant_rule(self, constant_info: Dict, rules: List[BaseRule]) -> Optional[ConstantRule]:
        """상수 규칙 생성"""
        return ConstantRule(
            value_id=f"const_{hash(json.dumps(constant_info['value']))[:8]}",
            value=constant_info['value'],
            references=constant_info['instance_ids']
        )
    
    def _create_template_rule(self, template_info: Dict, rules: List[BaseRule], nodes: Dict) -> Optional[TemplateRule]:
        """템플릿 규칙 생성"""
        # 구조 템플릿 생성
        structure = json.loads(template_info['structure_key'])
        
        # 각 인스턴스의 값들 추출
        instances = []
        for inst in template_info['instances']:
            values = {}
            # 구현 생략 - 실제 값 추출 로직
            instances.append(values)
        
        return TemplateRule(
            template_id=f"tmpl_{hash(template_info['structure_key'])[:8]}",
            structure={k: None for k in structure},
            instances=instances
        )
    
    def _create_range_rule(self, range_info: Dict, rules: List[BaseRule]) -> Optional[RangeRule]:
        """범위 규칙 생성"""
        return RangeRule(
            start=range_info['start'],
            end=range_info['end'],
            step=1
        )
    
    def _create_compact_list_rule(self, list_info: Dict, rules: List[BaseRule]) -> Optional[CompactListRule]:
        """압축 리스트 규칙 생성"""
        exceptions = []
        # 기본값과 다른 값들의 인덱스 수집
        # 구현 생략
        
        return CompactListRule(
            length=list_info['total'],
            default_value=list_info['default_value'],
            exceptions=exceptions
        )

# ===== 3. 개선된 인코더/디코더 =====

class OptimizedGPEEncoder:
    """규칙 최적화를 포함한 GPE 인코더"""
    
    def __init__(self):
        self.rule_optimizer = RuleOptimizer()
        # 기존 컴포넌트들
        from gpe_core.ast_builder import ASTBuilder
        from gpe_core.repetition_detector import RepetitionDetector
        from gpe_core.seed_generator import SeedGenerator
        self.ast_builder = ASTBuilder
        self.repetition_detector = RepetitionDetector
        self.seed_generator = SeedGenerator
    
    def encode(self, data: Any) -> Dict:
        # 1. 기존 GPE 프로세스
        builder = self.ast_builder()
        root_id = builder.build(data)
        
        # 2. 반복 감지 (파라미터 조정)
        detector = self.repetition_detector(
            builder.nodes,
            min_occ=2,  # 낮춰서 더 많은 패턴 감지
            min_size=1  # 작은 패턴도 감지
        )
        repetitions = detector.detect()
        
        # 3. 시드 생성
        generator = self.seed_generator(builder.nodes, repetitions)
        seeds = generator.generate(root_id)
        
        # 4. 규칙 추출 및 최적화
        all_rules = []
        for seed in seeds:
            all_rules.extend(seed.rules)
        
        # 5. 규칙 최적화 ⭐ 핵심 개선
        optimized_rules = self.rule_optimizer.optimize_rules(all_rules, builder.nodes)
        
        # 6. 최적화 통계
        stats = self._calculate_optimization_stats(all_rules, optimized_rules)
        
        return {
            "version": "gpe-optimized-1.0",
            "root_id": root_id,
            "rules": [self._rule_to_dict(r) for r in optimized_rules],
            "optimization": stats,
            "node_count": len(builder.nodes)
        }
    
    def _rule_to_dict(self, rule: BaseRule) -> Dict:
        """규칙을 딕셔너리로 변환"""
        if isinstance(rule, ConstantRule):
            return {
                "op_code": "CONSTANT",
                "value_id": rule.value_id,
                "value": rule.value,
                "refs": rule.references
            }
        elif isinstance(rule, TemplateRule):
            return {
                "op_code": "TEMPLATE",
                "template_id": rule.template_id,
                "structure": rule.structure,
                "instances": rule.instances
            }
        elif isinstance(rule, RangeRule):
            return {
                "op_code": "RANGE",
                "start": rule.start,
                "end": rule.end,
                "step": rule.step
            }
        elif isinstance(rule, CompactListRule):
            return {
                "op_code": "COMPACT_LIST",
                "length": rule.length,
                "default": rule.default_value,
                "exceptions": rule.exceptions
            }
        else:
            # 기존 규칙들
            return rule.__dict__
    
    def _calculate_optimization_stats(self, original: List[BaseRule], optimized: List[BaseRule]) -> Dict:
        """최적화 통계 계산"""
        return {
            "original_rules": len(original),
            "optimized_rules": len(optimized),
            "reduction": f"{(1 - len(optimized)/len(original))*100:.1f}%",
            "constant_rules": sum(1 for r in optimized if isinstance(r, ConstantRule)),
            "template_rules": sum(1 for r in optimized if isinstance(r, TemplateRule)),
            "range_rules": sum(1 for r in optimized if isinstance(r, RangeRule)),
            "list_rules": sum(1 for r in optimized if isinstance(r, CompactListRule))
        }

class OptimizedGPEDecoder:
    """최적화된 규칙을 처리하는 디코더"""
    
    def decode(self, payload: Dict) -> Any:
        rules = payload["rules"]
        objects = {}
        
        # 각 규칙 타입별 처리
        for rule in rules:
            op_code = rule["op_code"]
            
            if op_code == "CONSTANT":
                # 상수 규칙: 여러 객체에 동일 값 할당
                for ref in rule["refs"]:
                    objects[ref] = rule["value"]
                    
            elif op_code == "TEMPLATE":
                # 템플릿 규칙: 구조 + 인스턴스 값 결합
                for i, instance in enumerate(rule["instances"]):
                    obj = rule["structure"].copy()
                    obj.update(instance)
                    # ID 생성 로직 필요
                    objects[f"tmpl_inst_{i}"] = obj
                    
            elif op_code == "RANGE":
                # 범위 규칙: 연속 값 생성
                for i, val in enumerate(range(rule["start"], rule["end"] + 1, rule["step"])):
                    objects[f"range_{i}"] = val
                    
            elif op_code == "COMPACT_LIST":
                # 압축 리스트: 기본값 + 예외
                lst = [rule["default"]] * rule["length"]
                for idx, val in rule["exceptions"]:
                    lst[idx] = val
                objects[f"list_{hash(str(rule))[:8]}"] = lst
                
            else:
                # 기존 규칙 처리 (NEW, APPEND, REPEAT)
                self._process_standard_rule(rule, objects)
        
        return objects.get(payload["root_id"])
    
    def _process_standard_rule(self, rule: Dict, objects: Dict):
        """기존 GPE 규칙 처리"""
        # 기존 디코더 로직 사용
        pass

# ===== 4. 테스트 및 비교 =====

def test_rule_optimization():
    print("=== GPE Rule Optimization Test ===\n")
    
    # 다양한 패턴의 테스트 데이터
    test_cases = [
        {
            "name": "Constant repetition",
            "data": {
                "status": "active",
                "items": [
                    {"id": i, "status": "active", "type": "item"}
                    for i in range(20)
                ]
            }
        },
        {
            "name": "Range pattern",
            "data": {
                "numbers": list(range(1, 101)),
                "ids": [f"ID_{i:04d}" for i in range(50)]
            }
        },
        {
            "name": "Template pattern",
            "data": [
                {
                    "id": i,
                    "name": f"User {i}",
                    "role": "user",
                    "active": True,
                    "permissions": ["read", "write"]
                }
                for i in range(30)
            ]
        },
        {
            "name": "Mixed patterns",
            "data": {
                "config": {"version": "1.0", "debug": False},
                "users": [{"id": i, "type": "user"} for i in range(10)],
                "items": [{"id": i, "type": "item"} for i in range(10)],
                "constants": ["active"] * 20
            }
        }
    ]
    
    # 규칙 최적화 테스트
    optimizer = RuleOptimizer()
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        
        # 간단한 규칙 생성 시뮬레이션
        rules = []
        
        # 데이터에서 규칙 추출 (시뮬레이션)
        if isinstance(test['data'], dict):
            for key, value in test['data'].items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, (int, str)):
                            rules.append(InstantiateRule(
                                op_code="NEW",
                                class_name=type(item).__name__,
                                instance_id=f"{key}_{i}",
                                attributes={"value": item}
                            ))
                        elif isinstance(item, dict):
                            rules.append(InstantiateRule(
                                op_code="NEW",
                                class_name="dict",
                                instance_id=f"{key}_{i}",
                                attributes={}
                            ))
        
        print(f"  Original rules: {len(rules)}")
        
        # 최적화
        optimized = optimizer.optimize_rules(rules, {})
        
        print(f"  Optimized rules: {len(optimized)}")
        
        # 최적화된 규칙 타입별 개수
        rule_types = defaultdict(int)
        for rule in optimized:
            rule_types[rule.op_code] += 1
        
        print(f"  Rule types: {dict(rule_types)}")
        
        # 압축률 계산
        original_size = len(json.dumps([r.__dict__ for r in rules]))
        optimized_size = len(json.dumps([r.__dict__ if hasattr(r, '__dict__') else str(r) for r in optimized]))
        
        print(f"  Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")

# 실행
if __name__ == "__main__":
    test_rule_optimization()
    
    # 실제 GPE와 통합 테스트
    print("\n\n=== Integrated Test ===")
    
    # 복잡한 실제 데이터
    real_data = {
        "api_version": "2.0",
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {
            "users": [
                {
                    "id": i,
                    "username": f"user_{i}",
                    "email": f"user_{i}@example.com",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z",
                    "settings": {
                        "theme": "dark",
                        "notifications": True,
                        "language": "en"
                    }
                }
                for i in range(100)
            ],
            "stats": {
                "total": 100,
                "active": 100,
                "inactive": 0
            }
        }
    }
    
    # 최적화된 인코더 테스트
    encoder = OptimizedGPEEncoder()
    result = encoder.encode(real_data)
    
    print(f"\nOptimized GPE Encoding:")
    print(f"  Node count: {result['node_count']}")
    print(f"  Total rules: {len(result['rules'])}")
    print(f"  Optimization stats: {result['optimization']}")
    
    # 규칙 타입 분석
    rule_types = defaultdict(int)
    for rule in result['rules']:
        rule_types[rule['op_code']] += 1
    
    print(f"  Rule distribution: {dict(rule_types)}")
