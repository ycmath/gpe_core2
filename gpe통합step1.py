# Step 1: 기존 GPE에 최적화 규칙 추가

import os
os.chdir('/content/gpe_core')

# ===== 1. models.py 확장 =====
print("=== Step 1: models.py 확장 ===")

models_extension = '''# 기존 models.py 끝에 추가할 내용

# ===== 최적화된 규칙 타입들 =====

@dataclass
class ConstantRule(BaseRule):
    """상수 값 규칙 - 반복되는 단순 값"""
    op_code: str = field(default="CONSTANT", init=False)
    value_id: str
    value: Any
    references: List[str] = field(default_factory=list)

@dataclass
class TemplateRule(BaseRule):
    """템플릿 규칙 - 구조는 같고 값만 다른 경우"""
    op_code: str = field(default="TEMPLATE", init=False)
    template_id: str
    structure: Dict[str, Any]
    variable_keys: List[str]  # 변하는 키들
    instances: List[Dict[str, Any]]  # 각 인스턴스의 변하는 값들만

@dataclass
class RangeRule(BaseRule):
    """범위 규칙 - 연속된 숫자나 패턴"""
    op_code: str = field(default="RANGE", init=False)
    instance_ids: List[str]
    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float] = 1

@dataclass
class CompactListRule(BaseRule):
    """압축 리스트 규칙"""
    op_code: str = field(default="COMPACT_LIST", init=False)
    parent_id: str
    length: int
    default_value: Any
    exceptions: List[Tuple[int, Any]] = field(default_factory=list)
'''

# models.py 읽기
with open('gpe_core/models.py', 'r') as f:
    models_content = f.read()

# 이미 추가되지 않았다면 추가
if 'ConstantRule' not in models_content:
    # import 추가
    models_content = models_content.replace(
        'from typing import Any, Dict, List, Optional',
        'from typing import Any, Dict, List, Optional, Tuple, Union'
    )
    
    # 끝에 새 규칙들 추가
    models_content += '\n\n' + models_extension
    
    # 저장
    with open('gpe_core/models.py', 'w') as f:
        f.write(models_content)
    print("✅ models.py 확장 완료")
else:
    print("✅ models.py는 이미 확장됨")

# ===== 2. rule_optimizer.py 생성 =====
print("\n=== Step 2: rule_optimizer.py 생성 ===")

rule_optimizer_code = '''"""Rule Optimizer - GPE 규칙 최적화"""
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json
from .models import (
    BaseRule, InstantiateRule, AppendChildRule, RepeatRule,
    ConstantRule, TemplateRule, RangeRule, CompactListRule
)

class RuleOptimizer:
    """생성된 규칙을 분석하여 최적화"""
    
    def __init__(self, 
                 constant_threshold: int = 5,
                 template_threshold: float = 0.7,
                 range_min_length: int = 5):
        self.constant_threshold = constant_threshold
        self.template_threshold = template_threshold
        self.range_min_length = range_min_length
        self.stats = {
            'original_rules': 0,
            'optimized_rules': 0,
            'constants_found': 0,
            'templates_found': 0,
            'ranges_found': 0,
            'lists_compressed': 0
        }
    
    def optimize_rules(self, rules: List[BaseRule]) -> List[BaseRule]:
        """규칙 리스트를 최적화"""
        self.stats['original_rules'] = len(rules)
        
        # 1. 규칙 분석
        analysis = self._analyze_rules(rules)
        
        # 2. 최적화 적용
        optimized = []
        processed_indices = set()
        
        # 상수 최적화
        constant_rules = self._optimize_constants(analysis['new_rules'], processed_indices)
        optimized.extend(constant_rules)
        
        # 범위 최적화
        range_rules = self._optimize_ranges(analysis['new_rules'], processed_indices)
        optimized.extend(range_rules)
        
        # 리스트 압축
        list_rules = self._optimize_lists(analysis['append_groups'], rules, processed_indices)
        optimized.extend(list_rules)
        
        # 템플릿 최적화 (가장 복잡하므로 마지막)
        template_rules = self._optimize_templates(analysis['dict_rules'], rules, processed_indices)
        optimized.extend(template_rules)
        
        # 처리되지 않은 규칙들 추가
        for i, rule in enumerate(rules):
            if i not in processed_indices:
                optimized.append(rule)
        
        self.stats['optimized_rules'] = len(optimized)
        return optimized
    
    def _analyze_rules(self, rules: List[BaseRule]) -> Dict:
        """규칙 패턴 분석"""
        analysis = {
            'new_rules': [],
            'dict_rules': [],
            'append_groups': defaultdict(list),
            'repeat_rules': []
        }
        
        for i, rule in enumerate(rules):
            if isinstance(rule, InstantiateRule):
                analysis['new_rules'].append((i, rule))
                if rule.class_name == 'dict':
                    analysis['dict_rules'].append((i, rule))
            elif isinstance(rule, AppendChildRule):
                analysis['append_groups'][rule.parent_id].append((i, rule))
            elif isinstance(rule, RepeatRule):
                analysis['repeat_rules'].append((i, rule))
        
        return analysis
    
    def _optimize_constants(self, new_rules: List[Tuple[int, InstantiateRule]], 
                          processed: Set[int]) -> List[ConstantRule]:
        """상수 값 최적화"""
        constant_rules = []
        
        # 값별로 그룹화
        value_groups = defaultdict(list)
        for idx, rule in new_rules:
            if 'value' in rule.attributes and rule.attributes['value'] is not None:
                # JSON으로 정규화하여 비교
                value_key = json.dumps(rule.attributes['value'], sort_keys=True)
                value_groups[value_key].append((idx, rule))
        
        # 충분히 반복되는 값들을 상수로
        for value_key, group in value_groups.items():
            if len(group) >= self.constant_threshold:
                value = json.loads(value_key)
                const_rule = ConstantRule(
                    value_id=f"const_{abs(hash(value_key)) % 10000000:08x}",
                    value=value,
                    references=[rule.instance_id for _, rule in group]
                )
                constant_rules.append(const_rule)
                processed.update(idx for idx, _ in group)
                self.stats['constants_found'] += 1
        
        return constant_rules
    
    def _optimize_ranges(self, new_rules: List[Tuple[int, InstantiateRule]], 
                        processed: Set[int]) -> List[RangeRule]:
        """연속 숫자 패턴 최적화"""
        range_rules = []
        
        # 숫자 값 추출
        number_rules = []
        for idx, rule in new_rules:
            if idx not in processed and 'value' in rule.attributes:
                value = rule.attributes['value']
                if isinstance(value, (int, float)):
                    number_rules.append((idx, rule, value))
        
        if len(number_rules) < self.range_min_length:
            return range_rules
        
        # 값으로 정렬
        number_rules.sort(key=lambda x: x[2])
        
        # 연속 구간 찾기
        i = 0
        while i < len(number_rules):
            j = i + 1
            current_range = [number_rules[i]]
            
            # 연속된 값 찾기
            while j < len(number_rules):
                expected = number_rules[i][2] + (j - i)
                if number_rules[j][2] == expected:
                    current_range.append(number_rules[j])
                    j += 1
                else:
                    break
            
            # 충분히 긴 범위면 규칙 생성
            if len(current_range) >= self.range_min_length:
                range_rule = RangeRule(
                    instance_ids=[r[1].instance_id for r in current_range],
                    start=current_range[0][2],
                    end=current_range[-1][2],
                    step=1
                )
                range_rules.append(range_rule)
                processed.update(r[0] for r in current_range)
                self.stats['ranges_found'] += 1
            
            i = j
        
        return range_rules
    
    def _optimize_lists(self, append_groups: Dict[str, List[Tuple[int, AppendChildRule]]], 
                       all_rules: List[BaseRule],
                       processed: Set[int]) -> List[CompactListRule]:
        """리스트 압축 최적화"""
        list_rules = []
        
        # 자식이 많은 리스트 찾기
        for parent_id, appends in append_groups.items():
            if len(appends) < 5:  # 너무 작은 리스트는 스킵
                continue
            
            # 자식들의 값 수집
            child_values = []
            child_indices = []
            
            for idx, append_rule in appends:
                # 자식 노드 찾기
                child_value = None
                for rule_idx, rule in enumerate(all_rules):
                    if isinstance(rule, InstantiateRule) and rule.instance_id == append_rule.child_id:
                        if 'value' in rule.attributes:
                            child_value = rule.attributes['value']
                            child_indices.append((idx, rule_idx))
                        break
                child_values.append(child_value)
            
            # 가장 흔한 값 찾기
            if child_values and any(v is not None for v in child_values):
                value_counts = defaultdict(int)
                for v in child_values:
                    if v is not None:
                        value_counts[json.dumps(v, sort_keys=True)] += 1
                
                if value_counts:
                    most_common_key = max(value_counts.items(), key=lambda x: x[1])[0]
                    most_common_value = json.loads(most_common_key)
                    
                    # 예외 수집
                    exceptions = []
                    for i, v in enumerate(child_values):
                        if v != most_common_value:
                            exceptions.append((i, v))
                    
                    # 압축이 효과적인 경우만
                    if len(exceptions) < len(child_values) * 0.5:
                        list_rule = CompactListRule(
                            parent_id=parent_id,
                            length=len(child_values),
                            default_value=most_common_value,
                            exceptions=exceptions
                        )
                        list_rules.append(list_rule)
                        
                        # 처리된 규칙들 표시
                        processed.update(idx for idx, _ in appends)
                        processed.update(rule_idx for _, rule_idx in child_indices)
                        self.stats['lists_compressed'] += 1
        
        return list_rules
    
    def _optimize_templates(self, dict_rules: List[Tuple[int, InstantiateRule]], 
                          all_rules: List[BaseRule],
                          processed: Set[int]) -> List[TemplateRule]:
        """템플릿 패턴 최적화"""
        template_rules = []
        
        # 처리되지 않은 dict 규칙들만
        unprocessed_dicts = [(idx, rule) for idx, rule in dict_rules if idx not in processed]
        
        if len(unprocessed_dicts) < 3:  # 너무 적으면 스킵
            return template_rules
        
        # 각 dict의 구조 분석
        dict_structures = []
        for idx, dict_rule in unprocessed_dicts:
            # 이 dict의 자식들 찾기
            children = []
            for i, rule in enumerate(all_rules):
                if isinstance(rule, AppendChildRule) and rule.parent_id == dict_rule.instance_id:
                    # 자식의 키와 값 찾기
                    for j, r in enumerate(all_rules):
                        if isinstance(r, InstantiateRule) and r.instance_id == rule.child_id:
                            key = r.attributes.get('key', rule.attribute_name)
                            value = r.attributes.get('value')
                            children.append({
                                'key': key,
                                'value': value,
                                'type': r.class_name,
                                'append_idx': i,
                                'new_idx': j
                            })
                            break
            
            if children:
                # 구조 시그니처 생성
                structure_sig = tuple(sorted(c['key'] for c in children))
                dict_structures.append({
                    'idx': idx,
                    'rule': dict_rule,
                    'signature': structure_sig,
                    'children': children
                })
        
        # 같은 구조끼리 그룹화
        structure_groups = defaultdict(list)
        for struct in dict_structures:
            structure_groups[struct['signature']].append(struct)
        
        # 충분히 반복되는 구조를 템플릿으로
        for sig, group in structure_groups.items():
            if len(group) >= 3:
                # 템플릿 구조 생성
                first = group[0]
                structure = {}
                variable_keys = []
                
                # 모든 인스턴스에서 같은 값인지 확인
                for child in first['children']:
                    key = child['key']
                    values = [g['children'][i]['value'] for i, g in enumerate(group) 
                             if i < len(g['children']) and g['children'][i]['key'] == key]
                    
                    if len(set(json.dumps(v, sort_keys=True) for v in values)) == 1:
                        # 모든 인스턴스에서 같은 값
                        structure[key] = values[0]
                    else:
                        # 다른 값 - 변수
                        structure[key] = None
                        variable_keys.append(key)
                
                # 각 인스턴스의 변하는 값들만 추출
                instances = []
                for g in group:
                    instance_values = {}
                    for child in g['children']:
                        if child['key'] in variable_keys:
                            instance_values[child['key']] = child['value']
                    instances.append(instance_values)
                
                # 템플릿 규칙 생성
                template_rule = TemplateRule(
                    template_id=f"tmpl_{abs(hash(sig)) % 10000000:08x}",
                    structure=structure,
                    variable_keys=variable_keys,
                    instances=instances
                )
                template_rules.append(template_rule)
                
                # 처리된 규칙들 표시
                for g in group:
                    processed.add(g['idx'])
                    for c in g['children']:
                        processed.add(c['append_idx'])
                        processed.add(c['new_idx'])
                
                self.stats['templates_found'] += 1
        
        return template_rules
    
    def get_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        return self.stats.copy()
'''

# rule_optimizer.py 생성
with open('gpe_core/rule_optimizer.py', 'w') as f:
    f.write(rule_optimizer_code)
print("✅ rule_optimizer.py 생성 완료")

# ===== 3. __init__.py 업데이트 =====
print("\n=== Step 3: __init__.py 업데이트 ===")

with open('gpe_core/__init__.py', 'r') as f:
    init_content = f.read()

# 새 모델들 추가
if 'ConstantRule' not in init_content:
    init_content = init_content.replace(
        'from .models import *',
        '''from .models import *
from .rule_optimizer import RuleOptimizer'''
    )
    
    # __all__에 추가
    init_content = init_content.replace(
        '"GpePayload",',
        '''"GpePayload",
    "ConstantRule",
    "TemplateRule", 
    "RangeRule",
    "CompactListRule",
    "RuleOptimizer",'''
    )
    
    with open('gpe_core/__init__.py', 'w') as f:
        f.write(init_content)
    print("✅ __init__.py 업데이트 완료")

# ===== 4. 테스트 =====
print("\n=== Step 4: 기본 통합 테스트 ===")

test_code = '''
# 모듈 재로드
import sys
for mod in list(sys.modules.keys()):
    if mod.startswith('gpe_core'):
        del sys.modules[mod]

# Import 테스트
try:
    from gpe_core.models import ConstantRule, TemplateRule, RangeRule, CompactListRule
    from gpe_core.rule_optimizer import RuleOptimizer
    print("✅ 새 모델과 최적화기 import 성공")
    
    # 간단한 최적화 테스트
    from gpe_core.models import InstantiateRule, AppendChildRule
    
    # 테스트 규칙들 생성
    test_rules = []
    
    # 반복되는 상수값
    for i in range(10):
        test_rules.append(InstantiateRule(
            op_code="NEW",
            class_name="str", 
            instance_id=f"const_{i}",
            attributes={"value": "active"}
        ))
    
    # 연속 숫자
    for i in range(10, 20):
        test_rules.append(InstantiateRule(
            op_code="NEW",
            class_name="int",
            instance_id=f"num_{i}",
            attributes={"value": i}
        ))
    
    print(f"\\n원본 규칙 수: {len(test_rules)}")
    
    # 최적화
    optimizer = RuleOptimizer(constant_threshold=3)
    optimized = optimizer.optimize_rules(test_rules)
    
    print(f"최적화된 규칙 수: {len(optimized)}")
    print(f"최적화 통계: {optimizer.get_stats()}")
    
    # 최적화된 규칙 타입 확인
    rule_types = {}
    for rule in optimized:
        rule_type = type(rule).__name__
        rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
    print(f"규칙 타입별 개수: {rule_types}")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
'''

exec(test_code)

print("\n✅ Step 1 완료: 기본 구조 확장 성공!")
print("\n다음 단계:")
print("- Step 2: 기존 encoder.py에 RuleOptimizer 통합")
print("- Step 3: 최적화된 규칙을 처리하는 decoder.py 확장")
print("- Step 4: 전체 통합 테스트")
