
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Set, Union
from collections import defaultdict
import json, hashlib, logging

from .models import (
    BaseRule, InstantiateRule, AppendChildRule, RepeatRule,
    ConstantRule, TemplateRule, RangeRule, CompactListRule,
)

# ── logger (무소음 기본) ───────────────────────────────
LOGGER = logging.getLogger("gpe.rule_optimizer")
LOGGER.addHandler(logging.NullHandler())

class RuleOptimizer:
    def __init__(
        self,
        constant_threshold: int = 5,
        template_threshold: float = 0.7,
        range_min_length: int = 5,
    ):
        self.constant_threshold = constant_threshold
        self.template_threshold = template_threshold
        self.range_min_length = range_min_length
        self.stats = {
            "original_rules": 0,
            "optimized_rules": 0,
            "constants_found": 0,
            "templates_found": 0,
            "ranges_found": 0,
            "lists_compressed": 0,
        }

    # ─────────────────────────────────────────────────────────────
    def optimize_rules(self, rules: List[BaseRule]) -> List[BaseRule]:
        self.stats["original_rules"] = len(rules)
        analysis = self._analyze_rules(rules)
        processed: Set[int] = set()
        out: List[BaseRule] = []

        out += self._optimize_constants(analysis["new_rules"], processed)
        out += self._optimize_ranges(analysis["new_rules"], processed)
        out += self._optimize_lists(analysis["append_groups"], rules, processed)
        out += self._optimize_templates(analysis["dict_rules"], rules, processed)

        # 미처리 규칙 그대로 보존
        out += [r for i, r in enumerate(rules) if i not in processed]
        self.stats["optimized_rules"] = len(out)
        return out
        
    # ── 공통 유틸 ───────────────────────────────────────
    def _digest8(self, obj: Any) -> str:
        """환경 독립적 8-char ID (MD5 of JSON-normalized obj)"""
        md5 = hashlib.md5(json.dumps(obj, sort_keys=True).encode())
        return md5.hexdigest()[:8]
    
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
                try:
                    value_key = json.dumps(rule.attributes['value'], sort_keys=True)
                    value_groups[value_key].append((idx, rule))
                except TypeError:
                    # hash 불가(예: set) → 상수 최적화 건너뜀
                    LOGGER.debug("Constant-scan skip (unhashable): %s", rule.instance_id)
        
        # 충분히 반복되는 값들을 상수로
        for value_key, group in value_groups.items():
            if len(group) >= self.constant_threshold:
                value = json.loads(value_key)
                const_rule = ConstantRule(
                    value_id=f"const_{self._digest8(value)}",
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
                # key 순서 무시: frozenset 으로 시그니처 고정
                structure_sig = frozenset(c['key'] for c in children)
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
                    template_id=f"tmpl_{self._digest8(tuple(sorted(sig)))}",
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
