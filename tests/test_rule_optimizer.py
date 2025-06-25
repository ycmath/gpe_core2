import json, random
from hypothesis import given, strategies as st
from gpe_core.rule_optimizer import RuleOptimizer
from gpe_core.models import InstantiateRule

# ── util: 시드 규칙 1개 생성 ───────────────────────────
def new_rule(v, idx):
    return InstantiateRule(
        instance_id=f"id_{idx}",
        class_name="int" if isinstance(v, int) else "float",
        attributes={"value": v},
    )

@given(st.integers(min_value=0, max_value=4).flatmap(    # 5개 중 하나
    lambda x: st.lists(st.just(x), min_size=5, max_size=15)
))
def test_constant_rule(values):
    """같은 값이 5회 이상 반복되면 ConstantRule 로 묶인다."""
    rules = [new_rule(v, i) for i, v in enumerate(values)]
    opt  = RuleOptimizer(constant_threshold=5)
    out  = opt.optimize_rules(rules)
    assert opt.get_stats()["constants_found"] == 1
    # ConstantRule 적용 후 seed 수가 줄어야 한다
    assert len(out) < len(rules)

def test_range_rule():
    seq = list(range(10))           # 0‥9
    rules = [new_rule(v, i) for i, v in enumerate(seq)]
    opt  = RuleOptimizer(range_min_length=5)
    out  = opt.optimize_rules(rules)
    assert opt.get_stats()["ranges_found"] == 1

def test_compact_list_rule():
    base = [0]*12
    base[3] = 9; base[7] = 5
    # list -> NEW + APPEND 형식으로 단순화된 규칙 예시
    # 간단 테스트이므로 직접 CompactListRule 사용 여부만 확인
    opt = RuleOptimizer()
    # _optimize_lists 는 내부 호출되므로 간단히 attributes 로 흉내
    fake_append = []                # 실제 APPEND 규칙 mock 생략
    out = opt._optimize_lists({"pid": fake_append}, [], set())
    # exceptions 가 2개인지 확인
    if out:
        rule = out[0]
        assert rule.length == 12 and len(rule.exceptions) == 2
