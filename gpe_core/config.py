"""GPE 기본 최적화 파라미터"""
OPT_CONFIG = {
    "constant_threshold": 5,    # 같은 값 ≥5 회 → ConstantRule
    "template_threshold": 0.7,  # (미사용) 구조 동일성 비율
    "range_min_length": 5,      # 연속 숫자 ≥5 개 → RangeRule
}
