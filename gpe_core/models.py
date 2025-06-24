# gpe_core/models.py
# 이 파일은 GPE 프로토콜에서 사용되는 핵심 데이터 구조를 정의합니다.
# Pydantic 라이브러리를 사용하여 데이터의 형태와 타입을 강제합니다.

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# --- 규칙(Rule)의 기본 형태 정의 ---
# 모든 규칙들은 이 BaseRule을 따라야 합니다.
class BaseRule(BaseModel):
    """규칙의 기본이 되는 추상 클래스입니다."""
    op_code: str = Field(..., description="연산의 종류를 나타내는 코드 (e.g., 'repeat', 'pattern')")

# --- 구체적인 규칙들 정의 ---
class RepetitionRule(BaseRule):
    """특허 청구항 1의 '반복 규칙'을 정의합니다."""
    op_code: str = "repeat"
    count: int = Field(..., description="반복할 횟수")
    instruction: 'BaseRule' = Field(..., description="반복해서 실행할 명령어")

class PatternRule(BaseRule):
    """특허 청구항 3의 '패턴 생성 규칙'을 정의합니다."""
    op_code: str = "pattern"
    pattern_type: str = Field(..., description="패턴 종류 (e.g., 'circular', 'linear')")
    # 예시: radius, step, axis 등 패턴에 필요한 파라미터들...
    parameters: Dict[str, Any]

# ... 향후 ConditionalRule 등 다른 규칙들도 여기에 추가될 수 있습니다. ...


# --- GPE 페이로드의 핵심 구성 요소들 정의 ---
class AttentionSeed(BaseModel):
    """
    '어텐션 시드': 객체를 어떻게 재구성할지 알려주는 명령어(규칙)들의 집합입니다.
    이 부분이 GPE의 핵심 두뇌입니다.
    """
    library_version: str = Field(..., description="사용된 정보 조각 라이브러리의 버전")
    reconstruction_cost: Optional[str] = Field(None, description="재구성 시 예상되는 연산 복잡도 (e.g., 'low', 'high')")
    checksum: Optional[str] = Field(None, description="시드 데이터의 무결성을 검증하기 위한 해시값")
    rules: List[BaseRule] = Field(..., description="실행할 규칙들의 리스트")

class FallbackPayload(BaseModel):
    """GPE를 이해하지 못하는 구형 시스템을 위한 대체 데이터입니다."""
    description: str = Field("This is a GPE payload. For a human-readable summary, please use a GPE-compatible decoder.")
    data: Optional[Dict[str, Any]] = Field(None, description="간소화된 원본 데이터의 일부")


# --- 최종 GPE 페이로드 구조 정의 (JSON의 최상위 레벨) ---
class GpePayload(BaseModel):
    """
    GPE 페이로드의 전체 구조입니다. 이 형태가 네트워크를 통해 전송됩니다.
    마치 물건을 담는 '배송 상자'와 같습니다.
    """
    payload_type: str = Field("gpe/v1.0", description="페이로드의 유형과 버전을 나타내는 식별자")
    generative_payload: AttentionSeed = Field(..., description="생성적 페이로드, 즉 '설계도(Attention Seed)'가 담기는 곳")
    fallback_payload: Optional[FallbackPayload] = Field(None, description="대체 페이로드 (하위 호환성을 위함)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="생성 시간, 출처 등 추가적인 메타데이터")

# Pydantic v2에서 재귀적 모델 참조를 위해 필요
RepetitionRule.model_rebuild()
