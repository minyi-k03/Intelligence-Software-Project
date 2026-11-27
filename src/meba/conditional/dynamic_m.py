"""동적 압축량 — 조건부 부품, 전략 ③ (SPEC §6, 담당 C).

질문에 고유명사 덩어리가 하나 이하면 한 문단만 보면 되는 질문으로 보고 남길 문장 창 수(m)를
절반으로 줄인다. 둘 이상이면 여러 문단을 이어야 할 수 있어 기본 m을 그대로 둔다.
질문 글자만 보고 정하며 문항 유형은 쓰지 않는다. 모델을 부르지 않는다.

개발 세트 분석(실험 기록 `docs/experiments/2026-11-26-c-conditional.md`): 고유명사 덩어리가 하나 이하인
검색 문항 20개는 m을 7에서 3으로 줄여도 근거에 남는 정답 문자열·근거 문장 수가 같았고,
근거 텍스트는 절반 가까이 줄었다.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.common.types import PipelineResult
from src.config.settings import Settings
from src.meba.graph import ConditionalHooks, MComponents, build_m


def proper_noun_spans(question: str) -> int:
    """질문 속 고유명사 덩어리 수를 센다 (첫 단어를 뺀, 대문자로 시작하는 연속 단어 묶음).

    Args:
        question: 질문 원문

    Returns:
        덩어리 수. 예: "Who founded the company that makes the Model S?" → 1 ("Model S")
    """
    spans = 0
    inside = False
    for position, word in enumerate(question.split()):
        capital = position > 0 and word[:1].isupper()
        if capital and not inside:
            spans += 1
        inside = capital
    return spans


def choose_m(question: str, m: int) -> int:
    """이번 압축에서 남길 문장 창 수 (`ConditionalHooks.choose_m`).

    Args:
        question: 질문 원문
        m: 설정의 기본 `compress.m`

    Returns:
        고유명사 덩어리가 하나 이하면 `max(1, m // 2)`, 아니면 `m`
    """
    return max(1, m // 2) if proper_noun_spans(question) <= 1 else m


def build_m_dynamic_m(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """M + 동적 압축량 파이프라인 (`scripts/run_eval.py`의 `m_dynamic_m`).

    Args:
        settings: 설정. 없으면 기본 설정

    Returns:
        `(qid, question) -> PipelineResult`
    """
    return build_m(
        settings,
        components=MComponents(dynamic_m=True),
        hooks=ConditionalHooks(choose_m=choose_m),
    )
