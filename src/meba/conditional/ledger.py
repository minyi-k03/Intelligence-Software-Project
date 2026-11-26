"""근거 장부 상세화 — 조건부 부품, 전략 ③ (SPEC §6, 담당 C).

근거 판정 호출 안에서 질문에 필요한 정보 항목을 FOUND / MISSING으로 함께 받고(판정 스키마의
`ledger` 칸, `src/meba/judge.py`), RETRIEVE 뒤에는 MISSING 항목만 후속 검색어로 쓴다.
판정 호출 수는 늘지 않는다.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.common.types import PipelineResult
from src.config.settings import Settings
from src.meba.graph import ConditionalHooks, MComponents, build_m

# `Judgment.missing`에서 MISSING 항목을 잇는 구분자 (`src/meba/judge.py`의 `parse_judgment`와 같다).
_ITEM_SEPARATOR = ";"


def next_missing_item(question: str, missing: str) -> str:
    """빠진 정보에서 다음에 찾을 항목 하나를 고른다 (`ConditionalHooks.ledger`).

    장부를 켜면 판정의 빠진 정보가 MISSING 항목들을 `;`로 이은 문자열이다. 한 번의 재검색은
    한 항목에 집중하도록 첫 항목만 쓴다. 다음 RETRIEVE에서 남은 항목이 다시 올라온다.

    Args:
        question: 질문 원문 (쓰지 않는다. 훅 시그니처를 맞추기 위한 인자)
        missing: 판정이 낸 빠진 정보

    Returns:
        후속 검색어 재료. 항목이 없으면 받은 문자열 그대로
    """
    _ = question
    first = missing.split(_ITEM_SEPARATOR, 1)[0].strip()
    return first or missing


def build_m_ledger(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """M + 근거 장부 상세화 파이프라인 (`scripts/run_eval.py`의 `m_ledger`).

    Args:
        settings: 설정. 없으면 기본 설정

    Returns:
        `(qid, question) -> PipelineResult`
    """
    return build_m(
        settings,
        components=MComponents(ledger=True),
        hooks=ConditionalHooks(ledger=next_missing_item),
    )
