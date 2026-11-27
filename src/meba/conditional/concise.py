"""조건부 간결 출력: 핵심 근거와 수학 풀이 단계를 남기며 중복 서술을 줄인다.

모델 입력의 system 메시지만 보강한다. 문항 유형은 받지 않고, 그래프가 이미 정한
호출 단계만 사용한다. 출력 상한과 공통 답 형식 지시문은 바꾸지 않는다.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.common.types import PipelineResult
from src.config.settings import Settings


_MATH_GUIDANCE = (
    "For the math route, show each calculation and intermediate value needed to verify "
    "the answer. Keep the steps in order. Do not restate the question or repeat a "
    "calculation or explanation already given. Keep the required final-answer line."
)
_JUDGE_GUIDANCE = (
    "In reasoning, state only the decisive evidence or the specific missing fact. "
    "Do not restate the question, repeat evidence already checked, or add unrelated "
    "explanations. Keep every required JSON field and its order."
)


def concise_messages(step: str, model_input: Any) -> Any:
    """모델 호출 단계의 system 메시지에 짧은 서술 지시를 더한다.

    Args:
        step: 그래프의 모델 호출 단계(`router`, `finalize`, `judge`).
        model_input: router/finalize는 메시지 사전 목록, judge는 `(role, content)` 목록.

    Returns:
        같은 형식의 새 메시지 목록. 마무리 호출과 알 수 없는 단계는 입력 그대로.

    Raises:
        TypeError: 알려진 단계의 메시지 형식이 인터페이스와 다를 때.
    """
    if step == "finalize" or step not in {"router", "judge"}:
        return model_input
    if not isinstance(model_input, list):
        raise TypeError(f"{step} model input must be a list of messages")

    guidance = _MATH_GUIDANCE if step == "router" else _JUDGE_GUIDANCE
    rewritten: list[Any] = []
    found_system = False
    for message in model_input:
        if step == "router":
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise TypeError("router messages must be dictionaries with string content")
            if message.get("role") == "system":
                rewritten.append({**message, "content": f"{message['content']}\n\n{guidance}"})
                found_system = True
            else:
                rewritten.append(dict(message))
        else:
            if not isinstance(message, tuple) or len(message) != 2 or not isinstance(message[1], str):
                raise TypeError("judge messages must be (role, content) pairs")
            role, content = message
            rewritten.append((role, f"{content}\n\n{guidance}" if role == "system" else content))
            found_system |= role == "system"
    if not found_system:
        raise ValueError(f"{step} model input has no system message")
    return rewritten


def build_m_concise(
    settings: Optional[Settings] = None,
    *,
    llm_factory: Optional[Callable[..., Any]] = None,
) -> Callable[[str, str], PipelineResult]:
    """기본 M과 같은 설정으로 간결 출력 훅만 켠 파이프라인을 만든다.

    Args:
        settings: 공통 실행 설정. 없으면 M의 기본 설정을 읽는다.
        llm_factory: 개발 실험에서 종료 사유·원문을 관찰할 때 쓰는 ChatOllama 생성 함수.

    Returns:
        `(qid, question) -> PipelineResult` 함수.
    """
    from src.meba.graph import ConditionalHooks, MComponents, build_m
    from src.config.llm import create_chat_ollama

    return build_m(
        settings,
        components=MComponents(concise=True),
        hooks=ConditionalHooks(concise=concise_messages),
        llm_factory=llm_factory or create_chat_ollama,
    )
