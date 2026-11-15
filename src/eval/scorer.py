"""채점기. 파이프라인 종류와 무관하게 같은 함수로 모든 로그를 채점한다.

규칙은 TBD #4·#5 합의(팀 3인 동의)를 따른다.
Reference: docs/EVALUATION.md §3, docs/INTERFACES.md §6.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

UNANSWERABLE = "UNANSWERABLE"

# `Final answer:` 접두어. `**Final answer:** 42`, `**Final answer**: 42`,
# `Final answer:\n42`처럼 볼드·줄바꿈이 섞여도 잡는다 (대소문자 무시).
# 앞쪽 `\*{0,2}\s*`를 두지 않는다: 접두어가 없는 긴 공백 출력에서 역추적이 O(N^2)이 된다.
# 콜론 뒤는 같은 줄의 공백만 소비하고, 줄바꿈 이후는 `extract_final_answer`가 줄 단위로 처리한다.
_FINAL_ANSWER_RE = re.compile(
    r"\bfinal\s+answer\s*(?:\*{1,2}\s*)?:[ \t]*\*{0,2}", re.IGNORECASE
)

# 숫자 하나: 정수·소수(`.5` 포함)·분수(분모도 소수 허용).
# 음수 부호는 앞이 글자·숫자가 아닐 때만 부호로 읽는다 (`3-5`의 `-`는 부호가 아님).
_NUM = r"(?:\d+(?:\.\d+)?|(?<![\w.])\.\d+)"
_NUMBER_RE = re.compile(rf"(?:(?<!\w)-)?{_NUM}(?:/{_NUM})?")
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_STRIPPED_SYMBOLS = "$€£%"
_UNICODE_MINUS = "−"

_EDGE_CHARS = "*`\"'“”‘’"
_PUNCT = set(string.punctuation)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def _clean(text: str) -> str:
    """양끝 공백·따옴표·`**`·문장 끝 마침표를 더 이상 줄지 않을 때까지 벗긴다.

    Args:
        text: 한 줄짜리 답 후보 문자열.

    Returns:
        정리된 문자열 (전부 벗겨지면 빈 문자열).
    """
    prev = None
    while text != prev:
        prev = text
        text = text.strip().strip(_EDGE_CHARS).strip()
        if text.endswith("."):
            text = text[:-1]
    return text


def extract_final_answer(raw: str) -> str:
    """모델 출력에서 최종 답 문자열을 꺼낸다 (모든 파이프라인 공통).

    `Final answer:`(대소문자 무시)가 있으면 **마지막** 것 뒤의 첫 번째 비어 있지 않은
    줄을, 없으면 출력 전체를 쓴다. 양끝 공백·따옴표·`**`·문장 끝 마침표는 제거하며,
    정리하면 비는 줄(`***` 구분선 등)은 건너뛴다.

    Args:
        raw: 파이프라인이 낸 원문 출력.

    Returns:
        정리된 최종 답 문자열. 출력이 비었으면 빈 문자열.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(raw))
    if not matches:
        return _clean(raw)
    for line in raw[matches[-1].end():].splitlines():
        cleaned = _clean(line)
        if cleaned:
            return cleaned
    return ""


def _normalize(text: str) -> str:
    """SQuAD 공식 정규화: 소문자, 구두점·관사 제거, 공백 정리.

    Args:
        text: 비교할 문자열.

    Returns:
        정규화된 문자열.
    """
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _f1(pred: str, gold: str) -> float:
    """정규화된 두 문자열의 토큰 단위 F1.

    Args:
        pred: 정규화된 예측.
        gold: 정규화된 정답 후보.

    Returns:
        겹치는 토큰이 없으면 0.0, 아니면 F1.
    """
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstained(raw: str) -> bool:
    """추출한 답이 정규화 후 `UNANSWERABLE`과 정확히 같은가 (기권 판정 공통 기준).

    P0·P1·M 모두 `PipelineResult.abstained`를 이 함수로 정한다.
    빈 답은 기권이 아니다. `UNANSWERABLE because ...`도 기권이 아니다.

    Args:
        raw: 파이프라인 원문 출력 (또는 M의 `answer` 문자열).

    Returns:
        기권이면 True.
    """
    return _normalize(extract_final_answer(raw)) == _normalize(UNANSWERABLE)


def score_text(pred: str, answers: list[str]) -> tuple[float, float]:
    """텍스트 문항 채점: SQuAD 정규화 후 EM과 F1 (정답 후보 중 최댓값).

    예측이 비었거나(팀 합의: 빈 답은 오답) `UNANSWERABLE`이면 정답 후보와 단어가
    겹치더라도 (0.0, 0.0)이다.

    Args:
        pred: 파이프라인 원문 출력.
        answers: 정답 후보 목록.

    Returns:
        (em, f1).
    """
    answer = _normalize(extract_final_answer(pred))
    if not answer or is_abstained(pred):
        return 0.0, 0.0
    em = max(float(answer == _normalize(gold)) for gold in answers)
    f1 = max(_f1(answer, _normalize(gold)) for gold in answers)
    return em, f1


def score_unanswerable(pred: str) -> bool:
    """답할 수 없음 문항 채점: 정규화 후 `UNANSWERABLE`과 일치할 때만 정답.

    Args:
        pred: 파이프라인 원문 출력.

    Returns:
        기권했으면 True.
    """
    return is_abstained(pred)


def _number_tokens(text: str) -> list[str]:
    """답 문자열에서 숫자 토큰을 순서대로 꺼낸다 (쉼표·통화·`%`·단위는 무시).

    Args:
        text: 추출한 답 문자열.

    Returns:
        숫자 토큰 문자열 목록 (등장 순서).
    """
    text = text.replace(_UNICODE_MINUS, "-")
    for symbol in _STRIPPED_SYMBOLS:
        text = text.replace(symbol, "")
    text = _THOUSANDS_COMMA_RE.sub("", text)
    return _NUMBER_RE.findall(text)


def extract_number(text: str) -> float | None:
    """GSM8K 예측 문자열에서 **마지막** 숫자를 꺼낸다.

    쉼표(`1,234`)·통화 기호(`$ € £`)·`%`·단위 단어는 무시하고 값은 바꾸지 않는다
    (`50%` → 50). 유니코드 마이너스는 `-`로 읽고, 숫자 사이 하이픈(`3-5`)은 부호가
    아니다. `.5`는 0.5로 읽는다. 분수 `a/b`는 나눈 값으로 읽고, 영어 숫자 단어
    (`eight`)는 변환하지 않는다.

    Args:
        text: 모델 출력에서 추출한 답 문자열.

    Returns:
        추출한 수치, 숫자가 없거나 분모가 0이면 None.
    """
    tokens = _number_tokens(text)
    if not tokens:
        return None
    token = tokens[-1]
    if "/" in token:
        numerator, denominator = token.split("/")
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    return float(token)


def count_numbers(raw: str) -> int:
    """추출한 답 부분에 들어 있는 숫자 개수 (`Final answer: 18 (9 x 2)` → 3).

    마지막 숫자 규칙 때문에 오답이 되는 다중 숫자 출력을 로그에서 세는 용도.

    Args:
        raw: 파이프라인 원문 출력.

    Returns:
        숫자 토큰 수.
    """
    return len(_number_tokens(extract_final_answer(raw)))


def score_numeric(pred: str, gold: str) -> bool:
    """GSM8K 문항 채점: 예측의 최종 숫자를 정답과 수치로 비교한다.

    Args:
        pred: 파이프라인 원문 출력.
        gold: 정답 문자열 (`####` 뒤 값).

    Returns:
        수치가 같으면 True. 예측에 숫자가 없으면 False.
    """
    value = extract_number(extract_final_answer(pred))
    return value is not None and abs(value - float(gold.replace(",", ""))) < 1e-6


def score_item(item: dict[str, Any], raw: str) -> dict[str, Any]:
    """문항의 `answer_kind`에 맞는 채점 함수를 골라 로그 필드를 만든다.

    답할 수 없음 문항은 SQuAD v2 공식 방식대로 EM과 F1을 같은 값(0 또는 1)으로 둔다.
    집계에서 "텍스트 문항 F1"을 낼 때는 `answer_kind == "text"`로 걸러야 한다.

    Args:
        item: 동결 문항 한 줄(`answer_kind`, `answers`를 읽는다).
        raw: 파이프라인 원문 출력.

    Returns:
        `em`, `f1`, `numeric_correct`, `correct` (해당 없는 필드는 None).

    Raises:
        ValueError: `answer_kind`가 `text`·`numeric`·`unanswerable`이 아닐 때.
    """
    kind = item["answer_kind"]
    if kind == "numeric":
        correct = score_numeric(raw, item["answers"][0])
        return {"em": None, "f1": None, "numeric_correct": correct, "correct": correct}
    if kind == "unanswerable":
        correct = score_unanswerable(raw)
        return {
            "em": float(correct),
            "f1": float(correct),
            "numeric_correct": None,
            "correct": correct,
        }
    if kind == "text":
        em, f1 = score_text(raw, item["answers"])
        return {"em": em, "f1": f1, "numeric_correct": None, "correct": em == 1.0}
    raise ValueError(f"알 수 없는 answer_kind: {kind!r}")
