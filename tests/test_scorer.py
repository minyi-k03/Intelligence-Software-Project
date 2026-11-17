"""채점기 단위 테스트 (PR A-2). 모델·GPU 없이 돈다.

TBD #4·#5 합의 규칙(최종 답 추출, GSM8K 숫자 추출, 기권 판정)을 테스트로 고정한다.
seam: src/eval/scorer.py의 공개 함수.
"""

from __future__ import annotations

import time

import pytest

from src.eval.scorer import (
    count_numbers,
    extract_final_answer,
    extract_number,
    is_abstained,
    score_item,
    score_numeric,
    score_text,
    score_unanswerable,
)


# --- extract_number -------------------------------------------------------


def test_extract_number_strips_comma_and_currency() -> None:
    """쉼표와 통화 기호를 지우고 값을 읽는다."""
    assert extract_number("The answer is $1,234.") == 1234.0


def test_extract_number_keeps_negative_sign() -> None:
    """공백 뒤의 `-`는 음수 부호다."""
    assert extract_number("The balance is -5 dollars") == -5.0


def test_extract_number_converts_fraction_to_value() -> None:
    """분수 a/b는 나눈 값이다."""
    assert extract_number("She ate 3/4 of the pizza") == 0.75


def test_extract_number_negative_fraction() -> None:
    """음수 분수도 부호를 유지한다."""
    assert extract_number("-3/4") == -0.75


def test_extract_number_returns_none_when_no_number() -> None:
    """숫자가 없으면 None."""
    assert extract_number("UNANSWERABLE") is None


def test_extract_number_does_not_crash_on_zero_denominator() -> None:
    """분모 0은 None (예외 없음)."""
    assert extract_number("1/0") is None


def test_extract_number_hyphen_between_digits_is_not_a_sign() -> None:
    """숫자 사이 하이픈은 음수 부호가 아니다."""
    assert extract_number("3-5") == 5.0
    assert extract_number("10-5") == 5.0


def test_extract_number_reads_unicode_minus_as_negative() -> None:
    """유니코드 마이너스를 `-`로 읽는다."""
    assert extract_number("The change is −5") == -5.0


def test_extract_number_strips_percent_and_other_currency_without_conversion() -> None:
    """`%`와 `€ £`는 지우되 값은 바꾸지 않는다."""
    assert extract_number("50%") == 50.0
    assert extract_number("€12 or £13") == 13.0


def test_extract_number_percent_before_hyphen_is_not_a_sign() -> None:
    """`%` 뒤 하이픈도 `$5-$3`과 같게 부호가 아니다."""
    assert extract_number("5%-3%") == 3.0
    assert extract_number("$5-$3") == 3.0


def test_extract_number_negative_currency_forms() -> None:
    """`-$5`와 `$-5`는 모두 -5."""
    assert extract_number("-$5") == -5.0
    assert extract_number("$-5") == -5.0


def test_extract_number_reads_decimal_without_leading_zero() -> None:
    """`.5`는 0.5, `-.5`는 -0.5다."""
    assert extract_number(".5") == 0.5
    assert extract_number("-.5") == -0.5


def test_extract_number_decimal_denominator() -> None:
    """분모가 소수인 분수도 한 수로 읽는다."""
    assert extract_number("1/2.5") == pytest.approx(0.4)


def test_extract_number_trailing_period_is_not_part_of_number() -> None:
    """문장 끝 마침표와 `3.5.`는 소수점으로 오인하지 않는다."""
    assert extract_number("The total is 3.5.") == 3.5
    assert extract_number("It is 10.") == 10.0


def test_extract_number_does_not_merge_comma_separated_list() -> None:
    """천 단위가 아닌 쉼표는 지우지 않는다."""
    assert extract_number("5, 6") == 6.0


def test_extract_number_does_not_convert_number_words() -> None:
    """영어 숫자 단어는 변환하지 않는다."""
    assert extract_number("eight") is None


def test_extract_number_takes_last_number_when_several() -> None:
    """마지막 숫자를 쓴다."""
    assert extract_number("18 (9 x 2)") == 2.0


def test_count_numbers_counts_numbers_in_answer_part_only() -> None:
    """답 부분의 숫자만 센다."""
    assert count_numbers("I computed 3 + 4.\nFinal answer: 18 (9 x 2)") == 3
    assert count_numbers("Final answer: 18") == 1


# --- extract_final_answer -------------------------------------------------


def test_extract_final_answer_uses_text_after_prefix() -> None:
    """접두어 뒤 문자열을 쓴다."""
    raw = "Let me think.\nThe film was released in 1994.\nFinal answer: Paris"
    assert extract_final_answer(raw) == "Paris"


def test_extract_final_answer_without_prefix_uses_whole_output() -> None:
    """접두어가 없으면 출력 전체를 쓴다."""
    assert extract_final_answer("Paris is the capital.\nParis\n\n") == "Paris is the capital.\nParis"


def test_extract_final_answer_empty_output_gives_empty_string() -> None:
    """빈 출력은 빈 문자열."""
    assert extract_final_answer("  \n") == ""


def test_extract_final_answer_uses_last_prefix_and_ignores_case() -> None:
    """접두어가 여러 번이면 마지막 것을 쓰고 대소문자는 무시한다."""
    raw = "final answer: wrong\nOn reflection...\nFINAL ANSWER: Paris"
    assert extract_final_answer(raw) == "Paris"


def test_extract_final_answer_accepts_bold_markers() -> None:
    """`**`가 콜론 앞뒤에 섞여도 잡는다."""
    assert extract_final_answer("**Final answer:** 42") == "42"
    assert extract_final_answer("**Final answer**: 42") == "42"


def test_extract_final_answer_takes_first_nonempty_line_after_prefix() -> None:
    """접두어 뒤 첫 번째 비어 있지 않은 줄만 쓴다."""
    assert extract_final_answer("Final answer:\n\n42\nbecause 6 x 7") == "42"


def test_extract_final_answer_skips_separator_line_after_prefix() -> None:
    """`***` 같은 구분선 줄은 건너뛰고 다음 줄을 쓴다."""
    assert extract_final_answer("Final answer:\n***\n42") == "42"


def test_extract_final_answer_strips_quotes_bold_and_trailing_period() -> None:
    """따옴표·`**`·문장 끝 마침표를 벗긴다."""
    assert extract_final_answer('Final answer: "**Paris**".') == "Paris"


def test_extract_final_answer_empty_after_prefix_is_empty() -> None:
    """접두어 뒤가 비면 빈 문자열."""
    assert extract_final_answer("Final answer:") == ""


def test_extract_final_answer_requires_word_boundary_before_prefix() -> None:
    """단어 일부(`semifinal answer:`)는 접두어가 아니다."""
    assert extract_final_answer("semifinal answer: 42") == "semifinal answer: 42"


def test_extract_final_answer_long_whitespace_output_is_fast() -> None:
    """접두어 없는 긴 공백 출력에서 정규식 역추적으로 멈추지 않는다."""
    start = time.perf_counter()
    assert extract_final_answer(" " * 50_000) == ""
    assert extract_final_answer("Final answer" + " " * 50_000) == "Final answer"
    assert time.perf_counter() - start < 1.0


# --- score_numeric --------------------------------------------------------


def test_score_numeric_compares_by_value_not_string() -> None:
    """문자열이 아니라 수치로 비교한다."""
    assert score_numeric("Let me compute.\nFinal answer: $1,234.00", "1234") is True


def test_score_numeric_distinguishes_42_from_420() -> None:
    """`42.0`은 42이고 420이 아니다."""
    assert score_numeric("Final answer: 42.0", "420") is False
    assert score_numeric("Final answer: 42.0", "42") is True


def test_score_numeric_wrong_when_no_number_in_prediction() -> None:
    """예측에 숫자가 없으면 오답."""
    assert score_numeric("UNANSWERABLE", "18") is False


# --- score_text -----------------------------------------------------------


def test_score_text_em_ignores_case_articles_and_punctuation() -> None:
    """SQuAD 정규화로 대소문자·관사·구두점을 무시한다."""
    em, _ = score_text("Final answer: The Eiffel Tower!", ["eiffel tower"])
    assert em == 1.0


def test_score_text_f1_gives_partial_credit_for_token_overlap() -> None:
    """토큰이 일부 겹치면 EM 0, F1 부분 점수."""
    em, f1 = score_text("Eiffel Tower in Paris", ["Eiffel Tower"])
    assert em == 0.0
    assert f1 == pytest.approx(2 / 3)


def test_score_text_takes_max_over_answer_candidates() -> None:
    """정답 후보 중 EM·F1 최댓값을 쓴다 (첫 후보만 보면 안 된다)."""
    assert score_text("Final answer: New York City", ["NYC", "New York City"]) == (1.0, 1.0)
    em, f1 = score_text("Final answer: New York", ["NYC", "New York City"])
    assert em == 0.0
    assert f1 == pytest.approx(0.8)


def test_score_text_unanswerable_prediction_scores_zero() -> None:
    """예측이 UNANSWERABLE이면 (0, 0)."""
    assert score_text("Final answer: UNANSWERABLE", ["Paris"]) == (0.0, 0.0)


def test_score_text_unanswerable_prediction_scores_zero_even_with_word_overlap() -> None:
    """정답에 `unanswerable`이 들어 있어도 기권 예측은 (0, 0)."""
    assert score_text("Final answer: UNANSWERABLE", ["The Unanswerable Question"]) == (0.0, 0.0)


def test_score_text_empty_prediction_is_wrong() -> None:
    """빈 예측은 오답 (정답이 관사뿐이어도 EM 1이 되지 않는다)."""
    assert score_text("", ["a"]) == (0.0, 0.0)
    assert score_text("Final answer:", ["Paris"]) == (0.0, 0.0)


# --- is_abstained / score_unanswerable ------------------------------------


def test_is_abstained_requires_exact_unanswerable() -> None:
    """정규화 후 정확히 일치할 때만 기권이다."""
    assert is_abstained("Final answer: unanswerable.") is True
    assert is_abstained("Final answer: UNANSWERABLE because the text is silent") is False
    assert is_abstained("") is False


def test_score_unanswerable_matches_after_normalization() -> None:
    """대소문자·마침표를 무시하고 일치하면 정답."""
    assert score_unanswerable("Final answer: unanswerable.") is True


def test_score_unanswerable_rejects_other_abstention_wording() -> None:
    """다른 기권 표현은 오답."""
    assert score_unanswerable("I don't know") is False


# --- score_item -----------------------------------------------------------


def test_score_item_text_fills_em_f1_and_leaves_numeric_null() -> None:
    """텍스트 문항은 em·f1을 채우고 numeric_correct는 null."""
    item = {"answer_kind": "text", "answers": ["Paris"]}
    assert score_item(item, "Final answer: Paris") == {
        "em": 1.0,
        "f1": 1.0,
        "numeric_correct": None,
        "correct": True,
    }


def test_score_item_numeric_fills_numeric_correct_and_leaves_text_null() -> None:
    """수치 문항은 numeric_correct를 채우고 em·f1은 null."""
    item = {"answer_kind": "numeric", "answers": ["1234"]}
    assert score_item(item, "Final answer: $1,234") == {
        "em": None,
        "f1": None,
        "numeric_correct": True,
        "correct": True,
    }


def test_score_item_unanswerable_correct_only_when_abstained() -> None:
    """답할 수 없음 문항은 기권했을 때만 정답이고 EM·F1은 같은 값(SQuAD v2 방식)."""
    item = {"answer_kind": "unanswerable", "answers": ["UNANSWERABLE"]}
    assert score_item(item, "Final answer: UNANSWERABLE") == {
        "em": 1.0,
        "f1": 1.0,
        "numeric_correct": None,
        "correct": True,
    }
    wrong = score_item(item, "Final answer: Paris")
    assert (wrong["em"], wrong["f1"], wrong["correct"]) == (0.0, 0.0, False)


def test_score_item_rejects_unknown_answer_kind() -> None:
    """answer_kind 오타는 조용히 텍스트 채점으로 새지 않고 ValueError."""
    with pytest.raises(ValueError):
        score_item({"answer_kind": "typo", "answers": ["42"]}, "Final answer: 42")
