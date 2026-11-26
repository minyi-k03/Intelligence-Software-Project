"""M(EBA) 그래프 조립 — LangGraph 골격과 부품 켜기/끄기 (SPEC §4·§5·§6, 담당 A 통합).

흐름 (SPEC §4):

    router ─┬─ math ─▶ END
            └─ search ─▶ compress ─▶ judge ─┬─ ANSWER / ABSTAIN ─▶ END
                 ▲                          └─ RETRIEVE ─▶ requery ─┐
                 └──────────────────────────────────────────────────┘

- 모델을 부르는 노드는 `router`(수학 경로의 마무리 호출 포함)와 `judge` 둘뿐이다.
  호출은 모두 `BudgetedLLM`(출력 상한·잘림 재시도·호출 예산)을 거친다.
- `search`·`compress`는 모델을 부르지 않는다. 같은 검색을 다시 하려 하면 `LoopBreaker`가 막고,
  지금 근거로 RETRIEVE 없는 마지막 판정을 받는다 (`scripts/compare_judge_format.py`와 같은 규칙).
- 절제 변형은 코드 복사본이 아니라 `MComponents`의 켜기/끄기로 만든다 (src/README.md).
  조건부 부품(근거 장부 · 간결 출력 · 동적 압축량)은 `ConditionalHooks`에 붙일 자리를 두었고,
  기본은 꺼져 있다. 켰는데 부품이 없으면 조용히 M으로 돌지 않고 만들 때 실패한다.
- 문항 유형은 어디서도 받지 않는다. 파이프라인은 `(qid, question)`만 받고 모델에는 질문만 간다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.baselines.prompts import validate_question
from src.common.types import PipelineResult
from src.config.llm import create_chat_ollama
from src.config.settings import Settings, get_settings
from src.eval.scorer import UNANSWERABLE, is_abstained
from src.meba.budget import BudgetedLLM
from src.meba.compress import compress, full_paragraphs
from src.meba.judge import judge
from src.meba.loop_breaker import LoopBreaker
from src.meba.requery import Requery, search_args_key
from src.meba.router import route_question
from src.tools.bm25_search import BM25Index, bm25_search, get_default_index
from src.tools.tokenizer import tokenize

# --- 부품 켜기/끄기 -----------------------------------------------------------

CORE_PARTS: tuple[str, ...] = ("router", "requery", "judge", "compress", "cap", "loop_breaker")
CONDITIONAL_PARTS: tuple[str, ...] = ("ledger", "concise", "dynamic_m")

# 핵심 부품을 빼면 그 위에서 동작하는 조건부 부품도 함께 빠진다 (SPEC §6, ADR-0006).
_CONDITIONAL_DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "ledger": ("requery", "judge"),
    "dynamic_m": ("compress",),
    "concise": ("cap",),
}
_VARIANT_PREFIX = "m_minus_"


@dataclass(frozen=True)
class MComponents:
    """M의 부품 켜기/끄기. 기본값이 M 자체이고, 절제 변형은 부품 하나를 끈 것이다.

    router: 끄면 라우터 호출 없이 모든 질문을 검색 경로로 (`m_minus_router`)
    requery: 끄면 RETRIEVE 없이 한 번만 검색 (`m_minus_requery`)
    judge: 끄면 판정에서 ABSTAIN 제거, 재검색은 유지 (`m_minus_judge`)
    compress: 끄면 상위 k개 문단을 통째로 (`m_minus_compress`)
    cap: 끄면 출력 상한과 잘림 재시도 제거 (`m_minus_cap`)
    loop_breaker: 끄면 같은 도구 호출 반복 허용 (`m_minus_loop_breaker`)
    ledger, concise, dynamic_m: 조건부 부품. 개발 세트에서 채택되기 전까지 꺼 둔다
    """

    router: bool = True
    requery: bool = True
    judge: bool = True
    compress: bool = True
    cap: bool = True
    loop_breaker: bool = True
    ledger: bool = False
    concise: bool = False
    dynamic_m: bool = False

    def __post_init__(self) -> None:
        """핵심 부품이 꺼졌는데 그 위의 조건부 부품이 켜진 구성을 막는다 (SPEC §6, ADR-0006).

        Raises:
            ValueError: 켜진 조건부 부품이 기대는 핵심 부품 중 하나라도 꺼져 있을 때
        """
        for conditional, needs in _CONDITIONAL_DEPENDS_ON.items():
            missing = [core for core in needs if not getattr(self, core)]
            if getattr(self, conditional) and missing:
                raise ValueError(
                    f"conditional part {conditional!r} needs core part(s) {missing} switched on; "
                    f"use MComponents.without() to drop a core part together with its conditional parts"
                )

    def without(self, part: str) -> "MComponents":
        """핵심 부품 하나를 끈 구성. 그 위에서 동작하는 조건부 부품도 함께 끈다.

        Args:
            part: `CORE_PARTS` 중 하나

        Returns:
            부품을 끈 새 구성

        Raises:
            ValueError: 핵심 부품 이름이 아닐 때
        """
        if part not in CORE_PARTS:
            raise ValueError(f"unknown core part {part!r}; expected one of {CORE_PARTS}")
        off = {part: False}
        for conditional, needs in _CONDITIONAL_DEPENDS_ON.items():
            if part in needs:
                off[conditional] = False
        return dataclasses.replace(self, **off)


def variant_components(name: str) -> MComponents:
    """파이프라인 이름에서 부품 구성을 만든다.

    Args:
        name: `"m"` 또는 `"m_minus_<핵심 부품>"` (INTERFACES §8)

    Returns:
        해당 변형의 부품 구성

    Raises:
        ValueError: 알 수 없는 이름
    """
    if name == "m":
        return MComponents()
    if name.startswith(_VARIANT_PREFIX) and name[len(_VARIANT_PREFIX):] in CORE_PARTS:
        return MComponents().without(name[len(_VARIANT_PREFIX):])
    raise ValueError(
        f"unknown M variant {name!r}; expected 'm' or "
        + ", ".join(f"'{_VARIANT_PREFIX}{part}'" for part in CORE_PARTS)
    )


# --- 조건부 부품이 붙을 자리 ---------------------------------------------------


@dataclass(frozen=True)
class ConditionalHooks:
    """조건부 부품(C-5 · D-5)이 그래프에 붙는 자리. 부품이 생기면 여기에 함수를 넘긴다.

    choose_m: 동적 압축량(C). `(question, 기본 m) -> 이번 압축에서 남길 m`
    ledger: 근거 장부 상세화(C). RETRIEVE 뒤 `(question, 빠진 정보) -> 후속 검색어 재료`
        (현재는 자리 확보용이며 C-5에서 필요 시 시그니처를 확장한다)
    concise: 간결 출력(D). 모델 호출 직전 `(단계 이름, 모델 입력) -> 모델 입력`.
        `step`이 `"router"`·`"finalize"`일 때는 `list[dict[str, Any]]`(`{"role", "content", ...}`),
        `"judge"`일 때는 `list[tuple[str, str]]`(`("system" | "human", content)`)이 넘어오므로
        훅 구현은 두 메시지 형식을 모두 처리해야 한다 (INTERFACES §7.1).
    """

    choose_m: Optional[Callable[[str, int], int]] = None
    ledger: Optional[Callable[[str, str], str]] = None
    concise: Optional[Callable[[str, Any], Any]] = None


_HOOK_FOR_PART = {"ledger": "ledger", "concise": "concise", "dynamic_m": "choose_m"}


def _check_hooks(components: MComponents, hooks: ConditionalHooks) -> None:
    """켜진 조건부 부품마다 붙일 함수가 있는지 확인한다.

    Raises:
        ValueError: 켜져 있는데 함수가 없을 때
    """
    for part in CONDITIONAL_PARTS:
        if getattr(components, part) and getattr(hooks, _HOOK_FOR_PART[part]) is None:
            raise ValueError(
                f"conditional part {part!r} is switched on but ConditionalHooks."
                f"{_HOOK_FOR_PART[part]} is not attached yet (C-5 / D-5)"
            )


class _HookedLLM:
    """`concise` 부품이 모델 입력을 바꿀 수 있게 `BudgetedLLM.invoke`를 감싼다. 나머지는 그대로 넘긴다."""

    def __init__(self, inner: BudgetedLLM, rewrite: Callable[[str, Any], Any]) -> None:
        self._inner = inner
        self._rewrite = rewrite

    def invoke(self, step: str, model_input: Any, **kwargs: Any) -> Any:
        """`concise`가 입력을 고친 뒤 호출한다."""
        return self._inner.invoke(step, self._rewrite(step, model_input), **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --- 상태와 노드 --------------------------------------------------------------


class MState(TypedDict, total=False):
    """그래프가 노드 사이에 넘기는 상태. 문항 유형 필드는 없다.

    question: 질문 원문
    route: "math" | "search" | None
    answer: 지금까지 정해진 최종 답 (접두어 없음)
    query: 다음 검색에 쓸 검색어
    followup_query: 재검색일 때 후속 검색어 (압축 점수에 질문과 함께 쓴다)
    paragraphs: 마지막 검색 결과
    contexts: 회차별 압축 근거 (판정에는 이어 붙여 넘긴다)
    decisions: 근거 판정 결과 순서
    missing: 마지막 RETRIEVE 판정의 빠진 정보
    retrieve: 마지막 판정이 RETRIEVE라서 재검색으로 가야 하는가
    final_only: 다음 판정은 RETRIEVE 없이 끝내야 하는가 (루프 끊기·후속 검색어 없음)
    blocked: 방금 검색이 루프 끊기에 막혔는가
    retrieved_doc_ids, kept_windows: 회차별 검색·압축 결과 (로그용)
    loop_breaks: 라우터(계산기)에서 막은 반복 횟수
    """

    question: str
    route: Optional[str]
    answer: str
    query: str
    followup_query: Optional[str]
    paragraphs: list[dict[str, Any]]
    contexts: list[str]
    decisions: list[str]
    missing: str
    retrieve: bool
    final_only: bool
    blocked: bool
    retrieved_doc_ids: list[list[str]]
    kept_windows: list[list[str]]
    loop_breaks: int


@dataclass
class _QuestionRun:
    """한 문항에서만 쓰는 상태 객체. 문항 사이에 예산·재검색 횟수·루프 끊기가 새지 않게 문항마다 만든다."""

    llm: Any
    requery: Requery
    breaker: Optional[LoopBreaker]


def _run_of(config: RunnableConfig) -> _QuestionRun:
    """그래프 호출 설정에서 이 문항의 `_QuestionRun`을 꺼낸다."""
    return config["configurable"]["run"]


def _compile_graph(
    settings: Settings,
    components: MComponents,
    hooks: ConditionalHooks,
    index: BM25Index,
) -> Any:
    """노드와 간선을 조립해 컴파일한다. 문항 사이에 재사용한다.

    Args:
        settings: 설정 (`retrieval.k`, `compress.*`)
        components: 부품 켜기/끄기
        hooks: 조건부 부품이 붙을 자리
        index: 코퍼스 BM25 색인

    Returns:
        컴파일된 LangGraph
    """

    def router_node(state: MState, config: RunnableConfig) -> dict[str, Any]:
        """첫 호출. 수학이면 그 자리에서 답까지, 검색이면 경로만 정한다. 끄면 모델을 부르지 않는다."""
        if not components.router:
            return {"route": "search"}
        outcome = route_question(
            state["question"], _run_of(config).llm, use_loop_breaker=components.loop_breaker
        )
        return {
            "route": outcome.route,
            "answer": outcome.answer or "",
            "loop_breaks": outcome.loop_breaks,
        }

    def math_node(state: MState) -> dict[str, Any]:
        """수학 경로의 끝. 답이 UNANSWERABLE이면 형식을 통일한다 (INTERFACES §4)."""
        answer = state.get("answer", "")
        return {"answer": UNANSWERABLE if is_abstained(answer) else answer}

    def search_node(state: MState, config: RunnableConfig) -> dict[str, Any]:
        """BM25 검색 (모델 호출 없음). 같은 검색이면 루프 끊기가 막고 직전 결과를 쓰게 한다."""
        run = _run_of(config)
        query = state["query"]

        def execute() -> Any:
            return bm25_search(query, settings.retrieval.k, index=index)

        if run.breaker is None:
            result: Any = execute()
        else:
            outcome = run.breaker.run("bm25_search", {"query": query}, execute)
            if outcome.blocked:
                return {"blocked": True, "final_only": True}
            result = outcome.result
        # 검색 오류 문자열은 근거 없음으로 본다 (도구 오류는 관찰 결과이고 루프를 멈추지 않는다).
        paragraphs = result if isinstance(result, list) else []
        return {
            "blocked": False,
            "paragraphs": paragraphs,
            "retrieved_doc_ids": [*state.get("retrieved_doc_ids", []), [p["doc_id"] for p in paragraphs]],
        }

    def compress_node(state: MState) -> dict[str, Any]:
        """검색 결과를 문장 창으로 줄이고(끄면 문단 통째로) 지금까지의 근거에 더한다 (모델 호출 없음)."""
        paragraphs = state.get("paragraphs", [])
        if components.compress:
            m = settings.compress.m
            if components.dynamic_m:
                m = hooks.choose_m(state["question"], m)  # type: ignore[misc]
            packed = compress(
                state["question"],
                paragraphs,
                m=m,
                window=settings.compress.window,
                tokenizer=tokenize,
                followup_query=state.get("followup_query"),
            )
        else:
            packed = full_paragraphs(paragraphs)
        contexts = state.get("contexts", [])
        return {
            "contexts": [*contexts, packed.context] if packed.context else contexts,
            "kept_windows": [*state.get("kept_windows", []), packed.kept_windows],
        }

    def judge_node(state: MState, config: RunnableConfig) -> dict[str, Any]:
        """근거 판정 + 답 호출. 재검색 횟수를 다 썼거나 `final_only`면 RETRIEVE를 낼 수 없다."""
        run = _run_of(config)
        allow_retrieve = run.requery.allow_retrieve and not state.get("final_only", False)
        judgment = judge(
            run.llm,
            state["question"],
            "\n\n".join(state.get("contexts", [])),
            allow_retrieve=allow_retrieve,
            allow_abstain=components.judge,
        )
        if judgment is None:  # 호출 예산 소진: 답을 만들지 못했다 (기권이 아니다).
            return {"answer": "", "retrieve": False}
        update: dict[str, Any] = {
            "decisions": [*state.get("decisions", []), judgment.decision],
            "retrieve": judgment.decision == "RETRIEVE",
        }
        if judgment.decision == "RETRIEVE":
            update["missing"] = judgment.missing
        else:
            update["answer"] = judgment.answer
        return update

    def requery_node(state: MState, config: RunnableConfig) -> dict[str, Any]:
        """빠진 정보를 후속 검색어로 바꾼다. 못 만들면 지금 근거로 마지막 판정을 받는다."""
        missing = state.get("missing", "")
        if components.ledger:
            missing = hooks.ledger(state["question"], missing)  # type: ignore[misc]
        followup = _run_of(config).requery.next_query(missing)
        if followup is None:
            return {"final_only": True, "retrieve": False}
        return {"query": followup, "followup_query": followup, "retrieve": False}

    def after_router(state: MState) -> str:
        """라우터 뒤 갈림길. 경로를 못 정했으면(예산 소진) 끝낸다."""
        return {"math": "math", "search": "search"}.get(state.get("route") or "", END)

    def after_search(state: MState) -> str:
        """막힌 검색은 새 근거가 없으니 압축을 건너뛰고 바로 마지막 판정으로 간다."""
        return "judge" if state.get("blocked") else "compress"

    def after_judge(state: MState) -> str:
        """RETRIEVE면 재검색, 아니면 끝."""
        return "requery" if state.get("retrieve") else END

    def after_requery(state: MState) -> str:
        """후속 검색어가 있으면 검색, 없으면 바로 마지막 판정."""
        return "judge" if state.get("final_only") else "search"

    graph = StateGraph(MState)
    graph.add_node("router", router_node)
    graph.add_node("math", math_node)
    graph.add_node("search", search_node)
    graph.add_node("compress", compress_node)
    graph.add_node("judge", judge_node)
    graph.add_node("requery", requery_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router, ["math", "search", END])
    graph.add_edge("math", END)
    graph.add_conditional_edges("search", after_search, ["compress", "judge"])
    graph.add_edge("compress", "judge")
    graph.add_conditional_edges("judge", after_judge, ["requery", END])
    graph.add_conditional_edges("requery", after_requery, ["search", "judge"])
    return graph.compile()


# --- 파이프라인 조립 ----------------------------------------------------------


def _to_result(state: MState, run: _QuestionRun, error: Optional[str]) -> PipelineResult:
    """그래프 상태와 문항 객체의 집계를 `PipelineResult`로 바꾼다.

    Args:
        state: 마지막으로 얻은 그래프 상태 (오류면 오류 직전 상태)
        run: 이 문항의 예산·루프 끊기 객체
        error: 처리하지 못한 예외 메시지

    Returns:
        INTERFACES §4 규격의 결과. 빈 답은 기권으로 바꾸지 않는다.
    """
    answer = "" if error else state.get("answer", "")
    abstained = is_abstained(answer)
    return PipelineResult(
        answer=UNANSWERABLE if abstained else answer,
        abstained=abstained,
        budget_exhausted=run.llm.budget_exhausted,
        route=state.get("route"),  # type: ignore[arg-type]
        decisions=list(state.get("decisions", [])),
        n_calls=run.llm.n_calls,
        n_retries=run.llm.n_retries,
        calls=list(run.llm.calls),
        retrieved_doc_ids=[list(ids) for ids in state.get("retrieved_doc_ids", [])],
        kept_windows=[list(ids) for ids in state.get("kept_windows", [])],
        loop_breaks=state.get("loop_breaks", 0) + (run.breaker.loop_breaks if run.breaker else 0),
        error=error,
    )


def build_m(
    settings: Optional[Settings] = None,
    *,
    components: Optional[MComponents] = None,
    hooks: Optional[ConditionalHooks] = None,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    index: Optional[BM25Index] = None,
) -> Callable[[str, str], PipelineResult]:
    """M(또는 절제 변형) 파이프라인을 만든다. `scripts/run_eval.py`의 `"m"` 등록 항목이다.

    그래프는 한 번만 컴파일하고 문항마다 재사용한다. 문항마다 새로 만드는 것은
    `BudgetedLLM`(예산·콜백)·`Requery`·`LoopBreaker`뿐이다.

    Args:
        settings: 설정. 없으면 `get_settings()` (모델 태그·상한·예산은 모두 여기서 읽는다)
        components: 부품 켜기/끄기. 없으면 M 자체
        hooks: 조건부 부품이 붙을 자리
        llm_factory: ChatOllama 생성 함수 (테스트에서 가짜 모델 주입)
        index: BM25 색인. 없으면 `data/frozen/corpus.jsonl`로 만든 기본 색인

    Returns:
        `(qid, question) -> PipelineResult`. `qid`는 모델에 넘기지 않는다.

    Raises:
        ValueError: 조건부 부품을 켰는데 붙일 함수가 없을 때
    """
    cfg = settings or get_settings()
    parts = components or MComponents()
    attached = hooks or ConditionalHooks()
    _check_hooks(parts, attached)
    graph = _compile_graph(cfg, parts, attached, index or get_default_index())
    # 가장 긴 순환(검색 → 압축 → 판정 → 재검색)을 재검색 최대 횟수만큼 돌 수 있는 여유.
    recursion_limit = 4 * (cfg.budget.max_requery + 2) + 5

    def pipeline(qid: str, question: str) -> PipelineResult:
        """한 문항을 끝까지 돌린다. 처리하지 못한 예외는 `error`에 남기고 그때까지의 호출 기록을 보존한다."""
        _ = qid  # 로그는 러너가 남긴다. 모델에는 질문만 간다.
        llm: Any = BudgetedLLM(cfg, llm_factory=llm_factory, enforce_caps=parts.cap)
        if parts.concise:
            llm = _HookedLLM(llm, attached.concise)  # type: ignore[arg-type]
        run = _QuestionRun(
            llm=llm,
            requery=Requery(cfg.budget.max_requery if parts.requery else 0),
            breaker=LoopBreaker(normalizer=search_args_key) if parts.loop_breaker else None,
        )
        state: MState = {}
        try:
            clean = validate_question(question)
            initial: MState = {"question": clean, "query": clean}
            for state in graph.stream(
                initial,
                config={"configurable": {"run": run}, "recursion_limit": recursion_limit},
                stream_mode="values",
            ):
                pass
        except Exception as exc:  # 러너의 오답 한 줄보다 호출 기록이 남는 쪽이 분석에 낫다.
            return _to_result(state, run, f"{type(exc).__name__}: {exc}")
        return _to_result(state, run, None)

    return pipeline


def build_variant(
    name: str,
    settings: Optional[Settings] = None,
    *,
    hooks: Optional[ConditionalHooks] = None,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    index: Optional[BM25Index] = None,
) -> Callable[[str, str], PipelineResult]:
    """이름(`m`, `m_minus_<부품>`)으로 파이프라인을 만든다.

    Args:
        name: 파이프라인 이름 (INTERFACES §7.1)
        settings: 설정
        hooks: 조건부 부품이 붙을 자리
        llm_factory: ChatOllama 생성 함수
        index: BM25 색인

    Returns:
        `(qid, question) -> PipelineResult`

    Raises:
        ValueError: 알 수 없는 이름
    """
    return build_m(
        settings,
        components=variant_components(name),
        hooks=hooks,
        llm_factory=llm_factory,
        index=index,
    )


# `scripts/run_eval.py`가 "모듈:함수" 문자열로 불러오는 절제 변형 생성 함수 (설정만 받는다).


def build_m_minus_router(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """절제 변형: 모든 질문을 검색 경로로."""
    return build_variant("m_minus_router", settings)


def build_m_minus_requery(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """절제 변형: RETRIEVE 제거, 한 번만 검색."""
    return build_variant("m_minus_requery", settings)


def build_m_minus_judge(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """절제 변형: ABSTAIN 제거 (재검색은 유지)."""
    return build_variant("m_minus_judge", settings)


def build_m_minus_compress(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """절제 변형: 상위 k개 문단을 통째로."""
    return build_variant("m_minus_compress", settings)


def build_m_minus_cap(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """절제 변형: 출력 상한과 잘림 재시도 제거."""
    return build_variant("m_minus_cap", settings)


def build_m_minus_loop_breaker(settings: Optional[Settings] = None) -> Callable[[str, str], PipelineResult]:
    """절제 변형: 반복 허용."""
    return build_variant("m_minus_loop_breaker", settings)
