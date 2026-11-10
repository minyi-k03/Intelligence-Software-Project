# M(EBA) — Evidence- and Budget-Aware Agent

> 3조 · 지능형 소프트웨어 기말 프로젝트 · 고용민 · 이현성 · 조환호 · 심동섭

**모델은 그대로, 하네스만 바꾼다.**
M(EBA)는 Qwen2.5-7B-Instruct(4-bit)를 그대로 두고 하네스만 바꿔, 표준 ReAct(P1)보다 정확도는 높고 호출 수와 시간은 P1 이하인 단일 에이전트입니다. 근거를 따져 필요한 것만 검색하고(입력 쪽), 생성량과 호출 수를 예산으로 묶습니다(출력 쪽).

보고 문장: **P1 대비 정확도 +Δ, 호출 수 ×k, 추론 시간 ×t, 최대 VRAM n GB**

---

## 처음 왔다면 이 순서로 읽으세요

| 순서 | 문서 | 내용 |
|---|---|---|
| 1 | [docs/SPEC.md](docs/SPEC.md) | 구현 명세 — 무엇을 만드는지 (최종 설계안 요약, 구현의 기준) |
| 2 | [GLOSSARY.md](GLOSSARY.md) | 팀 공통 용어. 코드·문서·발표에서 같은 말을 쓰기 위함 |
| 3 | [docs/INTERFACES.md](docs/INTERFACES.md) | 모듈 사이 규격 — 데이터 형식, 로그 필드, 설정 키 |
| 4 | [docs/SCHEDULE.md](docs/SCHEDULE.md) | 일정과 역할별 체크리스트 |
| 5 | [CONTRIBUTING.md](CONTRIBUTING.md) | 브랜치·커밋·PR 규칙 |
| - | [AGENTS.md](AGENTS.md) | AI 코딩 에이전트(Claude Code · Codex · Antigravity)용 작업 규칙 |

설계 결정의 이유는 [docs/adr/](docs/adr/)에, 평가·측정 규칙은 [docs/EVALUATION.md](docs/EVALUATION.md)에 있습니다.

## 고정 조건

| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-7B-Instruct 4-bit (교수님 지정, 모든 파이프라인 동일) |
| 서빙 | Ollama, temperature 0, num_ctx 8192 |
| 호출 방식 | 모든 호출을 LangChain `ChatOllama`로 — 모델 태그는 설정 한 곳에서만 정의 |
| 평가 | HotpotQA · SQuAD 2.0 · GSM8K 각 40문항 = 120문항 (11/12 동결) |
| 비교 대상 | P0 · P0-raw · P1 · M |
| 실행 환경 | Google Colab 무료 (시간 추정은 T4 기준) |
| 금지 | 평가 세트 사후 변경 · 외부 API · 파이프라인별 다른 모델 (감점 3종) |

## 레포 구조

```
.
├── README.md            ← 지금 보는 문서
├── AGENTS.md            ← AI 에이전트 공통 규칙 (CLAUDE.md는 이 파일을 불러옴)
├── GLOSSARY.md          ← 팀 용어집
├── CONTRIBUTING.md      ← 협업 규칙
├── docs/
│   ├── SPEC.md          ← 구현 명세 (설계안 요약)
│   ├── INTERFACES.md    ← 모듈 간 규격
│   ├── EVALUATION.md    ← 채점·측정·통계 규칙
│   ├── SCHEDULE.md      ← 일정·역할 체크리스트
│   ├── adr/             ← 설계 결정 기록 0001~0006
│   ├── experiments/     ← 개발 세트 실험 기록 (조건부 부품 채택 근거)
│   └── reports/         ← 최종 보고서·발표 개요
├── configs/             ← 실행 설정 (모델 태그·상한·예산 등 단일 관리)
├── data/
│   ├── raw/             ← 원본 데이터 (git 제외)
│   └── frozen/          ← 동결된 평가·개발 세트·공용 코퍼스 (수정 금지)
├── src/
│   ├── config/          ← 설정 로더·ChatOllama 생성 (B)
│   ├── tools/           ← BM25 검색·계산기 (B)
│   ├── baselines/       ← P0 · P0-raw · P1 (B)
│   ├── meba/            ← M(EBA) 부품과 그래프 (A·C·D)
│   │   └── conditional/ ← 조건부 부품 3개 (C·D)
│   └── eval/            ← 채점기·로거·러너·통계 (A·D)
├── scripts/             ← 데이터 동결, 일괄 실행 등 CLI 스크립트
├── notebooks/           ← Colab 원클릭 노트북, 검증 모드
├── results/
│   ├── runs/            ← 문항별 .jsonl 로그
│   └── tables/          ← 결과표·절제표·McNemar
└── tests/
```

모듈별 담당과 들어갈 파일은 [src/README.md](src/README.md)에 있습니다.

## 빠른 시작

> 코드가 들어오면 채웁니다. B(기반)와 A(원클릭 노트북) 담당.

```bash
# 1) 의존성 설치          — TODO
# 2) Ollama 모델 받기       — TODO (모델 태그는 configs/ 참고)
# 3) 검증 모드로 소수 문항 실행 — TODO
```

## 역할

| 역할 | 담당 | 범위 |
|---|---|---|
| A. 측정·채점·재현성 + 통합 | 고용민 | 세트·코퍼스 동결, 채점기, 측정 로거, 라운드 실행, 원클릭 노트북, 라우터, M 전체 통합 |
| B. 기반·기준선 | 이현성 | Ollama 설정, BM25 검색·계산기, P0 · P0-raw · P1, 검색 결과 수 k 측정 |
| C. 근거 쪽 (전략 ③) | 조환호 | 근거 판정 + 답 호출(프롬프트 최종 책임), 다중 홉 재검색, 압축(m 측정), 조건부: 근거 장부·동적 압축량 |
| D. 예산 쪽 (전략 ①) | 심동섭 | 출력 상한·잘림 처리, 루프 끊기, 민감도 실험, 조건부: 간결 출력, 결과표·절제표·McNemar |

## 주요 일정

| 날짜 | 마일스톤 |
|---|---|
| 11/12 | 평가·개발 세트와 공용 코퍼스 동결, 해시 기록 |
| 11/13 | 핵심 부품 개발 시작 (P1과 병행) |
| 11/19 | P1 로그 점검 → 조건부 시험 순서·출력 상한 초기값 조정 |
| 11/24 | M 핵심 부품 통합 |
| 11/27 | 조건부 부품 채택 확정 |
| 12/04~05 | 버퍼 (측정 뒤) |
| 12/08 | 발표 |
