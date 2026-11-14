"""평가 세트·개발 세트·공용 코퍼스 동결 스크립트 (PR A-1).

사용법 (레포 루트에서)::

    python scripts/freeze_data.py            # 원본 다운로드 → 샘플링 → data/frozen/ 에 기록
    python scripts/freeze_data.py --verify   # 이미 기록된 파일의 해시·구성·교차 검증만 수행

출력 (``docs/INTERFACES.md`` §1):
    data/frozen/eval.jsonl, dev.jsonl, corpus.jsonl, MANIFEST.json

같은 시드·같은 데이터셋 리비전이면 어떤 환경에서도 바이트 단위로 같은 파일이 나온다.
(UTF-8, LF 줄바꿈, 고정 키 순서. Windows에서도 ``\\r\\n``이 섞이지 않게 바이트로 쓴다.)

동결 이후에는 이 스크립트로 ``data/frozen/``을 다시 만들지 않는다. 기존 MANIFEST가 있으면
``--force`` 없이는 멈춘다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
FROZEN_DIR = REPO_ROOT / "data" / "frozen"

# ---------------------------------------------------------------------------
# 고정 값 (바꾸면 다른 데이터가 나온다 → 동결 전에만 바꾼다)
# ---------------------------------------------------------------------------

SEED = 20261113  # data.seed. MANIFEST에 기록된다.

# 데이터셋 리비전은 2026-11-14에 확인한 HF 커밋 해시. 이후 원본이 바뀌어도 같은 결과를 얻는다.
DATASETS = {
    "hotpotqa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
    },
    "squad2": {
        "hf_id": "rajpurkar/squad_v2",
        "config": None,
        "split": "validation",
        "revision": "3ffb306f725f7d2ce8394bc1873b24868140c412",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
    },
}

# (평가 세트 문항 수, 개발 세트 문항 수) — docs/EVALUATION.md §1
QUOTA = {
    "hotpotqa": {"bridge": (20, 10), "comparison": (20, 10)},
    "squad2": {"answerable": (20, 10), "unanswerable": (20, 10)},
    "gsm8k": {None: (40, 20)},
}

UNANSWERABLE = "UNANSWERABLE"

# ---------------------------------------------------------------------------
# 문장 분리 (SQuAD 문맥용). HotpotQA는 원본 문장 분할을 그대로 쓴다.
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "mt.", "vs.", "no.", "inc.",
    "ltd.", "co.", "corp.", "gen.", "col.", "lt.", "sgt.", "capt.", "rev.", "hon.", "gov.",
    "sen.", "rep.", "fig.", "approx.", "dept.", "est.", "e.g.", "i.e.", "etc.", "u.s.", "u.k.",
}
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=["“‘(\[]?[A-Z0-9])')
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """SQuAD 문맥을 문장 리스트로 나눈다.

    입력: 문단 원문. 출력: 앞뒤 공백을 지운 문장 리스트 (``" ".join(...)``이 공백 정규화한 원문과 같다).
    규칙: ``.``/``!``/``?`` 뒤 공백 + 대문자·숫자·여는 따옴표로 시작하면 경계. 약어(``St.``, ``U.S.`` 등)와
    한 글자 이니셜(``J.``) 뒤에서는 나누지 않는다. 외부 모델·사전이 필요 없어 어디서나 같은 결과가 나온다.
    """
    pieces = _SENT_BOUNDARY.split(" ".join(text.split()))
    sentences: list[str] = []
    for piece in pieces:
        if sentences:
            prev = sentences[-1]
            last_token = prev.split()[-1].lower() if prev.split() else ""
            if last_token in _ABBREVIATIONS or _INITIAL.search(prev):
                sentences[-1] = prev + " " + piece
                continue
        sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# 입출력 유틸
# ---------------------------------------------------------------------------


def dumps_line(obj: dict[str, Any]) -> bytes:
    """dict를 한 줄 JSON 바이트(UTF-8, 끝에 LF)로 만든다. 키 순서는 삽입 순서를 따른다."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """rows를 JSONL로 쓴다 (LF 고정)."""
    path.write_bytes(b"".join(dumps_line(r) for r in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL 파일을 읽어 dict 리스트로 돌려준다."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    """파일 바이트의 SHA-256 16진수 문자열."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict[str, Any]:
    """생성 스크립트의 git 상태(HEAD 커밋, 이 스크립트가 커밋되지 않은 변경을 가졌는지)."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    script_rel = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    return {
        "script": script_rel,
        "script_sha256": sha256_file(Path(__file__)),
        "git_head": run("rev-parse", "HEAD") or None,
        "script_dirty": bool(run("status", "--porcelain", "--", script_rel)),
    }


def load_hf(name: str):
    """원본 데이터셋을 지정 리비전으로 불러온다 (캐시는 ``data/raw/hf_cache``, git 제외)."""
    from datasets import load_dataset  # 지연 import: --verify는 datasets 없이도 돈다

    spec = DATASETS[name]
    args = [spec["hf_id"]] + ([spec["config"]] if spec["config"] else [])
    return load_dataset(
        *args,
        split=spec["split"],
        revision=spec["revision"],
        cache_dir=str(RAW_DIR / "hf_cache"),
    )


# ---------------------------------------------------------------------------
# 유형별 샘플링
# ---------------------------------------------------------------------------


def _rng(name: str) -> random.Random:
    """유형별로 독립된 난수 생성기 (한 유형의 표본 수가 바뀌어도 다른 유형이 안 바뀐다)."""
    return random.Random(f"{SEED}:{name}")


def build_hotpotqa(corpus: "Corpus") -> tuple[list[dict], list[dict]]:
    """HotpotQA distractor validation에서 다리형·비교형을 뽑고 10문단을 코퍼스에 넣는다."""
    ds = load_hf("hotpotqa")
    rng = _rng("hotpotqa")
    by_subtype: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(ds["type"]):
        by_subtype[t].append(i)

    eval_rows: list[dict] = []
    dev_rows: list[dict] = []
    counter = 0
    picked: dict[str, tuple[list[int], list[int]]] = {}
    for subtype, (n_eval, n_dev) in QUOTA["hotpotqa"].items():
        idx = rng.sample(by_subtype[subtype], n_eval + n_dev)
        picked[subtype] = (idx[:n_eval], idx[n_eval:])

    for target, which in ((eval_rows, 0), (dev_rows, 1)):
        for subtype in QUOTA["hotpotqa"]:
            for i in picked[subtype][which]:
                ex = ds[i]
                title_to_doc: dict[str, str] = {}
                for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
                    title_to_doc[title] = corpus.add_hotpot(title, sents)
                sf = list(zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]))
                gold_titles = list(dict.fromkeys(t for t, _ in sf))
                target.append(
                    {
                        "qid": f"hotpotqa-{counter:04d}",
                        "type": "hotpotqa",
                        "subtype": subtype,
                        "question": ex["question"].strip(),
                        "answers": [ex["answer"].strip()],
                        "answer_kind": "text",
                        "gold_doc_ids": [title_to_doc[t] for t in gold_titles],
                        "supporting_facts": [[t, int(s)] for t, s in sf],
                        "source_id": ex["id"],
                    }
                )
                counter += 1
    # qid 번호는 평가 → 개발 순으로 이어 붙인다(두 세트에서 겹치지 않는다).
    return eval_rows, dev_rows


def build_squad2(corpus: "Corpus") -> tuple[list[dict], list[dict]]:
    """SQuAD 2.0 validation에서 문맥 단위로 평가·개발 세트를 나눠 뽑는다.

    한 문맥에서 질문을 하나만 뽑고, 문맥이 평가 세트와 개발 세트에 동시에 들어가지 않는다.
    """
    ds = load_hf("squad2")
    rng = _rng("squad2")

    # 문맥별 질문 인덱스. 문단 번호는 같은 제목 안에서 처음 등장한 순서(데이터셋 순서)로 매긴다.
    ctx_questions: dict[str, list[int]] = {}
    para_idx: dict[str, int] = {}
    per_title_count: Counter = Counter()
    ctx_title: dict[str, str] = {}
    for i, (title, ctx) in enumerate(zip(ds["title"], ds["context"])):
        if ctx not in ctx_questions:
            ctx_questions[ctx] = []
            para_idx[ctx] = per_title_count[title]
            per_title_count[title] += 1
            ctx_title[ctx] = title
        ctx_questions[ctx].append(i)

    contexts = list(ctx_questions)  # 데이터셋 등장 순서 = 결정적
    rng.shuffle(contexts)

    buckets: dict[str, list[tuple[str, int]]] = {"answerable": [], "unanswerable": []}
    for ctx in contexts:
        qi = rng.choice(ctx_questions[ctx])
        kind = "answerable" if ds[qi]["answers"]["text"] else "unanswerable"
        buckets[kind].append((ctx, qi))

    eval_rows: list[dict] = []
    dev_rows: list[dict] = []
    counter = 0
    for target, which in ((eval_rows, 0), (dev_rows, 1)):
        for kind, (n_eval, n_dev) in QUOTA["squad2"].items():
            start = 0 if which == 0 else n_eval
            n = n_eval if which == 0 else n_dev
            for ctx, qi in buckets[kind][start : start + n]:
                ex = ds[qi]
                title = ctx_title[ctx]
                assert "::" not in title, f"제목에 '::'가 있어 doc_id가 모호하다: {title!r}"
                doc_id = f"sq::{title}::{para_idx[ctx]}"
                corpus.add_squad(doc_id, title, ctx)
                texts = list(dict.fromkeys(t.strip() for t in ex["answers"]["text"]))
                target.append(
                    {
                        "qid": f"squad2-{counter:04d}",
                        "type": "squad2",
                        "subtype": kind,
                        "question": ex["question"].strip(),
                        "answers": texts if kind == "answerable" else [UNANSWERABLE],
                        "answer_kind": "text" if kind == "answerable" else "unanswerable",
                        # 답할 수 없음 문항도 원래 문맥(함정 문단)을 근거 후보로 기록한다.
                        "gold_doc_ids": [doc_id],
                        "supporting_facts": [],
                        "source_id": ex["id"],
                    }
                )
                counter += 1
    return eval_rows, dev_rows


_GSM_ANSWER = re.compile(r"####\s*(.+)\s*$")


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K 해설에서 ``####`` 뒤 정답 값을 꺼낸다 (쉼표 제거)."""
    m = _GSM_ANSWER.search(answer_field.strip())
    if not m:
        raise ValueError(f"'####' 정답을 찾지 못함: {answer_field[-80:]!r}")
    return m.group(1).strip().replace(",", "")


def build_gsm8k() -> tuple[list[dict], list[dict]]:
    """GSM8K test에서 평가·개발 세트를 겹치지 않게 뽑는다."""
    ds = load_hf("gsm8k")
    rng = _rng("gsm8k")
    n_eval, n_dev = QUOTA["gsm8k"][None]
    idx = rng.sample(range(len(ds)), n_eval + n_dev)
    rows: list[dict] = []
    for counter, i in enumerate(idx):
        ex = ds[i]
        rows.append(
            {
                "qid": f"gsm8k-{counter:04d}",
                "type": "gsm8k",
                "subtype": None,
                "question": ex["question"].strip(),
                "answers": [gsm8k_gold(ex["answer"])],
                "answer_kind": "numeric",
                "gold_doc_ids": [],
                "supporting_facts": [],
                "source_id": f"gsm8k-test-{i}",
            }
        )
    return rows[:n_eval], rows[n_eval:]


# ---------------------------------------------------------------------------
# 코퍼스
# ---------------------------------------------------------------------------


class Corpus:
    """중복을 합치며 문단을 모으는 코퍼스 빌더.

    같은 ``(제목, 본문)``은 하나로 합친다. 같은 제목이지만 본문이 다르면 별도 문서로 두고
    ``hp::<title>::2``처럼 번호를 붙인다(먼저 나온 쪽이 ``hp::<title>``).
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self._hotpot_variants: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.title_conflicts = 0

    def add_hotpot(self, title: str, sentences: list[str]) -> str:
        """HotpotQA 문단을 추가하고 doc_id를 돌려준다. 문장 번호는 원본과 같게 유지한다."""
        sents = [s.strip() for s in sentences]
        if any(not s for s in sents):
            raise ValueError(f"빈 문장이 있어 supporting_facts 번호가 어긋난다: {title!r}")
        text = " ".join(sents)
        variants = self._hotpot_variants[title]
        for doc_id, existing_text in variants:
            if existing_text == text:
                return doc_id
        doc_id = f"hp::{title}" if not variants else f"hp::{title}::{len(variants) + 1}"
        if variants:
            self.title_conflicts += 1
        variants.append((doc_id, text))
        self.docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "sentences": sents,
            "text": text,
            "origin": "hotpotqa",
        }
        return doc_id

    def add_squad(self, doc_id: str, title: str, context: str) -> None:
        """SQuAD 문맥을 문장으로 나눠 추가한다."""
        if doc_id in self.docs:
            return
        sents = split_sentences(context)
        self.docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "sentences": sents,
            "text": " ".join(sents),
            "origin": "squad2",
        }

    def rows(self) -> list[dict[str, Any]]:
        """doc_id 순으로 정렬한 문단 리스트."""
        return [self.docs[k] for k in sorted(self.docs)]


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def validate(eval_rows: list[dict], dev_rows: list[dict], corpus_rows: list[dict]) -> dict[str, Any]:
    """구성·겹침·참조 무결성을 검사한다. 어긋나면 AssertionError. 요약 통계를 돌려준다."""
    assert len(eval_rows) == 120, f"평가 세트 {len(eval_rows)}문항"
    assert len(dev_rows) == 60, f"개발 세트 {len(dev_rows)}문항"

    for name, rows, col in (("eval", eval_rows, 0), ("dev", dev_rows, 1)):
        got = Counter((r["type"], r["subtype"]) for r in rows)
        want = {
            (t, s): q[col] for t, subs in QUOTA.items() for s, q in subs.items()
        }
        assert dict(got) == want, f"{name} 구성 불일치: {dict(got)} != {want}"

    qids = [r["qid"] for r in eval_rows + dev_rows]
    assert len(qids) == len(set(qids)), "qid 중복"
    src = [r["source_id"] for r in eval_rows + dev_rows]
    assert len(src) == len(set(src)), "source_id 중복(세트 간 문항 겹침)"
    questions = [r["question"] for r in eval_rows + dev_rows]
    assert len(questions) == len(set(questions)), "질문 텍스트 중복"

    corpus_ids = {d["doc_id"] for d in corpus_rows}
    assert len(corpus_ids) == len(corpus_rows), "코퍼스 doc_id 중복"
    for r in eval_rows + dev_rows:
        missing = [d for d in r["gold_doc_ids"] if d not in corpus_ids]
        assert not missing, f"{r['qid']}: 코퍼스에 없는 gold_doc_ids {missing}"
        if r["type"] == "hotpotqa":
            assert len(r["gold_doc_ids"]) == 2, f"{r['qid']}: 정답 문단이 2개가 아님"
        if r["answer_kind"] == "unanswerable":
            assert r["answers"] == [UNANSWERABLE]
        if r["answer_kind"] == "numeric":
            assert re.fullmatch(r"-?\d+(\.\d+)?", r["answers"][0]), r["answers"]
        assert r["answers"] and all(isinstance(a, str) and a for a in r["answers"]), r["qid"]

    # SQuAD: 평가 세트와 개발 세트가 같은 문맥(문단)을 공유하지 않는다 (EVALUATION §1).
    sq_eval = {d for r in eval_rows if r["type"] == "squad2" for d in r["gold_doc_ids"]}
    sq_dev = {d for r in dev_rows if r["type"] == "squad2" for d in r["gold_doc_ids"]}
    assert not (sq_eval & sq_dev), f"SQuAD 문맥 공유: {sq_eval & sq_dev}"
    assert len(sq_eval) == 40 and len(sq_dev) == 20, "SQuAD 문항당 문맥이 하나씩이어야 한다"

    # HotpotQA: 평가·개발 사이에 정답 문단이 겹치는지는 막지 않고 보고만 한다.
    hp_eval = {d for r in eval_rows if r["type"] == "hotpotqa" for d in r["gold_doc_ids"]}
    hp_dev = {d for r in dev_rows if r["type"] == "hotpotqa" for d in r["gold_doc_ids"]}

    # 근거 문장 번호가 문단 범위 안에 있는가.
    by_title = {d["title"]: d for d in corpus_rows if d["origin"] == "hotpotqa"}
    for r in eval_rows + dev_rows:
        for title, sid in r["supporting_facts"]:
            assert title in by_title and sid < len(by_title[title]["sentences"]), (r["qid"], title, sid)

    return {
        "n_eval": len(eval_rows),
        "n_dev": len(dev_rows),
        "n_corpus": len(corpus_rows),
        "corpus_by_origin": dict(Counter(d["origin"] for d in corpus_rows)),
        "hotpotqa_gold_paragraph_overlap_eval_dev": len(hp_eval & hp_dev),
        "squad2_context_overlap_eval_dev": 0,
    }


def sanity_text(corpus_rows: list[dict]) -> None:
    """문장 분리가 본문을 잃지 않았는지(공백 정규화 기준) 확인한다."""
    for d in corpus_rows:
        assert d["text"] == " ".join(d["sentences"]), d["doc_id"]
        assert all(s == s.strip() and s for s in d["sentences"]), d["doc_id"]


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

FILES = ("eval.jsonl", "dev.jsonl", "corpus.jsonl")


def freeze(force: bool) -> None:
    """원본을 받아 세트·코퍼스를 만들고 ``data/frozen/``에 기록한다."""
    manifest_path = FROZEN_DIR / "MANIFEST.json"
    if manifest_path.exists() and not force:
        sys.exit("이미 동결되어 있다(data/frozen/MANIFEST.json). 동결 후 수정 금지 — 덮어쓰지 않는다.")
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    corpus = Corpus()
    hp_eval, hp_dev = build_hotpotqa(corpus)
    sq_eval, sq_dev = build_squad2(corpus)
    gs_eval, gs_dev = build_gsm8k()
    eval_rows = hp_eval + sq_eval + gs_eval
    dev_rows = hp_dev + sq_dev + gs_dev
    corpus_rows = corpus.rows()

    sanity_text(corpus_rows)
    stats = validate(eval_rows, dev_rows, corpus_rows)

    write_jsonl(FROZEN_DIR / "eval.jsonl", eval_rows)
    write_jsonl(FROZEN_DIR / "dev.jsonl", dev_rows)
    write_jsonl(FROZEN_DIR / "corpus.jsonl", corpus_rows)

    manifest = {
        "seed": SEED,
        "datasets": DATASETS,
        "quota": {
            t: {str(s): {"eval": q[0], "dev": q[1]} for s, q in subs.items()}
            for t, subs in QUOTA.items()
        },
        "rules": {
            "squad2_split": "context-level (한 문맥에서 질문 1개, 평가·개발 문맥 비중복)",
            "squad2_sentence_splitter": "regex + abbreviation guard (scripts/freeze_data.py: split_sentences)",
            "hotpotqa_sentences": "원본 문장 분할, 각 문장 strip, text는 공백 1칸으로 이음",
            "doc_id": {
                "hotpotqa": "hp::<title> (같은 제목·다른 본문이면 hp::<title>::2, ...)",
                "squad2": "sq::<article title>::<paragraph_idx> (idx = 같은 제목 안 데이터셋 등장 순서)",
            },
            "qid": "<type>-<4자리 번호>; 번호는 유형별로 평가 세트 → 개발 세트 순서로 이어 붙임",
            "gold_doc_ids_squad2_unanswerable": "원래 문맥(함정 문단)을 기록한다",
        },
        "stats": {**stats, "hotpotqa_title_variants": corpus.title_conflicts},
        "generator": git_info(),
        "files": {
            name: {"rows": len(read_jsonl(FROZEN_DIR / name)), "sha256": sha256_file(FROZEN_DIR / name)}
            for name in FILES
        },
    }
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    print("동결 파일을 썼다:")
    for name, info in manifest["files"].items():
        print(f"  {name:14s} {info['rows']:5d}행  sha256={info['sha256']}")
    print("통계:", json.dumps(manifest["stats"], ensure_ascii=False))
    if manifest["generator"]["script_dirty"]:
        print("경고: 이 스크립트에 커밋되지 않은 변경이 있다. 스크립트를 커밋한 뒤 다시 동결해야 한다.")


def verify() -> None:
    """기록된 파일의 해시가 MANIFEST와 같은지, 구성·참조가 올바른지 확인한다."""
    manifest = json.loads((FROZEN_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    ok = True
    for name in FILES:
        actual = sha256_file(FROZEN_DIR / name)
        expected = manifest["files"][name]["sha256"]
        flag = "OK " if actual == expected else "FAIL"
        ok &= actual == expected
        print(f"[{flag}] {name:14s} {actual}")
    eval_rows = read_jsonl(FROZEN_DIR / "eval.jsonl")
    dev_rows = read_jsonl(FROZEN_DIR / "dev.jsonl")
    corpus_rows = read_jsonl(FROZEN_DIR / "corpus.jsonl")
    sanity_text(corpus_rows)
    stats = validate(eval_rows, dev_rows, corpus_rows)
    print("구성·참조 검증 통과:", json.dumps(stats, ensure_ascii=False))
    if not ok:
        sys.exit("해시 불일치")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="기록된 동결 파일만 검증한다")
    parser.add_argument("--force", action="store_true", help="기존 MANIFEST를 무시하고 다시 쓴다(동결 전에만)")
    args = parser.parse_args()
    verify() if args.verify else freeze(args.force)


if __name__ == "__main__":
    main()
