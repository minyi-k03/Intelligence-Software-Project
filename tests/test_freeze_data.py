"""동결 스크립트(scripts/freeze_data.py)와 동결 데이터의 무결성 테스트. 네트워크 불필요."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("freeze_data", ROOT / "scripts" / "freeze_data.py")
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)


def test_split_sentences_basic():
    text = "The war began in 1939. It ended in 1945! Was it long? Yes."
    assert fd.split_sentences(text) == [
        "The war began in 1939.",
        "It ended in 1945!",
        "Was it long?",
        "Yes.",
    ]


def test_split_sentences_keeps_abbreviations_and_initials():
    text = "Dr. Smith met J. K. Rowling in St. Louis. They talked."
    assert fd.split_sentences(text) == ["Dr. Smith met J. K. Rowling in St. Louis.", "They talked."]


def test_split_sentences_roundtrip_whitespace_normalized():
    text = "A  b.\nC d.   E f."
    assert " ".join(fd.split_sentences(text)) == " ".join(text.split())


def test_gsm8k_gold_strips_commas():
    assert fd.gsm8k_gold("blah\n#### 1,234") == "1234"
    assert fd.gsm8k_gold("x\n#### -5") == "-5"
    with pytest.raises(ValueError):
        fd.gsm8k_gold("no marker")


def test_corpus_dedup_and_title_variants():
    c = fd.Corpus()
    a = c.add_hotpot("T", ["One.", " Two."])
    assert c.add_hotpot("T", ["One.", "Two."]) == a == "hp::T"
    assert c.add_hotpot("T", ["Different."]) == "hp::T::2"
    assert len(c.docs) == 2 and c.docs["hp::T"]["text"] == "One. Two."


@pytest.mark.skipif(not (fd.FROZEN_DIR / "MANIFEST.json").exists(), reason="아직 동결 전")
def test_frozen_files_match_manifest_and_are_valid():
    import json

    manifest = json.loads((fd.FROZEN_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in fd.FILES:
        assert fd.sha256_file(fd.FROZEN_DIR / name) == manifest["files"][name]["sha256"], name
    eval_rows = fd.read_jsonl(fd.FROZEN_DIR / "eval.jsonl")
    dev_rows = fd.read_jsonl(fd.FROZEN_DIR / "dev.jsonl")
    corpus_rows = fd.read_jsonl(fd.FROZEN_DIR / "corpus.jsonl")
    fd.sanity_text(corpus_rows)
    fd.validate(eval_rows, dev_rows, corpus_rows)
