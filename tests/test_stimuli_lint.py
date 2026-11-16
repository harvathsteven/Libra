"""Phase 2 stimulus lint tests."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STIM = ROOT / "stimuli"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _lines(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


@pytest.fixture(scope="module")
def adjectives() -> dict:
    with (STIM / "adjectives.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def banned() -> list[str]:
    return [t.lower() for t in _lines(STIM / "banned_terms.txt")]


def test_adjective_lists_size_and_disjoint(adjectives: dict) -> None:
    for trait in ("extraversion", "agreeableness"):
        for pole in ("high", "low"):
            block = adjectives[trait][pole]
            all_words = block["all"]
            assert len(all_words) >= 16
            extraction = set(block["extraction"])
            test = set(block["test"])
            assert extraction.isdisjoint(test)
            assert extraction | test == set(all_words)
            assert len(extraction) >= 8
            assert len(test) >= 8


def test_question_sets_disjoint_and_counts() -> None:
    extraction = set(_lines(STIM / "extraction_questions.txt"))
    probes = set(_lines(STIM / "probe_questions.txt"))
    assert len(extraction) == 20
    assert len(probes) == 50
    assert extraction.isdisjoint(probes)


def test_ipip_counts() -> None:
    with (STIM / "ipip_neo_120_EA.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 48
    assert sum(1 for r in rows if r["trait"] == "E") == 24
    assert sum(1 for r in rows if r["trait"] == "A") == 24
    facets_e = {r["facet"] for r in rows if r["trait"] == "E"}
    facets_a = {r["facet"] for r in rows if r["trait"] == "A"}
    assert len(facets_e) == 6
    assert len(facets_a) == 6


def test_bc_item_counts() -> None:
    rows = _load_jsonl(STIM / "bc_items.jsonl")
    assert len(rows) == 200
    assert sum(1 for r in rows if r["trait"] == "E") == 100
    assert sum(1 for r in rows if r["trait"] == "A") == 100
    for r in rows:
        assert r["question"]
        assert r["response_high1"] and r["response_high2"]
        assert r["response_low1"] and r["response_low2"]


def test_persona_counts_and_word_limits() -> None:
    test = _load_jsonl(STIM / "personas_test.jsonl")
    # 96 personas + 1 baseline
    assert len(test) == 97
    explicit = [p for p in test if p["set"] == "explicit"]
    implicit = [p for p in test if p["set"] == "implicit"]
    baseline = [p for p in test if p["set"] == "baseline"]
    assert len(explicit) == 32
    assert len(implicit) == 64
    assert len(baseline) == 1
    for p in implicit:
        wc = len(re.findall(r"\b[\w'-]+\b", p["system_prompt"]))
        assert 60 <= wc <= 95, (p["persona_id"], wc)

    dev = _load_jsonl(STIM / "personas_dev.jsonl")
    assert len(dev) == 8
    test_ids = {p["persona_id"] for p in test}
    for p in dev:
        assert p["persona_id"] not in test_ids


def test_banned_terms_absent_from_implicit_and_probes(banned: list[str]) -> None:
    # Longer phrases first for matching
    terms = sorted(banned, key=len, reverse=True)
    implicit = [
        p
        for p in _load_jsonl(STIM / "personas_test.jsonl")
        if p["set"] == "implicit"
    ] + [
        p
        for p in _load_jsonl(STIM / "personas_dev.jsonl")
        if "implicit" in p["set"]
    ]
    texts = [p["system_prompt"].lower() for p in implicit]
    texts += [q.lower() for q in _lines(STIM / "probe_questions.txt")]
    texts += [q.lower() for q in _lines(STIM / "extraction_questions.txt")]

    failures: list[str] = []
    for text in texts:
        for term in terms:
            # word-boundary for single tokens; substring for multiword phrases
            if " " in term:
                if term in text:
                    failures.append(f"{term!r} in {text[:60]!r}…")
            else:
                if re.search(rf"\b{re.escape(term)}\b", text):
                    failures.append(f"{term!r} in {text[:60]!r}…")
    assert not failures, "\n".join(failures[:20])


def test_intended_levels_table_committed() -> None:
    path = STIM / "persona_intended_levels.csv"
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 97 + 8
