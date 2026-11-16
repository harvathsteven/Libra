#!/usr/bin/env python3
"""Sample BC items from mirlab/TRAIT (CC-BY-4.0). Requires HF auth for gated access."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stimuli" / "bc_items.jsonl"
SEED = 42
DATASET = "mirlab/TRAIT"
DATASET_REVISION = "8b31c078cb897c3917d2ee48735d0c15030680e0"


def main() -> None:
    token_path = ROOT / ".hf_token"
    token = None
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        login(token=token, add_to_git_credential=False)
    elif os.environ.get("HF_TOKEN"):
        token = os.environ["HF_TOKEN"]

    rng = random.Random(SEED)
    out: list[dict] = []
    for trait, split in [("E", "Extraversion"), ("A", "Agreeableness")]:
        ds = load_dataset(DATASET, split=split, token=token, revision=DATASET_REVISION)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        for rank, i in enumerate(idxs[:100]):
            row = ds[i]
            out.append(
                {
                    "item_id": f"TRAIT-{trait}-{i:04d}",
                    "source": DATASET,
                    "source_revision": DATASET_REVISION,
                    "source_split": split,
                    "source_index": int(i),
                    "sample_rank": rank,
                    "sample_seed": SEED,
                    "trait": trait,
                    "question": row["question"],
                    "response_high1": row["response_high1"],
                    "response_high2": row["response_high2"],
                    "response_low1": row["response_low1"],
                    "response_low2": row["response_low2"],
                    "license_note": "CC-BY-4.0 annotations; cite Lee et al. arXiv:2406.14703",
                }
            )

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} items to {OUT}")


if __name__ == "__main__":
    main()
