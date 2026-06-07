#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_id(row: dict) -> str:
    value = str(row.get("source_id") or row.get("id") or row.get("audio") or "")
    if "_chunk_" in value:
        value = value.rsplit("_chunk_", 1)[0]
    return value


def stable_order(items: list[str], seed: int) -> list[str]:
    salt = str(seed).encode("utf-8")
    return sorted(items, key=lambda item: hashlib.sha1(salt + item.encode("utf-8")).hexdigest())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict]) -> None:
    sources = {source_id(row) for row in rows}
    turn_counts = Counter(int(row.get("turn_count") or 0) for row in rows)
    both_roles = sum(
        1
        for row in rows
        if "DOCTOR:" in str(row.get("text", "")) and "PATIENT:" in str(row.get("text", ""))
    )
    pct = both_roles / max(len(rows), 1) * 100
    print(
        f"{name}: rows={len(rows)} sources={len(sources)} "
        f"both_roles={both_roles} ({pct:.1f}%) "
        f"turn_counts={dict(turn_counts.most_common(8))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL manifest by source_id/dialogue.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-sources", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[source_id(row)].append(row)

    sources = stable_order(sorted(by_source), args.seed)
    if args.val_sources > 0:
        val_count = min(len(sources), args.val_sources)
    else:
        val_count = max(1, round(len(sources) * args.val_ratio))
    val_sources = set(sources[:val_count])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for src in sorted(by_source):
        if src in val_sources:
            val_rows.extend(by_source[src])
        else:
            train_rows.extend(by_source[src])

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.val_output, val_rows)

    print(f"source_total={len(sources)} val_sources={len(val_sources)} seed={args.seed}")
    summarize("train", train_rows)
    summarize("val", val_rows)


if __name__ == "__main__":
    main()
