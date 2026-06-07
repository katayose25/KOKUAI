from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval_role_tag_asr import cer_units, labels, rate, wer_units


def load_manifest(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                out[line_no] = json.loads(line)
    return out


def bucket_for(meta: dict) -> list[str]:
    roles = [str(role).lower() for role in (meta.get("roles") or [])]
    turn_count = int(meta.get("turn_count") or len(roles) or 0)
    buckets = ["all"]
    buckets.append("turn_count_1" if turn_count == 1 else "turn_count_ge2")
    if {"doctor", "patient"}.issubset(set(roles)):
        buckets.append("both_roles")
    elif roles:
        buckets.append(f"single_role_{roles[0]}")
    return buckets


def blank_stats() -> dict:
    return {
        "rows": 0,
        "rows_with_label": 0,
        "rows_with_both": 0,
        "first_label_ok": 0,
        "prefix_ok": 0,
        "exact_ok": 0,
        "cer_edits": 0,
        "cer_ref_units": 0,
        "wer_edits": 0,
        "wer_ref_units": 0,
    }


def update(stats: dict, ref: str, hyp: str) -> None:
    ref_labels = labels(ref)
    hyp_labels = labels(hyp)
    stats["rows"] += 1
    if hyp_labels:
        stats["rows_with_label"] += 1
    if {"DOCTOR", "PATIENT"}.issubset(set(hyp_labels)):
        stats["rows_with_both"] += 1
    if ref_labels and hyp_labels and ref_labels[0] == hyp_labels[0]:
        stats["first_label_ok"] += 1
    if ref_labels and hyp_labels[: len(ref_labels)] == ref_labels:
        stats["prefix_ok"] += 1
    if ref_labels and hyp_labels == ref_labels:
        stats["exact_ok"] += 1
    c_dist, c_ref, _ = rate(cer_units(ref), cer_units(hyp))
    w_dist, w_ref, _ = rate(wer_units(ref), wer_units(hyp))
    stats["cer_edits"] += c_dist
    stats["cer_ref_units"] += c_ref
    stats["wer_edits"] += w_dist
    stats["wer_ref_units"] += w_ref


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def summarize(name: str, stats: dict) -> dict:
    rows = stats["rows"]
    return {
        "bucket": name,
        "rows": rows,
        "rows_with_label_pct": pct(stats["rows_with_label"], rows),
        "rows_with_both_pct": pct(stats["rows_with_both"], rows),
        "first_label_acc": stats["first_label_ok"] / rows if rows else 0.0,
        "label_sequence_prefix_acc": stats["prefix_ok"] / rows if rows else 0.0,
        "label_sequence_exact_acc": stats["exact_ok"] / rows if rows else 0.0,
        "cer": stats["cer_edits"] / max(stats["cer_ref_units"], 1),
        "wer": stats["wer_edits"] / max(stats["wer_ref_units"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified role-tag ASR eval by manifest turn_count/roles.")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    groups = defaultdict(blank_stats)
    with args.pred.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            line_no = int(row.get("manifest_line") or 0)
            meta = manifest.get(line_no, {})
            ref = row.get("ref") or meta.get("text") or ""
            hyp = row.get("hyp") or ""
            for bucket in bucket_for(meta):
                update(groups[bucket], ref, hyp)

    order = ["all", "turn_count_1", "turn_count_ge2", "both_roles", "single_role_doctor", "single_role_patient"]
    summaries = [summarize(name, groups[name]) for name in order if groups[name]["rows"]]
    for item in summaries:
        print(
            f"{item['bucket']}: rows={item['rows']} "
            f"first={item['first_label_acc']*100:.1f}% "
            f"prefix={item['label_sequence_prefix_acc']*100:.1f}% "
            f"exact={item['label_sequence_exact_acc']*100:.1f}% "
            f"CER={item['cer']:.4f} WER={item['wer']:.4f} "
            f"labels={item['rows_with_label_pct']:.1f}% both_hyp={item['rows_with_both_pct']:.1f}%"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
