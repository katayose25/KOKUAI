from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

ROLE_RE = re.compile(r"(?:\b(DOCTOR|PATIENT)\s*:|(<doctor>|<patient>))", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]|[^\s]")


def labels(text: str) -> list[str]:
    out: list[str] = []
    for match in ROLE_RE.finditer(text or ""):
        if match.group(1):
            out.append(match.group(1).upper())
        else:
            out.append("DOCTOR" if match.group(2).lower() == "<doctor>" else "PATIENT")
    return out


def strip_role_tags(text: str) -> str:
    return ROLE_RE.sub(" ", text or "")


def normalize_text(text: str) -> str:
    text = strip_role_tags(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cer_units(text: str) -> list[str]:
    return list(re.sub(r"\s+", "", normalize_text(text)))


def wer_units(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    # Japanese has no reliable whitespace in these outputs, so tokenize CJK chars,
    # Latin words, numbers, and punctuation separately.
    return TOKEN_RE.findall(text)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def rate(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, float]:
    dist = edit_distance(ref, hyp)
    denom = max(len(ref), 1)
    return dist, len(ref), dist / denom


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate role tags plus CER/WER for ASR JSONL.")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", default="")
    args = parser.parse_args()

    rows = 0
    rows_with_label = 0
    rows_with_both = 0
    first_label_ok = 0
    prefix_ok = 0
    exact_ok = 0
    cer_edits = 0
    cer_ref_units = 0
    wer_edits = 0
    wer_ref_units = 0

    examples: list[dict] = []
    with args.pred.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = row.get("ref") or row.get("text") or ""
            hyp = row.get("hyp") or ""
            ref_labels = labels(ref)
            hyp_labels = labels(hyp)
            rows += 1
            if hyp_labels:
                rows_with_label += 1
            if {"DOCTOR", "PATIENT"}.issubset(set(hyp_labels)):
                rows_with_both += 1
            if ref_labels and hyp_labels and ref_labels[0] == hyp_labels[0]:
                first_label_ok += 1
            if ref_labels and hyp_labels[: len(ref_labels)] == ref_labels:
                prefix_ok += 1
            if ref_labels and hyp_labels == ref_labels:
                exact_ok += 1

            c_dist, c_ref, _ = rate(cer_units(ref), cer_units(hyp))
            w_dist, w_ref, _ = rate(wer_units(ref), wer_units(hyp))
            cer_edits += c_dist
            cer_ref_units += c_ref
            wer_edits += w_dist
            wer_ref_units += w_ref
            if len(examples) < 5:
                examples.append({"ref": ref, "hyp": hyp, "ref_labels": ref_labels, "hyp_labels": hyp_labels})

    metrics = {
        "rows": rows,
        "rows_with_label": rows_with_label,
        "rows_with_label_pct": pct(rows_with_label, rows),
        "rows_with_both": rows_with_both,
        "rows_with_both_pct": pct(rows_with_both, rows),
        "speaker_first_label_acc": first_label_ok / rows if rows else 0.0,
        "speaker_sequence_prefix_acc": prefix_ok / rows if rows else 0.0,
        "speaker_sequence_exact_acc": exact_ok / rows if rows else 0.0,
        "cer": cer_edits / max(cer_ref_units, 1),
        "wer": wer_edits / max(wer_ref_units, 1),
        "cer_edits": cer_edits,
        "cer_ref_units": cer_ref_units,
        "wer_edits": wer_edits,
        "wer_ref_units": wer_ref_units,
    }

    print(f"{args.pred}")
    print(f"rows {rows}")
    print(f"rows_with_label {rows_with_label} {metrics['rows_with_label_pct']:.1f}%")
    print(f"rows_with_both {rows_with_both} {metrics['rows_with_both_pct']:.1f}%")
    print(f"first_label_acc {first_label_ok} / {rows} {pct(first_label_ok, rows):.1f}%")
    print(f"label_sequence_prefix_ok {prefix_ok} / {rows} {pct(prefix_ok, rows):.1f}%")
    print(f"label_sequence_exact_ok {exact_ok} / {rows} {pct(exact_ok, rows):.1f}%")
    print(f"CER {metrics['cer']:.4f} edits={cer_edits} ref_chars={cer_ref_units}")
    print(f"WER {metrics['wer']:.4f} edits={wer_edits} ref_tokens={wer_ref_units}")

    if args.wandb_project and args.wandb_mode != "disabled":
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001
            print(f"wandb unavailable: {exc}")
            return
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.pred.stem,
            mode=args.wandb_mode,
            tags=[tag for tag in args.wandb_tags.split(",") if tag],
            config={"pred": str(args.pred)},
        )
        run.log({f"eval/{key}": value for key, value in metrics.items()})
        try:
            table = wandb.Table(columns=["ref", "hyp", "ref_labels", "hyp_labels"])
            for item in examples:
                table.add_data(item["ref"], item["hyp"], " ".join(item["ref_labels"]), " ".join(item["hyp_labels"]))
            run.log({"eval/examples": table})
        except Exception as exc:  # noqa: BLE001
            print(f"wandb examples table skipped: {exc}")
        run.finish()


if __name__ == "__main__":
    main()
