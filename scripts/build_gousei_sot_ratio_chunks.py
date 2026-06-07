#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from build_voicevox_sot_turn_chunks import (
    Segment,
    rel_from_manifest,
    resolve_audio,
    slice_audio,
    sot_text,
    split_long_segment,
    turn_to_segments,
)


@dataclass(frozen=True)
class Candidate:
    source_index: int
    source_id: str
    chunk_type: str
    segments: tuple[Segment, ...]

    @property
    def start(self) -> float:
        return float(self.segments[0].start)

    @property
    def end(self) -> float:
        return float(self.segments[-1].end)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def roles(self) -> list[str]:
        return [seg.role for seg in self.segments]

    @property
    def both_roles(self) -> bool:
        return len(set(self.roles)) >= 2


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_segments(row: dict, *, max_sec: float, target_sec: float) -> list[Segment]:
    segments: list[Segment] = []
    for turn_index, turn in enumerate(row.get("turns") or []):
        for seg in turn_to_segments(turn, turn_index=turn_index, max_sec=max_sec):
            segments.extend(split_long_segment(seg, max_sec=max_sec, target_sec=target_sec))
    return [seg for seg in sorted(segments, key=lambda item: (item.start, item.end)) if seg.duration > 0 and seg.text.strip()]


def spans_ok(items: list[Segment], *, min_sec: float, max_sec: float, allow_short: bool = False) -> bool:
    if not items:
        return False
    duration = float(items[-1].end - items[0].start)
    if duration > max_sec:
        return False
    if not allow_short and duration < min_sec:
        return False
    return True


def make_candidates_for_source(
    *,
    source_index: int,
    source_id: str,
    segments: list[Segment],
    min_sec: float,
    max_sec: float,
    longer_max_sec: float,
) -> dict[str, list[Candidate]]:
    buckets = {"single": [], "two_turn": [], "multi_turn": [], "longer_mixed": []}

    # Single-turn: allow short utterances because many practical chunks are short answers.
    for seg in segments:
        if seg.duration <= max_sec:
            buckets["single"].append(Candidate(source_index, source_id, "single", (seg,)))

    n = len(segments)
    for i in range(n):
        pair = segments[i : i + 2]
        if len(pair) == 2 and len({seg.role for seg in pair}) >= 2 and spans_ok(pair, min_sec=min(1.0, min_sec), max_sec=max_sec, allow_short=True):
            buckets["two_turn"].append(Candidate(source_index, source_id, "two_turn", tuple(pair)))

        for width in (3, 4, 5):
            group = segments[i : i + width]
            if len(group) != width or len({seg.role for seg in group}) < 2:
                continue
            if spans_ok(group, min_sec=min_sec, max_sec=max_sec):
                buckets["multi_turn"].append(Candidate(source_index, source_id, "multi_turn", tuple(group)))
            elif width >= 4 and spans_ok(group, min_sec=max_sec, max_sec=longer_max_sec, allow_short=True):
                buckets["longer_mixed"].append(Candidate(source_index, source_id, "longer_mixed", tuple(group)))

    return buckets


def candidate_key(c: Candidate) -> tuple:
    return (c.source_id, c.chunk_type, tuple((s.source_turn_index, s.part_index, round(s.start, 3), round(s.end, 3)) for s in c.segments))


def choose_candidates(
    candidates_by_type: dict[str, list[Candidate]],
    *,
    rows_count: int,
    chunks_per_source: int,
    ratios: dict[str, float],
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    target_total = rows_count * chunks_per_source
    quotas = {name: int(round(target_total * ratio)) for name, ratio in ratios.items()}
    # Fix rounding drift on the largest bucket.
    drift = target_total - sum(quotas.values())
    quotas["single"] += drift

    selected: list[Candidate] = []
    selected_keys: set[tuple] = set()
    for name, quota in quotas.items():
        pool = list(candidates_by_type.get(name, []))
        rng.shuffle(pool)
        take = min(quota, len(pool))
        for cand in pool[:take]:
            key = candidate_key(cand)
            if key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(key)
        if take < quota:
            print(f"WARNING: {name} quota={quota} available={len(pool)} selected={take}", flush=True)

    # Keep output deterministic and source-grouped enough for debug.
    return sorted(selected, key=lambda c: (c.source_index, c.start, c.end, c.chunk_type))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ratio-controlled SOT chunks from gousei timed VOICEVOX manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-audio-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--chunks-per-source", type=int, default=10)
    parser.add_argument("--single-ratio", type=float, default=0.40)
    parser.add_argument("--two-turn-ratio", type=float, default=0.35)
    parser.add_argument("--multi-turn-ratio", type=float, default=0.20)
    parser.add_argument("--longer-ratio", type=float, default=0.05)
    parser.add_argument("--target-sec", type=float, default=14.0)
    parser.add_argument("--max-sec", type=float, default=18.0)
    parser.add_argument("--min-sec", type=float, default=8.0)
    parser.add_argument("--longer-max-sec", type=float, default=24.0)
    parser.add_argument("--pad-sec", type=float, default=0.05)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--limit-sources", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    if args.limit_sources:
        rows = rows[: args.limit_sources]

    ratios = {
        "single": args.single_ratio,
        "two_turn": args.two_turn_ratio,
        "multi_turn": args.multi_turn_ratio,
        "longer_mixed": args.longer_ratio,
    }
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {ratio_sum}")

    all_candidates = {"single": [], "two_turn": [], "multi_turn": [], "longer_mixed": []}
    source_audio: dict[str, Path] = {}
    source_row: dict[str, dict] = {}
    for row_idx, row in enumerate(rows, 1):
        source_id = str(row.get("id") or Path(str(row.get("audio", f"source_{row_idx}"))).stem)
        audio_path = resolve_audio(str(row["audio"]), args.manifest)
        if not audio_path.exists():
            raise FileNotFoundError(f"missing audio for {source_id}: {audio_path}")
        segments = source_segments(row, max_sec=args.max_sec, target_sec=args.target_sec)
        buckets = make_candidates_for_source(
            source_index=row_idx,
            source_id=source_id,
            segments=segments,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            longer_max_sec=args.longer_max_sec,
        )
        for name, items in buckets.items():
            all_candidates[name].extend(items)
        source_audio[source_id] = audio_path
        source_row[source_id] = row

    for name, items in all_candidates.items():
        print(f"candidates {name}={len(items)}", flush=True)

    selected = choose_candidates(
        all_candidates,
        rows_count=len(rows),
        chunks_per_source=args.chunks_per_source,
        ratios=ratios,
        seed=args.seed,
    )

    args.output_audio_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    audio_cache: dict[str, tuple[torch.Tensor, int]] = {}
    written = 0
    counts: dict[str, int] = {}
    starts: dict[str, int] = {}
    with args.output_manifest.open("w", encoding="utf-8") as out:
        for cand_idx, cand in enumerate(selected):
            source_id = cand.source_id
            if source_id not in audio_cache:
                wav, sr = torchaudio.load(str(source_audio[source_id]))
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != args.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
                    sr = args.sample_rate
                audio_cache[source_id] = (wav, sr)
                # Bound cache growth a bit; manifests are source-sorted, so this stays small in practice.
                if len(audio_cache) > 3:
                    for key in list(audio_cache)[:-2]:
                        audio_cache.pop(key, None)
            wav, sr = audio_cache[source_id]
            start = max(0.0, cand.start - args.pad_sec)
            end = cand.end + args.pad_sec
            audio = slice_audio(wav, sr, start, end)
            if audio.numel() == 0:
                continue
            chunk_duration = audio.shape[-1] / sr
            if chunk_duration < 1.0:
                continue
            chunk_id = f"{source_id}_{cand.chunk_type}_{cand_idx:05d}"
            out_audio = args.output_audio_root / f"{chunk_id}.wav"
            torchaudio.save(str(out_audio), audio, sr)
            text = sot_text(list(cand.segments))
            row = source_row[source_id]
            roles = cand.roles
            manifest_row = {
                "id": chunk_id,
                "source_id": source_id,
                "audio": rel_from_manifest(out_audio, args.output_manifest),
                "text": text,
                "sot_target": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_sec": round(chunk_duration, 3),
                "chunk_type": cand.chunk_type,
                "turn_count": len(cand.segments),
                "both_roles": cand.both_roles,
                "starts_with": roles[0] if roles else "",
                "roles": roles,
                "triage_label": row.get("triage_label", "unknown"),
                "doctor_speaker": row.get("doctor_speaker"),
                "doctor_speaker_name": row.get("doctor_speaker_name", ""),
                "patient_speaker": row.get("patient_speaker"),
                "patient_speaker_name": row.get("patient_speaker_name", ""),
                "segments": [
                    {
                        "role": seg.role,
                        "text": seg.text,
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "source_turn_index": seg.source_turn_index,
                        "part_index": seg.part_index,
                    }
                    for seg in cand.segments
                ],
            }
            out.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            written += 1
            counts[cand.chunk_type] = counts.get(cand.chunk_type, 0) + 1
            if roles:
                starts[roles[0]] = starts.get(roles[0], 0) + 1
            if written % 1000 == 0:
                print(f"wrote chunks={written}", flush=True)

    print(f"done sources={len(rows)} chunks={written} manifest={args.output_manifest}", flush=True)
    print("chunk_type_counts", counts, flush=True)
    print("starts_with_counts", starts, flush=True)


if __name__ == "__main__":
    main()
