#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


ROLE_PREFIX = {
    "doctor": "<doctor>",
    "patient": "<patient>",
}


@dataclass
class Segment:
    role: str
    text: str
    start: float
    end: float
    source_turn_index: int
    part_index: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_audio(path: str, manifest: Path) -> Path:
    audio = Path(path)
    if audio.is_absolute():
        return audio
    cwd_path = Path.cwd() / audio
    if cwd_path.exists():
        return cwd_path
    return manifest.parent / audio


def rel_from_manifest(path: Path, manifest_out: Path) -> str:
    return os.path.relpath(path.resolve(), manifest_out.parent.resolve())


def turn_to_segments(turn: dict, turn_index: int, max_sec: float) -> list[Segment]:
    role = str(turn.get("role") or "").lower()
    if role not in ROLE_PREFIX:
        return []
    text = str(turn.get("text") or "").strip()
    start = float(turn.get("start") or 0.0)
    end = float(turn.get("end") or start)
    if not text:
        return []

    duration = max(0.0, end - start)
    parts = turn.get("parts") or []
    if duration <= max_sec or not parts:
        return [Segment(role=role, text=text, start=start, end=end, source_turn_index=turn_index)]

    segments: list[Segment] = []
    bucket: list[dict] = []
    for part_index, part in enumerate(parts):
        p_text = str(part.get("text") or "").strip()
        p_start = float(part.get("start") or start)
        p_end = float(part.get("end") or p_start)
        if not p_text:
            continue
        if not bucket:
            bucket.append({"text": p_text, "start": p_start, "end": p_end, "part_index": part_index})
            continue
        next_start = float(bucket[0]["start"])
        next_end = p_end
        if next_end - next_start > max_sec:
            segments.append(
                Segment(
                    role=role,
                    text="".join(str(item["text"]) for item in bucket),
                    start=float(bucket[0]["start"]),
                    end=float(bucket[-1]["end"]),
                    source_turn_index=turn_index,
                    part_index=int(bucket[0]["part_index"]),
                )
            )
            bucket = [{"text": p_text, "start": p_start, "end": p_end, "part_index": part_index}]
        else:
            bucket.append({"text": p_text, "start": p_start, "end": p_end, "part_index": part_index})
    if bucket:
        segments.append(
            Segment(
                role=role,
                text="".join(str(item["text"]) for item in bucket),
                start=float(bucket[0]["start"]),
                end=float(bucket[-1]["end"]),
                source_turn_index=turn_index,
                part_index=int(bucket[0]["part_index"]),
            )
        )
    return segments



def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?])", text) if part.strip()]
    if len(parts) > 1:
        return parts
    # Fallback for long punctuation-light text. Keep chunks semantic enough for ASR targets.
    width = 45
    return [text[i : i + width].strip() for i in range(0, len(text), width) if text[i : i + width].strip()]


def split_long_segment(seg: Segment, max_sec: float, target_sec: float) -> list[Segment]:
    if seg.duration <= max_sec:
        return [seg]
    sentences = split_sentences(seg.text)
    if len(sentences) <= 1:
        return [seg]

    total_chars = max(1, sum(len(item) for item in sentences))
    sentence_bounds: list[tuple[str, float, float]] = []
    cursor = seg.start
    for sent in sentences:
        dur = seg.duration * len(sent) / total_chars
        sentence_bounds.append((sent, cursor, cursor + dur))
        cursor += dur
    # Snap the final boundary to the source end to avoid drift.
    sentence_bounds[-1] = (sentence_bounds[-1][0], sentence_bounds[-1][1], seg.end)

    out: list[Segment] = []
    bucket: list[tuple[str, float, float]] = []
    for item in sentence_bounds:
        if not bucket:
            bucket = [item]
            continue
        candidate_end = item[2]
        if candidate_end - bucket[0][1] > max_sec and bucket[-1][2] - bucket[0][1] >= min(target_sec, max_sec):
            out.append(
                Segment(
                    role=seg.role,
                    text="".join(part[0] for part in bucket),
                    start=bucket[0][1],
                    end=bucket[-1][2],
                    source_turn_index=seg.source_turn_index,
                    part_index=seg.part_index,
                )
            )
            bucket = [item]
        else:
            bucket.append(item)
    if bucket:
        out.append(
            Segment(
                role=seg.role,
                text="".join(part[0] for part in bucket),
                start=bucket[0][1],
                end=bucket[-1][2],
                source_turn_index=seg.source_turn_index,
                part_index=seg.part_index,
            )
        )
    return out

def make_chunks(segments: list[Segment], min_sec: float, target_sec: float, max_sec: float) -> list[list[Segment]]:
    chunks: list[list[Segment]] = []
    current: list[Segment] = []

    def span(items: list[Segment]) -> float:
        return max(0.0, items[-1].end - items[0].start) if items else 0.0

    for seg in segments:
        if not current:
            current = [seg]
            if seg.duration >= max_sec:
                chunks.append(current)
                current = []
            continue

        candidate = current + [seg]
        candidate_span = span(candidate)
        current_span = span(current)
        should_close = (
            current_span >= target_sec
            or (candidate_span > max_sec and (current_span >= min_sec or seg.duration >= min_sec))
        )
        if should_close:
            chunks.append(current)
            current = [seg]
            if seg.duration >= max_sec:
                chunks.append(current)
                current = []
        else:
            current = candidate

    if current:
        if chunks and span(current) < min_sec and span(chunks[-1] + current) <= max_sec:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


def slice_audio(wav: torch.Tensor, sr: int, start: float, end: float) -> torch.Tensor:
    start_frame = max(0, int(round(start * sr)))
    end_frame = min(wav.shape[-1], int(round(end * sr)))
    if end_frame <= start_frame:
        return wav[..., :0]
    return wav[..., start_frame:end_frame]


def sot_text(segments: list[Segment]) -> str:
    pieces: list[str] = []
    for seg in segments:
        pieces.append(f"{ROLE_PREFIX[seg.role]} {seg.text.strip()}")
    return " ".join(piece for piece in pieces if piece.strip()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build turn-bounded SOT chunks from timed VoiceVox BeTraC manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-audio-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target-sec", type=float, default=14.0)
    parser.add_argument("--max-sec", type=float, default=18.0)
    parser.add_argument("--min-sec", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit-sources", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    if args.limit_sources:
        rows = rows[: args.limit_sources]

    args.output_audio_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    long_single = 0
    with args.output_manifest.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(rows, 1):
            source_id = str(row.get("id") or Path(str(row.get("audio", f"source_{row_idx}"))).stem)
            audio_path = resolve_audio(str(row["audio"]), args.manifest)
            if not audio_path.exists():
                raise FileNotFoundError(f"missing audio for {source_id}: {audio_path}")

            segments: list[Segment] = []
            for turn_index, turn in enumerate(row.get("turns") or []):
                for seg in turn_to_segments(turn, turn_index=turn_index, max_sec=args.max_sec):
                    segments.extend(split_long_segment(seg, max_sec=args.max_sec, target_sec=args.target_sec))
            segments = [seg for seg in sorted(segments, key=lambda item: (item.start, item.end)) if seg.duration > 0 and seg.text.strip()]
            if not segments:
                continue

            wav, sr = torchaudio.load(str(audio_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != args.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
                sr = args.sample_rate

            chunks = make_chunks(segments, min_sec=args.min_sec, target_sec=args.target_sec, max_sec=args.max_sec)
            for chunk_idx, chunk in enumerate(chunks):
                start = float(chunk[0].start)
                end = float(chunk[-1].end)
                audio = slice_audio(wav, sr, start, end)
                if audio.numel() == 0:
                    continue
                chunk_duration = audio.shape[-1] / sr
                if chunk_duration > args.max_sec + 0.01:
                    long_single += 1

                chunk_id = f"{source_id}_sot_{chunk_idx:04d}"
                out_audio = args.output_audio_root / f"{chunk_id}.wav"
                torchaudio.save(str(out_audio), audio, sr)
                text = sot_text(chunk)
                manifest_row = {
                    "id": chunk_id,
                    "source_id": source_id,
                    "audio": rel_from_manifest(out_audio, args.output_manifest),
                    "text": text,
                    "sot_target": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration_sec": round(chunk_duration, 3),
                    "turn_count": len(chunk),
                    "roles": [seg.role for seg in chunk],
                    "segments": [
                        {
                            "role": seg.role,
                            "text": seg.text,
                            "start": round(seg.start, 3),
                            "end": round(seg.end, 3),
                            "source_turn_index": seg.source_turn_index,
                            "part_index": seg.part_index,
                        }
                        for seg in chunk
                    ],
                }
                out.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
                written += 1
            if row_idx % 50 == 0:
                print(f"processed sources={row_idx} chunks={written}", flush=True)

    print(
        f"done sources={len(rows)} chunks={written} long_chunks_over_max={long_single} "
        f"manifest={args.output_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
