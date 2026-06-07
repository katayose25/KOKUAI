#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import re
import time
import wave
from pathlib import Path
from typing import Iterable

import requests

from synthesize_betrac_ja_with_voicevox import (
    SAFE_PRIMARY_SPEAKERS,
    choose_dialogue_speakers,
    get_json,
    normalize_text,
    parse_speaker_pool,
    print_speakers,
    read_wav_bytes,
    silence_frames,
    split_text,
    synthesize_with_retries,
    wav_duration_sec,
)

TAG_RE = re.compile(r"(<doctor>|<patient>)\s*", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize cleaned gousei disaster dialogues with VOICEVOX Engine.")
    parser.add_argument("--input", required=True, help="dataset_clean.jsonl with turns[] and text fields.")
    parser.add_argument("--output-audio-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--engine-url", default="http://127.0.0.1:50021")
    parser.add_argument("--doctor-speaker", type=int, default=11)
    parser.add_argument("--patient-speaker", type=int, default=8)
    parser.add_argument("--unknown-speaker", type=int, default=8)
    parser.add_argument("--randomize-speakers", action="store_true")
    parser.add_argument("--speaker-pool", default="")
    parser.add_argument("--doctor-speaker-pool", default="")
    parser.add_argument("--patient-speaker-pool", default="")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--speed-scale", type=float, default=1.05)
    parser.add_argument("--pitch-scale", type=float, default=0.0)
    parser.add_argument("--intonation-scale", type=float, default=1.05)
    parser.add_argument("--volume-scale", type=float, default=1.0)
    parser.add_argument("--pre-phoneme-length", type=float, default=0.08)
    parser.add_argument("--post-phoneme-length", type=float, default=0.08)
    parser.add_argument("--turn-pause-ms", type=int, default=180)
    parser.add_argument("--part-pause-ms", type=int, default=70)
    parser.add_argument("--max-chars", type=int, default=140)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--error-output", default="")
    parser.add_argument("--list-speakers", action="store_true")
    return parser.parse_args()


def row_turns(row: dict) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for turn in row.get("turns") or []:
        role = str(turn.get("role", "")).lower().strip()
        if role not in {"doctor", "patient"}:
            continue
        text = normalize_text(str(turn.get("text", "")))
        if text:
            parsed.append((role, text))
    if parsed:
        return parsed

    text = str(row.get("text", ""))
    matches = list(TAG_RE.finditer(text))
    for idx, match in enumerate(matches):
        role = "doctor" if match.group(1).lower() == "<doctor>" else "patient"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = normalize_text(text[start:end])
        if content:
            parsed.append((role, content))
    return parsed


def speaker_for(role: str, args: argparse.Namespace) -> int:
    if hasattr(args, "current_doctor_speaker") and role == "doctor":
        return args.current_doctor_speaker
    if hasattr(args, "current_patient_speaker") and role == "patient":
        return args.current_patient_speaker
    if role == "doctor":
        return args.doctor_speaker
    if role == "patient":
        return args.patient_speaker
    return args.unknown_speaker


def write_concat_wav(parts: Iterable[tuple[bytes, int]], output_path: Path) -> float:
    params: wave._wave_params | None = None
    frame_chunks: list[bytes] = []
    total_frames = 0
    for wav_bytes, pause_ms in parts:
        current_params, frames = read_wav_bytes(wav_bytes)
        if params is None:
            params = current_params
        elif (
            current_params.nchannels,
            current_params.sampwidth,
            current_params.framerate,
            current_params.comptype,
        ) != (params.nchannels, params.sampwidth, params.framerate, params.comptype):
            raise ValueError("VOICEVOX returned WAV files with inconsistent parameters.")
        frame_chunks.append(frames)
        total_frames += len(frames) // (params.nchannels * params.sampwidth)
        pause = silence_frames(params, pause_ms)
        if pause:
            frame_chunks.append(pause)
            total_frames += len(pause) // (params.nchannels * params.sampwidth)
    if params is None:
        raise ValueError("No audio parts generated.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setparams(params)
        for frames in frame_chunks:
            wav.writeframes(frames)
    return total_frames / params.framerate


def synthesize_dialogue(session: requests.Session, args: argparse.Namespace, row: dict, output_path: Path) -> tuple[float, list[dict]]:
    turns = row_turns(row)
    if not turns:
        raise ValueError("no doctor/patient turns found")

    wav_parts: list[tuple[bytes, int]] = []
    manifest_turns: list[dict] = []
    cursor_sec = 0.0

    for turn_idx, (role, text) in enumerate(turns):
        speaker = speaker_for(role, args)
        text_parts = split_text(text, args.max_chars)
        turn_start = cursor_sec
        part_rows: list[dict] = []
        for part_idx, part in enumerate(text_parts):
            part_start = cursor_sec
            wav = synthesize_with_retries(session, args, part, speaker)
            part_duration = wav_duration_sec(wav)
            cursor_sec += part_duration
            part_end = cursor_sec
            pause_ms = args.part_pause_ms if part_idx + 1 < len(text_parts) else args.turn_pause_ms
            wav_parts.append((wav, pause_ms))
            part_rows.append({
                "text": part,
                "start": round(part_start, 3),
                "end": round(part_end, 3),
                "duration_sec": round(part_duration, 3),
                "pause_after_ms": pause_ms,
            })
            cursor_sec += pause_ms / 1000.0
        turn_end = part_rows[-1]["end"] if part_rows else turn_start
        manifest_turns.append({
            "role": role,
            "speaker": speaker,
            "speaker_name": SAFE_PRIMARY_SPEAKERS.get(speaker, ""),
            "start": round(turn_start, 3),
            "end": round(turn_end, 3),
            "text": text,
            "parts": part_rows,
        })

    duration = write_concat_wav(wav_parts, output_path)
    return duration, manifest_turns


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("id", "")))
            except json.JSONDecodeError:
                pass
    return done


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_audio_root = Path(args.output_audio_root)
    output_manifest = Path(args.output_manifest)
    error_output = Path(args.error_output) if args.error_output else None
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_audio_root.mkdir(parents=True, exist_ok=True)
    if error_output:
        error_output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    if args.list_speakers:
        print_speakers(session, args)
        return

    done_ids = load_done_ids(output_manifest) if args.skip_existing else set()
    rng = random.Random(args.seed)
    processed = 0
    with input_path.open(encoding="utf-8") as src, output_manifest.open("a", encoding="utf-8") as out:
        for line_no, line in enumerate(src, 1):
            if args.limit and processed >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or f"gousei_{line_no:05d}")
            if row_id in done_ids:
                continue
            output_path = output_audio_root / f"{row_id}.wav"
            if args.skip_existing and output_path.exists():
                continue

            print(f"[{line_no}] id={row_id}", flush=True)
            try:
                doctor_speaker, patient_speaker = choose_dialogue_speakers(args, rng)
                args.current_doctor_speaker = doctor_speaker
                args.current_patient_speaker = patient_speaker
                duration, turns = synthesize_dialogue(session, args, row, output_path)
                text = "\n".join(f"<{turn['role']}> {turn['text']}" for turn in turns)
                out_row = {
                    "id": row_id,
                    "audio": str(output_path),
                    "duration_sec": round(duration, 3),
                    "text": text,
                    "turns": turns,
                    "triage_label": row.get("triage_label", "unknown"),
                    "source_line": row.get("source_line", line_no),
                    "source_object_index": row.get("source_object_index", 0),
                    "doctor_speaker": doctor_speaker,
                    "doctor_speaker_name": SAFE_PRIMARY_SPEAKERS.get(doctor_speaker, ""),
                    "patient_speaker": patient_speaker,
                    "patient_speaker_name": SAFE_PRIMARY_SPEAKERS.get(patient_speaker, ""),
                    "manifest_line": line_no,
                }
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out.flush()
                processed += 1
                if args.sleep_sec:
                    time.sleep(args.sleep_sec)
            except Exception as exc:
                print(f"ERROR id={row_id}: {exc}", flush=True)
                if error_output:
                    with error_output.open("a", encoding="utf-8") as err:
                        err.write(json.dumps({"id": row_id, "line": line_no, "error": str(exc)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
