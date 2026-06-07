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


ROLE_RE = re.compile(r"\b(DOCTOR|PATIENT)\s*:", re.IGNORECASE)

SAFE_PRIMARY_SPEAKERS = {
    2: "四国めたん",
    3: "ずんだもん",
    8: "春日部つむぎ",
    10: "雨晴はう",
    9: "波音リツ",
    11: "玄野武宏",
    12: "白上虎太郎",
    14: "冥鳴ひまり",
    16: "九州そら",
    21: "剣崎雌雄",
    23: "WhiteCUL",
    42: "ちび式じい",
    43: "櫻歌ミコ",
    46: "小夜/SAYO",
    47: "ナースロボ＿タイプＴ",
    51: "†聖騎士 紅桜†",
    52: "雀松朱司",
    53: "麒ヶ島宗麟",
    54: "春歌ナナ",
    55: "猫使アル",
    58: "猫使ビィ",
    61: "中国うさぎ",
    67: "栗田まろん",
    68: "あいえるたん",
    69: "満別花丸",
    74: "琴詠ニア",
    94: "中部つるぎ",
    99: "離途",
    100: "黒沢冴白",
    107: "東北ずん子",
    108: "東北きりたん",
    109: "東北イタコ",
    113: "あんこもん",
    118: "夜語トバリ",
    122: "暁記ミタマ",
    126: "里石ユカ（つぼみ）",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize Japanese BeTraC translations with VOICEVOX Engine.")
    parser.add_argument("--input", required=True, help="JSONL with id and ja_text fields.")
    parser.add_argument("--output-audio-root", required=True, help="Directory for synthesized wav files.")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL manifest.")
    parser.add_argument("--engine-url", default="http://127.0.0.1:50021", help="VOICEVOX Engine URL.")
    parser.add_argument("--doctor-speaker", type=int, default=8, help="VOICEVOX speaker id for DOCTOR turns.")
    parser.add_argument("--patient-speaker", type=int, default=3, help="VOICEVOX speaker id for PATIENT turns.")
    parser.add_argument("--unknown-speaker", type=int, default=3, help="VOICEVOX speaker id for unlabeled text.")
    parser.add_argument("--randomize-speakers", action="store_true", help="Randomly choose doctor/patient speakers per dialogue.")
    parser.add_argument("--speaker-pool", default="", help="Comma-separated speaker IDs used for both roles when randomizing.")
    parser.add_argument("--doctor-speaker-pool", default="", help="Comma-separated speaker IDs used for doctor role when randomizing.")
    parser.add_argument("--patient-speaker-pool", default="", help="Comma-separated speaker IDs used for patient role when randomizing.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for speaker assignment.")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--pitch-scale", type=float, default=0.0)
    parser.add_argument("--intonation-scale", type=float, default=1.0)
    parser.add_argument("--volume-scale", type=float, default=1.0)
    parser.add_argument("--pre-phoneme-length", type=float, default=0.1)
    parser.add_argument("--post-phoneme-length", type=float, default=0.1)
    parser.add_argument("--turn-pause-ms", type=int, default=350, help="Inserted silence between dialogue turns.")
    parser.add_argument("--part-pause-ms", type=int, default=80, help="Inserted silence between split parts of a long turn.")
    parser.add_argument("--max-chars", type=int, default=180, help="Split long turns before synthesis.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--error-output", default="")
    parser.add_argument("--list-speakers", action="store_true", help="Print available speakers and exit.")
    return parser.parse_args()


def role_turns(text: str) -> list[tuple[str, str]]:
    matches = list(ROLE_RE.finditer(text or ""))
    if not matches:
        cleaned = normalize_text(text)
        return [("unknown", cleaned)] if cleaned else []

    turns: list[tuple[str, str]] = []
    prefix = normalize_text(text[: matches[0].start()])
    if prefix:
        turns.append(("unknown", prefix))
    for idx, match in enumerate(matches):
        role = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = normalize_text(text[start:end])
        if content:
            turns.append(("doctor" if role == "DOCTOR" else "patient", content))
    return turns


def normalize_text(text: str) -> str:
    text = (text or "").replace("*", "")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \n\t:：-")


def split_text(text: str, max_chars: int) -> list[str]:
    text = normalize_text(text)
    if not text or len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    current = ""
    for part in re.split(r"(?<=[。！？!?])", text):
        part = part.strip()
        if not part:
            continue
        if current and len(current) + len(part) > max_chars:
            pieces.extend(split_hard(current, max_chars))
            current = part
        else:
            current = f"{current}{part}" if current else part
    if current:
        pieces.extend(split_hard(current, max_chars))
    return pieces


def split_hard(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        chunks.append(text[cursor : cursor + max_chars].strip())
        cursor += max_chars
    return [chunk for chunk in chunks if chunk]


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


def parse_speaker_pool(text: str) -> list[int]:
    if not text.strip():
        return []
    speakers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            speakers.append(int(part))
    return speakers


def choose_dialogue_speakers(args: argparse.Namespace, rng: random.Random) -> tuple[int, int]:
    if not args.randomize_speakers:
        return args.doctor_speaker, args.patient_speaker

    shared_pool = parse_speaker_pool(args.speaker_pool) or list(SAFE_PRIMARY_SPEAKERS)
    doctor_pool = parse_speaker_pool(args.doctor_speaker_pool) or shared_pool
    patient_pool = parse_speaker_pool(args.patient_speaker_pool) or shared_pool
    if not doctor_pool or not patient_pool:
        raise ValueError("speaker pool is empty")

    doctor = rng.choice(doctor_pool)
    patient_choices = [speaker for speaker in patient_pool if speaker != doctor]
    patient = rng.choice(patient_choices or patient_pool)
    return doctor, patient


def get_json(session: requests.Session, url: str, timeout: float) -> object:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(session: requests.Session, url: str, *, params: dict, json_body: object | None, timeout: float) -> requests.Response:
    response = session.post(url, params=params, json=json_body, timeout=timeout)
    response.raise_for_status()
    return response


def synthesize_text(session: requests.Session, args: argparse.Namespace, text: str, speaker: int) -> bytes:
    base = args.engine_url.rstrip("/")
    query_response = post_json(
        session,
        f"{base}/audio_query",
        params={"text": text, "speaker": speaker},
        json_body=None,
        timeout=args.timeout,
    )
    query = query_response.json()
    for key, value in {
        "speedScale": args.speed_scale,
        "pitchScale": args.pitch_scale,
        "intonationScale": args.intonation_scale,
        "volumeScale": args.volume_scale,
        "prePhonemeLength": args.pre_phoneme_length,
        "postPhonemeLength": args.post_phoneme_length,
    }.items():
        if key in query:
            query[key] = value
    wav_response = post_json(
        session,
        f"{base}/synthesis",
        params={"speaker": speaker},
        json_body=query,
        timeout=args.timeout,
    )
    return wav_response.content


def synthesize_with_retries(session: requests.Session, args: argparse.Namespace, text: str, speaker: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            return synthesize_text(session, args, text, speaker)
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


def read_wav_bytes(data: bytes) -> tuple[wave._wave_params, bytes]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(wav.getnframes())
    return params, frames


def silence_frames(params: wave._wave_params, milliseconds: int) -> bytes:
    if milliseconds <= 0:
        return b""
    frame_count = int(params.framerate * milliseconds / 1000)
    return b"\x00" * frame_count * params.nchannels * params.sampwidth


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


def wav_duration_sec(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def synthesize_dialogue(session: requests.Session, args: argparse.Namespace, row: dict, output_path: Path) -> tuple[float, list[dict]]:
    turns = role_turns(row.get("ja_text", ""))
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
            part_rows.append(
                {
                    "text": part,
                    "start": round(part_start, 3),
                    "end": round(part_end, 3),
                    "duration_sec": round(part_duration, 3),
                    "pause_after_ms": pause_ms,
                }
            )
            cursor_sec += pause_ms / 1000.0
        turn_end = part_rows[-1]["end"] if part_rows else turn_start
        manifest_turns.append(
            {
                "role": role,
                "speaker": speaker,
                "speaker_name": SAFE_PRIMARY_SPEAKERS.get(speaker, ""),
                "start": round(turn_start, 3),
                "end": round(turn_end, 3),
                "text": text,
                "parts": part_rows,
            }
        )

    duration = write_concat_wav(wav_parts, output_path)
    return duration, manifest_turns


def print_speakers(session: requests.Session, args: argparse.Namespace) -> None:
    speakers = get_json(session, f"{args.engine_url.rstrip('/')}/speakers", args.timeout)
    for speaker in speakers:
        name = speaker.get("name", "")
        for style in speaker.get("styles", []):
            print(f"{style.get('id')}\t{name}\t{style.get('name')}")


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line).get("id", ""))
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
            row_id = row.get("id") or f"row_{line_no:06d}"
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
                out_row = {
                    "id": row_id,
                    "audio": str(output_path),
                    "duration_sec": round(duration, 3),
                    "text": row.get("ja_text", ""),
                    "turns": turns,
                    "doctor_speaker": doctor_speaker,
                    "doctor_speaker_name": SAFE_PRIMARY_SPEAKERS.get(doctor_speaker, ""),
                    "patient_speaker": patient_speaker,
                    "patient_speaker_name": SAFE_PRIMARY_SPEAKERS.get(patient_speaker, ""),
                    "source_audio": row.get("audio", ""),
                    "metadata": row.get("metadata", ""),
                    "manifest_line": row.get("manifest_line", line_no),
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
