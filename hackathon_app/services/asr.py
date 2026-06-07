from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from hackathon_app.models import Encounter, TranscriptTurn
from hackathon_app.services.mock_models import transcribe as mock_transcribe


ROLE_LABEL_RE = re.compile(r"(<\s*(doctor|patient)\s*>|\b(DOCTOR|PATIENT)\s*:)", re.IGNORECASE)
ROLE_LABEL_LOOSE_RE = re.compile(r"\b(DOCTOR|PATIENT)\b\s*:?")
ROLE_TO_SPEAKER = {"DOCTOR": "doctor", "PATIENT": "patient"}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

DOCTOR_CUES = (
    "what brings", "how long", "have you", "do you", "can you", "are you", "is it",
    "does that", "tell me", "we need", "i recommend", "i would like", "i want to order",
    "i'm going to", "i am going to", "prescribe", "exam", "blood pressure", "blood work",
    "results", "test", "tests", "symptoms", "follow up", "schedule", "referral",
    "does that sound", "do you understand", "any questions", "let's", "alright", "right",
)
PATIENT_CUES = (
    "doctor", "my ", "i feel", "i have", "i've", "i guess", "i think", "it started",
    "i work", "i don't", "i do not", "i want", "i need", "i was", "i am", "i'm",
    "it hurts", "worried", "thank you", "yes doctor", "no doctor", "not really", "for me",
)
QUESTION_FRAGMENTS = (
    "and what's", "what", "what sort", "how", "how long", "do you", "can you",
    "have you", "are you", "is it", "does that", "tell me", "and what",
)


def _role_tagging_enabled() -> bool:
    checkpoint_configured = bool(
        os.getenv("LFM_AUDIO_CHECKPOINT", "").strip()
        or os.getenv("LFM_AUDIO_CHECKPOINT_REPO", "").strip()
    )
    enabled = os.getenv("LFM_AUDIO_ENABLE_ROLE_TAGS", "1").lower() not in {"0", "false", "no"}
    return checkpoint_configured and enabled


def _resolve_audio_checkpoint() -> str:
    """Return a local adapter .pt path from either a local path or a HF model repo."""
    local_path = os.getenv("LFM_AUDIO_CHECKPOINT", "").strip()
    if local_path:
        return local_path

    repo_id = os.getenv("LFM_AUDIO_CHECKPOINT_REPO", "").strip()
    if not repo_id:
        return ""
    filename = os.getenv("LFM_AUDIO_CHECKPOINT_FILENAME", "adapter_lora_best.pt").strip() or "adapter_lora_best.pt"
    revision = os.getenv("LFM_AUDIO_CHECKPOINT_REVISION", "").strip() or None
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("huggingface_hub is required when LFM_AUDIO_CHECKPOINT_REPO is set") from exc
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def transcribe_encounter(encounter: Encounter) -> list[TranscriptTurn]:
    if not encounter.audio_sources:
        return mock_transcribe(encounter)
    turns = list(transcribe_encounter_stream(encounter))
    return turns or mock_transcribe(encounter)


def transcribe_encounter_stream(encounter: Encounter):
    offset = 0.0
    limit_chunks = int(os.getenv("LFM_AUDIO_LIMIT_CHUNKS", "0"))
    chunk_sec = float(os.getenv("LFM_AUDIO_CHUNK_SEC", "4.0"))
    boundary_silence_sec = float(os.getenv("LFM_AUDIO_TURN_SILENCE_SEC", "2.0"))
    max_turn_sec = float(os.getenv("LFM_AUDIO_MAX_TURN_SEC", "30.0"))
    processed_chunks = 0

    for audio in encounter.audio_sources:
        path = Path(audio.path)
        if not path.exists():
            continue
        turn_start = offset
        last_text_end = offset
        current_texts: list[str] = []

        for chunk in _chunk_audio(path, chunk_sec=chunk_sec):
            if limit_chunks and processed_chunks >= limit_chunks:
                break

            text = _transcribe_lfm(chunk.path).strip()
            processed_chunks += 1
            absolute_start = offset + chunk.start
            absolute_end = offset + chunk.end

            if text:
                if not current_texts:
                    turn_start = absolute_start
                current_texts.append(text)
                last_text_end = absolute_end

            turn_duration = absolute_end - turn_start
            should_finalize = (
                chunk.trailing_silence_sec >= boundary_silence_sec
                or turn_duration >= max_turn_sec
            )
            if should_finalize and current_texts:
                yield from _turns_from_role_text(
                    " ".join(current_texts).strip(),
                    start=turn_start,
                    end=last_text_end,
                )
                current_texts = []
                turn_start = absolute_end
                last_text_end = absolute_end

        if current_texts:
            yield from _turns_from_role_text(
                " ".join(current_texts).strip(),
                start=turn_start,
                end=last_text_end,
            )

        offset += _duration_sec(path)
        if limit_chunks and processed_chunks >= limit_chunks:
            break


def transcribe_chunk_stream(encounter: Encounter):
    offset = 0.0
    limit_chunks = int(os.getenv("LFM_AUDIO_LIMIT_CHUNKS", "0"))
    chunk_sec = float(os.getenv("LFM_AUDIO_CHUNK_SEC", "4.0"))
    processed_chunks = 0

    for audio in encounter.audio_sources:
        path = Path(audio.path)
        if not path.exists():
            continue
        for chunk in _chunk_audio(path, chunk_sec=chunk_sec):
            if limit_chunks and processed_chunks >= limit_chunks:
                break
            text = _transcribe_lfm(chunk.path).strip()
            processed_chunks += 1
            if text:
                yield from _turns_from_role_text(
                    text,
                    start=offset + chunk.start,
                    end=offset + chunk.end,
                )
        offset += _duration_sec(path)
        if limit_chunks and processed_chunks >= limit_chunks:
            break


def _turns_from_role_text(text: str, start: float | None, end: float | None):
    if not _role_tagging_enabled():
        cleaned = _strip_role_labels(text)
        if cleaned:
            yield TranscriptTurn(speaker="unknown", start=start, end=end, text=cleaned)
        return

    parsed = _parse_role_tagged_text(text)
    if not parsed:
        yield TranscriptTurn(speaker="unknown", start=start, end=end, text=_strip_role_labels(text))
        return

    parsed = _postprocess_role_segments(parsed)
    spans = _proportional_spans(parsed, start, end)
    for (speaker, piece), (piece_start, piece_end) in zip(parsed, spans):
        cleaned = piece.strip()
        if cleaned:
            yield TranscriptTurn(speaker=speaker, start=piece_start, end=piece_end, text=cleaned)


def normalize_transcript_turns(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Clean display turns after appending chunk-level ASR output."""
    items = [(turn.speaker, turn.text, turn.start, turn.end) for turn in turns if turn.text.strip()]
    items = _absorb_short_question_items(items)
    items = _merge_same_speaker_items(items)
    return [
        TranscriptTurn(speaker=speaker, text=text.strip(), start=start, end=end)
        for speaker, text, start, end in items
        if text.strip()
    ]


def _postprocess_role_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    segments = _maybe_flip_chunk_roles(segments)
    segments = _absorb_short_question_segments(segments)
    return _merge_same_speaker_segments(segments)


def _maybe_flip_chunk_roles(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if os.getenv("LFM_AUDIO_ROLE_FLIP", "1").lower() in {"0", "false", "no"}:
        return segments
    if not any(speaker == "doctor" for speaker, _text in segments) or not any(speaker == "patient" for speaker, _text in segments):
        return segments

    margin = float(os.getenv("LFM_AUDIO_ROLE_FLIP_MARGIN", "1.5"))
    current_score = _role_assignment_score(segments)
    flipped = [(_opposite_speaker(speaker), text) for speaker, text in segments]
    flipped_score = _role_assignment_score(flipped)
    if flipped_score > current_score + margin:
        return flipped
    return segments


def _role_assignment_score(segments: list[tuple[str, str]]) -> float:
    score = 0.0
    for speaker, text in segments:
        doctor = _cue_score(text, DOCTOR_CUES)
        patient = _cue_score(text, PATIENT_CUES)
        if speaker == "doctor":
            score += doctor - patient
        elif speaker == "patient":
            score += patient - doctor
    return score


def _cue_score(text: str, cues: tuple[str, ...]) -> float:
    lowered = f" {text.lower()} "
    score = 0.0
    for cue in cues:
        count = lowered.count(cue)
        if count:
            score += count
    # Short question-leading fragments are usually clinician turns.
    stripped = lowered.strip()
    if any(stripped.startswith(fragment) for fragment in QUESTION_FRAGMENTS):
        score += 1.0 if cues is DOCTOR_CUES else 0.0
    return score


def _opposite_speaker(speaker: str) -> str:
    if speaker == "doctor":
        return "patient"
    if speaker == "patient":
        return "doctor"
    return speaker


def _merge_same_speaker_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for speaker, text in segments:
        text = text.strip()
        if not text:
            continue
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, _join_text(merged[-1][1], text))
        else:
            merged.append((speaker, text))
    return merged


def _absorb_short_question_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    items = [(speaker, text, None, None) for speaker, text in segments]
    return [(speaker, text) for speaker, text, _start, _end in _absorb_short_question_items(items)]


def _merge_same_speaker_items(items: list[tuple[str, str, float | None, float | None]]) -> list[tuple[str, str, float | None, float | None]]:
    merged: list[tuple[str, str, float | None, float | None]] = []
    for speaker, text, start, end in items:
        text = text.strip()
        if not text:
            continue
        if merged and merged[-1][0] == speaker:
            prev_speaker, prev_text, prev_start, _prev_end = merged[-1]
            merged[-1] = (prev_speaker, _join_text(prev_text, text), prev_start, end)
        else:
            merged.append((speaker, text, start, end))
    return merged


def _absorb_short_question_items(items: list[tuple[str, str, float | None, float | None]]) -> list[tuple[str, str, float | None, float | None]]:
    result: list[tuple[str, str, float | None, float | None]] = []
    idx = 0
    while idx < len(items):
        speaker, text, start, end = items[idx]
        if idx + 1 < len(items) and _is_short_question_fragment(text):
            next_speaker, next_text, _next_start, next_end = items[idx + 1]
            result.append((speaker, _join_text(text, next_text), start, next_end))
            idx += 2
            continue
        words = _word_count(text)
        if 0 < words <= 2 and result:
            prev_speaker, prev_text, prev_start, _prev_end = result[-1]
            result[-1] = (prev_speaker, _join_text(prev_text, text), prev_start, end)
            idx += 1
            continue
        result.append((speaker, text, start, end))
        idx += 1
    return result


def _is_short_question_fragment(text: str) -> bool:
    lowered = text.lower().strip()
    words = _word_count(lowered)
    return 0 < words <= 5 and any(lowered.startswith(fragment) for fragment in QUESTION_FRAGMENTS)


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _join_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}".strip()


def _parse_role_tagged_text(text: str) -> list[tuple[str, str]]:
    matches = list(ROLE_LABEL_RE.finditer(text))
    if not matches:
        # Accept rare model outputs like "DOCTOR Good morning" only at line starts.
        loose = []
        for line_start in (0, *[m.end() for m in re.finditer(r"\n+", text)]):
            match = ROLE_LABEL_LOOSE_RE.match(text, line_start)
            if match:
                loose.append(match)
        matches = loose
    if not matches:
        return []

    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        role = (match.group(2) or match.group(3) or match.group(1)).upper()
        speaker = ROLE_TO_SPEAKER.get(role)
        if speaker is None:
            continue
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip(" \n\t:-")
        content = _strip_role_labels(content)
        if content:
            turns.append((speaker, content))
    return turns


def _strip_role_labels(text: str) -> str:
    text = ROLE_LABEL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _proportional_spans(turns: list[tuple[str, str]], start: float | None, end: float | None) -> list[tuple[float | None, float | None]]:
    if start is None or end is None or end <= start or len(turns) <= 1:
        return [(start, end) for _ in turns]
    total_chars = sum(max(1, len(text)) for _speaker, text in turns)
    cursor = start
    spans: list[tuple[float | None, float | None]] = []
    for idx, (_speaker, text) in enumerate(turns):
        if idx == len(turns) - 1:
            piece_end = end
        else:
            piece_end = cursor + (end - start) * max(1, len(text)) / total_chars
        spans.append((round(cursor, 3), round(piece_end, 3)))
        cursor = piece_end
    return spans


class AudioChunk:
    def __init__(self, start: float, end: float, path: Path, trailing_silence_sec: float) -> None:
        self.start = start
        self.end = end
        self.path = path
        self.trailing_silence_sec = trailing_silence_sec


@lru_cache(maxsize=1)
def _load_lfm() -> tuple[object, object, str, int, str]:
    import torch
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    from hackathon_app.services.lfm2_audio_with_adapter import enable_local_model_dirs, load_adapter_checkpoint

    enable_local_model_dirs()
    model_name = os.getenv("LFM_AUDIO_MODEL", "LiquidAI/LFM2.5-Audio-1.5B")
    device = os.getenv("LFM_AUDIO_DEVICE", "cuda:0")
    max_new_tokens = int(os.getenv("LFM_AUDIO_MAX_NEW_TOKENS", "96"))
    checkpoint = _resolve_audio_checkpoint()
    if checkpoint:
        if "JP" in model_name.upper():
            default_prompt = "Perform ASR in japanese. Use <doctor> and <patient> speaker prefix tokens."
        else:
            default_prompt = "Perform ASR. Include speaker role labels exactly as DOCTOR: and PATIENT:."
        system_prompt = os.getenv("LFM_AUDIO_SYSTEM_PROMPT", default_prompt)
    else:
        default_prompt = "Perform ASR in japanese." if "JP" in model_name.upper() else "Perform ASR."
        system_prompt = os.getenv("LFM_AUDIO_SYSTEM_PROMPT", default_prompt)

    processor = LFM2AudioProcessor.from_pretrained(model_name, device=device).eval()
    model = LFM2AudioModel.from_pretrained(model_name, device=device, dtype=torch.bfloat16).eval()
    if checkpoint:
        load_adapter_checkpoint(model, processor, Path(checkpoint))
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
    return model, processor, device, max_new_tokens, system_prompt


def _transcribe_lfm(audio_path: Path) -> str:
    from hackathon_app.services.lfm2_audio_with_adapter import transcribe as lfm_transcribe_audio

    model, processor, device, max_new_tokens, system_prompt = _load_lfm()
    return lfm_transcribe_audio(
        model=model,
        processor=processor,
        audio_path=audio_path,
        device=device,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        stream=False,
    )


def _transcribe_lfm_stream(audio_path: Path):
    """Yield ASR text pieces using the same full-transcription path as batch evaluation."""
    full_text = _transcribe_lfm(audio_path)
    for char in full_text:
        yield char


def transcribe_chunk_token_stream(encounter: Encounter):
    """Transcribe chunks and yield parsed turns for UI token playback."""
    offset = 0.0
    limit_chunks = int(os.getenv("LFM_AUDIO_LIMIT_CHUNKS", "0"))
    chunk_sec = float(os.getenv("LFM_AUDIO_CHUNK_SEC", "4.0"))
    processed_chunks = 0

    for audio in encounter.audio_sources:
        path = Path(audio.path)
        if not path.exists():
            continue
        for chunk in _chunk_audio(path, chunk_sec=chunk_sec):
            if limit_chunks and processed_chunks >= limit_chunks:
                break
            pieces = list(_transcribe_lfm_stream(chunk.path))
            processed_chunks += 1
            text = "".join(pieces).strip()
            if text:
                yield from _turns_from_role_text(
                    text,
                    start=offset + chunk.start,
                    end=offset + chunk.end,
                )
        offset += _duration_sec(path)
        if limit_chunks and processed_chunks >= limit_chunks:
            break


def _chunk_audio(path: Path, chunk_sec: float) -> list[AudioChunk]:
    mode = os.getenv("LFM_AUDIO_CHUNK_MODE", "silence").lower()
    total = _duration_sec(path)
    if mode in {"fixed", "window", "windows"}:
        boundaries = _fixed_chunk_boundaries(total, chunk_sec)
    else:
        boundaries = _silence_chunk_boundaries(path, total, chunk_sec)

    chunks: list[AudioChunk] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="clinical-asr-"))
    for idx, (start, end) in enumerate(boundaries):
        duration = end - start
        if duration < 1.0:
            continue
        out = temp_dir / f"{path.stem}_chunk_{idx:04d}.wav"
        _cut_wav(path, out, start, duration)
        trailing_silence = _trailing_silence_sec(out)
        chunks.append(AudioChunk(start=start, end=end, path=out, trailing_silence_sec=trailing_silence))
    return chunks


def _fixed_chunk_boundaries(total: float, chunk_sec: float) -> list[tuple[float, float]]:
    boundaries: list[tuple[float, float]] = []
    start = 0.0
    while start < total:
        end = min(start + chunk_sec, total)
        if end - start >= 1.0:
            boundaries.append((start, end))
        start = end
    return boundaries


def _silence_chunk_boundaries(path: Path, total: float, chunk_sec: float) -> list[tuple[float, float]]:
    search_before = float(os.getenv("LFM_AUDIO_SILENCE_SEARCH_BEFORE_SEC", "10.0"))
    min_silence = float(os.getenv("LFM_AUDIO_MIN_SILENCE_SEC", "0.2"))
    merge_short_tail = float(os.getenv("LFM_AUDIO_MERGE_SHORT_TAIL_SEC", "8.0"))
    top_db = float(os.getenv("LFM_AUDIO_SILENCE_TOP_DB", "35.0"))
    silences = _find_silence_intervals(path, min_silence_sec=min_silence, top_db=top_db)

    boundaries: list[tuple[float, float]] = []
    start = 0.0
    while start < total:
        target = min(start + chunk_sec, total)
        if target >= total:
            boundaries.append((start, total))
            break

        earliest = max(start + 1.0, target - search_before)
        cut = _choose_silence_cut(silences, earliest=earliest, latest=target)
        if cut is None or cut <= start + 0.5:
            cut = target
        boundaries.append((start, min(cut, total)))
        start = min(cut, total)

    return _merge_short_tail_boundaries(boundaries, merge_short_tail_sec=merge_short_tail)


def _choose_silence_cut(
    silences: list[tuple[float, float]],
    *,
    earliest: float,
    latest: float,
) -> float | None:
    candidates: list[tuple[float, float]] = []
    for silence_start, silence_end in silences:
        overlap_start = max(silence_start, earliest)
        overlap_end = min(silence_end, latest)
        if overlap_end <= overlap_start:
            continue
        midpoint = (overlap_start + overlap_end) / 2.0
        candidates.append((abs(latest - midpoint), midpoint))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _merge_short_tail_boundaries(
    boundaries: list[tuple[float, float]],
    *,
    merge_short_tail_sec: float,
) -> list[tuple[float, float]]:
    if len(boundaries) < 2:
        return boundaries
    last_start, last_end = boundaries[-1]
    if last_end - last_start >= merge_short_tail_sec:
        return boundaries
    prev_start, _prev_end = boundaries[-2]
    return [*boundaries[:-2], (prev_start, last_end)]


def _find_silence_intervals(
    path: Path,
    *,
    min_silence_sec: float,
    top_db: float,
) -> list[tuple[float, float]]:
    wav, sample_rate = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if wav.numel() == 0:
        return []

    frame_size = max(1, int(0.03 * sample_rate))
    hop_size = max(1, int(0.01 * sample_rate))
    if wav.numel() < frame_size:
        return []

    frames = wav.unfold(0, frame_size, hop_size)
    rms = torch.sqrt(torch.mean(frames.square(), dim=1) + 1e-12)
    db = 20.0 * torch.log10(rms + 1e-12)
    threshold = float(db.max().item()) - top_db
    silent = db <= threshold

    intervals: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_silent in enumerate(silent.tolist()):
        if is_silent and start_idx is None:
            start_idx = idx
        elif not is_silent and start_idx is not None:
            start = start_idx * hop_size / sample_rate
            end = min((idx * hop_size + frame_size) / sample_rate, wav.numel() / sample_rate)
            if end - start >= min_silence_sec:
                intervals.append((start, end))
            start_idx = None

    if start_idx is not None:
        start = start_idx * hop_size / sample_rate
        end = wav.numel() / sample_rate
        if end - start >= min_silence_sec:
            intervals.append((start, end))

    return intervals


def _trailing_silence_sec(path: Path, top_db: float | None = None) -> float:
    top_db = float(os.getenv("LFM_AUDIO_SILENCE_TOP_DB", str(top_db or 35.0)))
    wav, sample_rate = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if wav.numel() == 0:
        return 0.0

    frame_size = max(1, int(0.03 * sample_rate))
    hop_size = max(1, int(0.01 * sample_rate))
    if wav.numel() < frame_size:
        return 0.0

    frames = wav.unfold(0, frame_size, hop_size)
    rms = torch.sqrt(torch.mean(frames.square(), dim=1) + 1e-12)
    db = 20.0 * torch.log10(rms + 1e-12)
    threshold = float(db.max().item()) - top_db
    voiced = db > threshold
    voiced_indices = torch.nonzero(voiced, as_tuple=False).flatten()
    if voiced_indices.numel() == 0:
        return wav.numel() / sample_rate
    last_voiced = int(voiced_indices[-1].item())
    last_voiced_end = min((last_voiced * hop_size + frame_size) / sample_rate, wav.numel() / sample_rate)
    return max(0.0, wav.numel() / sample_rate - last_voiced_end)


def _cut_wav(src: Path, dst: Path, start: float, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ],
        check=True,
    )


def _duration_sec(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate
