from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torchaudio
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from hackathon_app.services.lora_utils import LoraConfig, inject_lora, load_adapter_lora_state_dict


def enable_local_model_dirs() -> None:
    from pathlib import Path as _Path

    import liquid_audio.model.lfm2_audio as _lfm2_audio
    import liquid_audio.processor as _processor

    original_model_get = _lfm2_audio.get_model_dir
    original_processor_get = _processor.get_model_dir

    def _local_or_hf(repo_id, revision=None):
        path = _Path(repo_id)
        if path.exists():
            return path.resolve()
        try:
            path = _Path(str(repo_id)).expanduser()
            if path.exists():
                return path.resolve()
        except Exception:
            pass
        return original_model_get(repo_id, revision=revision)

    def _local_or_hf_processor(repo_id, revision=None):
        path = _Path(repo_id)
        if path.exists():
            return path.resolve()
        try:
            path = _Path(str(repo_id)).expanduser()
            if path.exists():
                return path.resolve()
        except Exception:
            pass
        return original_processor_get(repo_id, revision=revision)

    _lfm2_audio.get_model_dir = _local_or_hf
    _processor.get_model_dir = _local_or_hf_processor


def load_adapter_checkpoint(model: torch.nn.Module, processor: LFM2AudioProcessor, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("special_token_metadata") or {}
    special_tokens = metadata.get("special_tokens") or []
    if special_tokens:
        added = processor.text.add_special_tokens({"additional_special_tokens": special_tokens})
        ids = {tok: processor.text.encode(tok, add_special_tokens=False) for tok in special_tokens}
        print(f"loaded checkpoint special tokens={special_tokens} added_to_processor={added} ids={ids}")
    state_dict = checkpoint.get("state_dict", checkpoint)

    if checkpoint.get("mode") == "audio_adapter_lora":
        cfg = checkpoint.get("lora_config", {})
        injected = inject_lora(
            model,
            LoraConfig(
                rank=int(cfg.get("rank", 8)),
                alpha=float(cfg.get("alpha", 16.0)),
                dropout=float(cfg.get("dropout", 0.0)),
            ),
        )
        missing, unexpected = load_adapter_lora_state_dict(model, state_dict)
        print(f"injected lora modules={len(injected)}")
        print(f"loaded adapter+lora tensors={len(state_dict)} from {checkpoint_path}")
    elif checkpoint.get("mode") == "audio_adapter_partial_lfm":
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        trained_keys = list(state_dict)
        missing = [key for key in missing if key in trained_keys]
        print(f"loaded partial tensors={len(trained_keys)} from {checkpoint_path}")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        adapter_keys = [key for key in state_dict if "audio_adapter" in key]
        if not adapter_keys:
            raise ValueError(f"No audio_adapter weights found in {checkpoint_path}")
        missing = [key for key in missing if "audio_adapter" in key]
        print(f"loaded adapter tensors={len(adapter_keys)} from {checkpoint_path}")

    rows = checkpoint.get("special_token_embedding_rows")
    if rows:
        embedding = model.lfm.get_input_embeddings().weight
        with torch.no_grad():
            for token_id, value in rows.items():
                embedding[int(token_id)].copy_(value.to(device=embedding.device, dtype=embedding.dtype))
        print(f"loaded special token embedding rows={len(rows)}")

    if unexpected:
        print(f"unexpected keys: {unexpected}")
    if missing:
        print(f"missing trained keys: {missing}")



ROLE_RE = re.compile(r"(?:\b(DOCTOR|PATIENT)\s*:|(<doctor>|<patient>))\s*", re.IGNORECASE)


def parse_role_turns(text: str) -> list[tuple[str, str]]:
    matches = list(ROLE_RE.finditer(text or ""))
    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        if match.group(1):
            role = match.group(1).upper()
        else:
            role = "DOCTOR" if match.group(2).lower() == "<doctor>" else "PATIENT"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = re.sub(r"\s+", " ", text[start:end]).strip(" \n\t:-")
        if content:
            turns.append((role, content))
    return turns


def build_context_prompt(base_prompt: str, mode: str, previous_turn: tuple[str, str] | None) -> str:
    if mode == "none" or previous_turn is None:
        return base_prompt
    role, content = previous_turn
    if mode == "last-speaker":
        return f"{base_prompt}\n\nPrevious final speaker: {role}."
    if mode == "last-turn":
        content = content[:240]
        return f"{base_prompt}\n\nPrevious final turn:\n{role}: {content}"
    raise ValueError(f"unknown context mode: {mode}")


def transcribe(
    *,
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    audio_path: Path,
    device: str,
    max_new_tokens: int,
    system_prompt: str,
    stream: bool = True,
) -> str:
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(system_prompt)
    chat.end_turn()

    chat.new_turn("user")
    wav, sampling_rate = torchaudio.load(str(audio_path))
    chat.add_audio(wav.to(device), sampling_rate)
    chat.end_turn()

    chat.new_turn("assistant")

    text_ids: list[int] = []
    streamed_text = ""
    with torch.inference_mode():
        for token in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
            if token.numel() == 1:
                token_id = int(token.item())
                if token_id == 7:  # <|im_end|>
                    break
                text_ids.append(token_id)
                if stream:
                    decoded = processor.text.decode(torch.tensor(text_ids, dtype=torch.long))
                    # Byte-level BPE can split Japanese UTF-8 bytes across tokens.
                    # Avoid printing temporary replacement chars that cannot be retracted.
                    if "�" not in decoded and decoded.startswith(streamed_text):
                        delta = decoded[len(streamed_text):]
                        if delta:
                            print(delta, end="", flush=True)
                        streamed_text = decoded
    decoded = processor.text.decode(torch.tensor(text_ids, dtype=torch.long)) if text_ids else ""
    if stream:
        if not decoded.startswith(streamed_text):
            print(decoded, end="", flush=True)
        else:
            tail = decoded[len(streamed_text):]
            if tail:
                print(tail, end="", flush=True)
        print()
    return decoded


def resolve_audio_path(audio_path: Path, manifest_path: Path) -> Path:
    if audio_path.is_absolute():
        return audio_path

    cwd_path = Path.cwd() / audio_path
    if cwd_path.exists():
        return cwd_path

    return manifest_path.parent / audio_path


def iter_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_path = resolve_audio_path(Path(row["audio"]), manifest_path)
            yield line_no, audio_path, row.get("text", ""), row.get("source_id") or row.get("id", "").rsplit("_chunk_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LFM2-Audio ASR with an optional trained audio_adapter checkpoint.")
    parser.add_argument("--audio", type=Path, help="Single audio file to transcribe.")
    parser.add_argument("--manifest", type=Path, help="JSONL manifest with audio/text rows.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1, help="Split manifest rows into N deterministic shards for parallel ASR.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this 0-based shard index.")
    parser.add_argument("--skip-existing", action="store_true", help="When --output exists, skip audio paths already written and append new rows.")
    parser.add_argument("--checkpoint", type=Path, help="Adapter checkpoint .pt saved by train_lfm2_audio_asr_adapter.py.")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--system-prompt", default="Perform ASR.")
    parser.add_argument("--context-mode", choices=["none", "last-speaker", "last-turn"], default="none", help="Add previous predicted chunk context to the system prompt in manifest mode.")
    parser.add_argument("--quiet", action="store_true", help="Do not stream generated transcript text to stdout; print progress only.")
    parser.add_argument("--progress-interval", type=int, default=25, help="In --quiet mode, print progress every N completed rows.")
    parser.add_argument("--output", type=Path, help="Optional JSONL output path for manifest mode.")
    args = parser.parse_args()

    enable_local_model_dirs()

    if not args.audio and not args.manifest:
        raise SystemExit("pass --audio or --manifest")

    processor = LFM2AudioProcessor.from_pretrained(args.model, device=args.device).eval()
    model = LFM2AudioModel.from_pretrained(args.model, device=args.device, dtype=torch.bfloat16).eval()

    if args.checkpoint:
        load_adapter_checkpoint(model, processor, args.checkpoint)
        model.to(device=args.device, dtype=torch.bfloat16)
        model.eval()

    if args.audio:
        print(f"audio: {args.audio}")
        hyp = transcribe(
            model=model,
            processor=processor,
            audio_path=args.audio,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
            stream=not args.quiet,
        )
        if args.quiet:
            print(f"HYP: {hyp}")
        return

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    existing_audio: set[str] = set()
    out_f = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and args.output.exists():
            with args.output.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            existing_audio.add(str(json.loads(line).get("audio", "")))
                        except json.JSONDecodeError:
                            pass
        mode = "a" if args.skip_existing else "w"
        out_f = args.output.open(mode, encoding="utf-8")

    try:
        emitted = 0
        previous_source_id: str | None = None
        previous_turn: tuple[str, str] | None = None
        for manifest_idx, (line_no, audio_path, ref, source_id) in enumerate(iter_manifest(args.manifest)):
            if manifest_idx % args.num_shards != args.shard_index:
                continue
            if str(audio_path) in existing_audio:
                continue
            emitted += 1
            if emitted > args.limit:
                break
            if source_id != previous_source_id:
                previous_turn = None
            prompt = build_context_prompt(args.system_prompt, args.context_mode, previous_turn)
            if args.quiet:
                print(f"[{emitted}] line={line_no} audio={audio_path} context={args.context_mode}", flush=True)
            else:
                print(f"\n[{emitted}] line={line_no} audio={audio_path}")
                if ref:
                    print(f"REF: {ref}")
                if args.context_mode != "none" and previous_turn is not None:
                    print(f"CONTEXT_PROMPT: {prompt}")
                print("HYP: ", end="", flush=True)
            hyp = transcribe(
                model=model,
                processor=processor,
                audio_path=audio_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                system_prompt=prompt,
                stream=not args.quiet,
            )
            hyp_turns = parse_role_turns(hyp)
            if hyp_turns:
                previous_turn = hyp_turns[-1]
                previous_source_id = source_id
            if out_f:
                out_f.write(
                    json.dumps(
                        {
                            "audio": str(audio_path),
                            "ref": ref,
                            "hyp": hyp,
                            "manifest_line": line_no,
                            "shard_index": args.shard_index,
                            "num_shards": args.num_shards,
                            "context_mode": args.context_mode,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out_f.flush()
            if args.quiet and emitted % max(args.progress_interval, 1) == 0:
                print(f"progress completed={emitted}", flush=True)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
