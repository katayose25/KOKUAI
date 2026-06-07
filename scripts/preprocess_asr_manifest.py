from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import datasets
from datasets import Features, Sequence, Value
from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment


def enable_local_model_dirs() -> None:
    from pathlib import Path as _Path

    import liquid_audio.processor as _processor

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
        return original_processor_get(repo_id, revision=revision)

    _processor.get_model_dir = _local_or_hf


def iter_asr_messages(manifest_path: Path, system_prompt: str) -> Iterator[list[ChatMessage]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path

            text = str(row["text"]).strip()
            if not text:
                raise ValueError(f"{manifest_path}:{line_no}: empty text")
            if not audio_path.exists():
                raise FileNotFoundError(f"{manifest_path}:{line_no}: missing audio: {audio_path}")

            yield [
                ChatMessage(role="system", content=[TextSegment(text=system_prompt)]),
                ChatMessage(role="user", content=[AudioSegment(audio=audio_path.read_bytes())]),
                ChatMessage(role="assistant", content=[TextSegment(text=text)]),
            ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an ASR JSONL manifest for LFM2-Audio finetuning.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL with {'audio': path, 'text': transcript}.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for datasets.save_to_disk().")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--system-prompt", default="Perform ASR.")
    parser.add_argument("--special-tokens", default="", help="Comma-separated extra tokens to add to processor.text before mapping, e.g. <doctor>,<patient>.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    enable_local_model_dirs()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)

    processor = LFM2AudioProcessor.from_pretrained(args.model, device=args.device).eval()
    tokenizer = processor.text
    special_tokens = [tok.strip() for tok in args.special_tokens.split(",") if tok.strip()]
    special_token_ids: dict[str, list[int]] = {}
    added_special_tokens = 0
    if special_tokens:
        before_vocab = len(tokenizer)
        added_special_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = {tok: tokenizer.encode(tok, add_special_tokens=False) for tok in special_tokens}
        non_atomic = {tok: ids for tok, ids in special_token_ids.items() if len(ids) != 1}
        if non_atomic:
            raise ValueError(f"Special tokens are not atomic after add_special_tokens: {non_atomic}")
        print(
            f"special_tokens={special_tokens} added={added_special_tokens} "
            f"vocab_before={before_vocab} vocab_after={len(tokenizer)} ids={special_token_ids}",
            flush=True,
        )
    mapper = LFM2AudioChatMapper(processor)

    rows = []
    for i, messages in enumerate(iter_asr_messages(args.manifest, args.system_prompt)):
        sample = mapper(messages)
        sample_len = int(sample.modality_flag.shape[-1])
        if 0 <= args.max_context_length < sample_len:
            print(f"WARNING: skipping sample {i} with {sample_len} tokens (max_context_length={args.max_context_length})")
            continue

        rows.append(
            {
                "text": sample.text.tolist(),
                "audio_in": sample.audio_in.tolist(),
                "audio_in_lens": sample.audio_in_lens.tolist(),
                "audio_out": sample.audio_out.tolist(),
                "modality_flag": sample.modality_flag.tolist(),
                "supervision_mask": sample.supervision_mask.tolist(),
            }
        )

        if (i + 1) % 50 == 0:
            print(f"preprocessed {i + 1} samples", flush=True)

    if not rows:
        raise ValueError(
            "No samples were preprocessed. All rows may have exceeded "
            f"--max-context-length={args.max_context_length}. "
            "Increase --max-context-length or use shorter audio/chunk-level targets."
        )

    features = Features(
        {
            "text": Sequence(Sequence(Value("int64"))),
            "audio_in": Sequence(Sequence(Value("float32"))),
            "audio_in_lens": Sequence(Value("int64")),
            "audio_out": Sequence(Sequence(Value("int64"))),
            "modality_flag": Sequence(Sequence(Value("int64"))),
            "supervision_mask": Sequence(Sequence(Value("bool"))),
        }
    )
    datasets.Dataset.from_list(rows, features=features).save_to_disk(args.output)
    if special_tokens:
        tokenizer_dir = args.output / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        (args.output / "special_tokens.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "system_prompt": args.system_prompt,
                    "special_tokens": special_tokens,
                    "special_token_ids": special_token_ids,
                    "added_special_tokens": added_special_tokens,
                    "tokenizer_dir": str(tokenizer_dir),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"saved {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
