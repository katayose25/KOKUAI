from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator
from lora_utils import LoraConfig, adapter_lora_state_dict, inject_lora, mark_trainable_adapter_and_lora


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


def serializable_args(args: argparse.Namespace) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def save_checkpoint(
    *,
    model: torch.nn.Module,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    trainable_names: list[str],
    lora_modules: list[str],
    special_token_ids: list[int] | None = None,
    special_token_metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_id": args.model,
        "mode": "audio_adapter_lora",
        "lora_config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "lora_modules": lora_modules,
        "trainable_names": trainable_names,
        "args": serializable_args(args),
        "state_dict": adapter_lora_state_dict(model),
    }
    if special_token_ids:
        embedding = model.lfm.get_input_embeddings().weight.detach().cpu()
        payload["special_token_ids"] = special_token_ids
        payload["special_token_metadata"] = special_token_metadata or {}
        payload["special_token_embedding_rows"] = {str(token_id): embedding[token_id].clone() for token_id in special_token_ids}
    torch.save(payload, output_dir / f"adapter_lora_step_{step:06d}.pt")




def load_init_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    trained_keys = list(state_dict)
    relevant_missing = [key for key in missing if key in trained_keys]
    adapter_keys = [key for key in trained_keys if "audio_adapter" in key]
    print(f"loaded init checkpoint tensors={len(trained_keys)} from {checkpoint_path}", flush=True)
    if adapter_keys:
        print(f"loaded init audio_adapter tensors={len(adapter_keys)}", flush=True)
    if unexpected:
        print(f"unexpected init keys: {unexpected}", flush=True)
    if relevant_missing:
        print(f"missing init keys: {relevant_missing}", flush=True)
    rows = checkpoint.get("special_token_embedding_rows")
    if rows:
        embedding = model.lfm.get_input_embeddings().weight
        with torch.no_grad():
            for token_id, value in rows.items():
                embedding[int(token_id)].copy_(value.to(device=embedding.device, dtype=embedding.dtype))
        print(f"loaded special token embedding rows={len(rows)}", flush=True)


def load_special_token_metadata(train_data: Path, explicit_ids: str) -> tuple[list[int], dict]:
    if explicit_ids.strip():
        ids = [int(item.strip()) for item in explicit_ids.split(",") if item.strip()]
        return ids, {"source": "args", "special_token_ids": ids}
    path = train_data / "special_tokens.json"
    if not path.exists():
        return [], {}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    token_ids = []
    for ids in (metadata.get("special_token_ids") or {}).values():
        if isinstance(ids, list) and len(ids) == 1:
            token_ids.append(int(ids[0]))
    return sorted(set(token_ids)), metadata


def enable_special_token_embedding_training(model: torch.nn.Module, token_ids: list[int]) -> str | None:
    if not token_ids:
        return None
    embedding = model.lfm.get_input_embeddings().weight
    embedding.requires_grad = True

    def keep_only_special_rows(grad: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(grad)
        ids = torch.tensor(token_ids, device=grad.device, dtype=torch.long)
        out.index_copy_(0, ids, grad.index_select(0, ids))
        return out

    embedding.register_hook(keep_only_special_rows)
    return "lfm.embed_tokens.weight"


def init_wandb(args: argparse.Namespace, config: dict):
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"wandb unavailable: {exc}", flush=True)
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        config=config,
        tags=[tag for tag in args.wandb_tags.split(",") if tag],
    )
    return run


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
):
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for idx, batch in enumerate(loader, 1):
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        losses.append(float(out.loss.detach().float().item()))
        if max_batches and idx >= max_batches:
            break
    if was_training:
        model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapter + LFM LoRA ASR finetuning for LFM2-Audio.")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--output-dir", default="checkpoints/lfm2_audio_ja_adapter_lora", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint to load before LoRA injection, e.g. adapter warmup.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Optimizer-update warmup steps, not micro-batch steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--eval-interval", type=int, default=0, help="Run validation every N micro-batch steps. 0 disables eval.")
    parser.add_argument("--eval-max-batches", type=int, default=0, help="Limit validation batches. 0 means full eval set.")
    parser.add_argument("--save-best", action="store_true", help="Save adapter_lora_best.pt when validation loss improves.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--special-token-ids", default="", help="Comma-separated token ids whose embedding rows should be trained and saved. If empty, reads train-data/special_tokens.json.")
    parser.add_argument("--train-special-token-embeddings", action="store_true", help="Train only the specified special-token embedding rows in addition to adapter/LoRA.")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"] )
    parser.add_argument("--wandb-tags", default="")
    args = parser.parse_args()

    enable_local_model_dirs()

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dataset = LFM2DataLoader(str(args.train_data), context_length=args.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lfm2_collator)
    eval_loader = None
    if args.eval_data:
        eval_dataset = LFM2DataLoader(str(args.eval_data), context_length=args.context_length)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lfm2_collator)

    model = LFM2AudioModel.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    if args.init_checkpoint:
        load_init_checkpoint(model, args.init_checkpoint)
        model.to(device=device, dtype=torch.bfloat16)

    lora_config = LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    lora_modules = inject_lora(model, lora_config)
    if not lora_modules:
        raise RuntimeError("No LoRA modules were injected. Check target module names.")
    model.to(device)
    model.train()

    trainable_names = mark_trainable_adapter_and_lora(model)
    special_token_ids, special_token_metadata = load_special_token_metadata(args.train_data, args.special_token_ids)
    if args.train_special_token_embeddings:
        embedding_name = enable_special_token_embedding_training(model, special_token_ids)
        if embedding_name and embedding_name not in trainable_names:
            trainable_names.append(embedding_name)
        print(f"train_special_token_embeddings ids={special_token_ids}", flush=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    total_update_steps = max(1, math.ceil(args.max_steps / args.grad_accum_steps))
    scheduler = make_lr_scheduler(
        optimizer,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        total_steps=total_update_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_args(args)
    config["lora_modules"] = lora_modules
    config["trainable_names"] = trainable_names
    config["special_token_ids"] = special_token_ids
    config["special_token_metadata"] = special_token_metadata
    (args.output_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    wandb_run = init_wandb(args, config)

    print(f"dataset_size={len(dataset)}", flush=True)
    print(f"lora_modules={len(lora_modules)}", flush=True)
    print(f"trainable_parameters={sum(p.numel() for p in trainable_params):,}", flush=True)
    print(f"trainable_tensors={len(trainable_names)}", flush=True)
    print(f"optimizer=AdamW lr={args.lr} weight_decay={args.weight_decay}", flush=True)
    print(f"lr_scheduler={args.lr_scheduler} warmup_update_steps={args.warmup_steps} total_update_steps={total_update_steps}", flush=True)
    if eval_loader is not None:
        print(f"eval_dataset_size={len(eval_loader.dataset)}", flush=True)

    step = 0
    update_step = 0
    best_eval_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    running_loss = 0.0
    running_count = 0

    while step < args.max_steps:
        for batch in loader:
            step += 1
            batch = batch.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
                loss = out.loss / args.grad_accum_steps

            loss.backward()
            running_loss += float(out.loss.detach().float().item())
            running_count += 1

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if step % args.log_interval == 0:
                elapsed = time.time() - started
                avg_loss = running_loss / max(running_count, 1)
                lr = optimizer.param_groups[0]["lr"]
                loss_value = float(out.loss.detach().float().item())
                print(f"step={step}/{args.max_steps} update={update_step}/{total_update_steps} lr={lr:.6g} loss={loss_value:.4f} avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/loss": loss_value, "train/avg_loss": avg_loss, "train/lr": lr, "train/update_step": update_step}, step=step)
                running_loss = 0.0
                running_count = 0

            if eval_loader is not None and args.eval_interval and step % args.eval_interval == 0:
                eval_loss = evaluate_loss(model=model, loader=eval_loader, device=device, max_batches=args.eval_max_batches)
                print(f"eval step={step}/{args.max_steps} update={update_step}/{total_update_steps} eval_loss={eval_loss:.4f} best_eval_loss={best_eval_loss:.4f}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"eval/loss": eval_loss, "eval/best_loss": best_eval_loss, "eval/update_step": update_step}, step=step)
                if args.save_best and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(model=model, output_dir=args.output_dir, step=step, args=args, trainable_names=trainable_names, lora_modules=lora_modules, special_token_ids=special_token_ids, special_token_metadata=special_token_metadata)
                    best_path = args.output_dir / f"adapter_lora_step_{step:06d}.pt"
                    import shutil
                    shutil.copy2(best_path, args.output_dir / "adapter_lora_best.pt")
                    print(f"saved best checkpoint eval_loss={eval_loss:.4f} path={args.output_dir / 'adapter_lora_best.pt'}", flush=True)

            if step % args.save_interval == 0:
                save_checkpoint(model=model, output_dir=args.output_dir, step=step, args=args, trainable_names=trainable_names, lora_modules=lora_modules, special_token_ids=special_token_ids, special_token_metadata=special_token_metadata)

            if step >= args.max_steps:
                break

    if step % args.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

    save_checkpoint(model=model, output_dir=args.output_dir, step=step, args=args, trainable_names=trainable_names, lora_modules=lora_modules, special_token_ids=special_token_ids, special_token_metadata=special_token_metadata)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved final adapter+lora checkpoint under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
