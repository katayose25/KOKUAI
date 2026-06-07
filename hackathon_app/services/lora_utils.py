from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def _get_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model: nn.Module, config: LoraConfig) -> list[str]:
    replaced: list[str] = []
    modules = list(model.named_modules())
    for name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("lfm.layers."):
            continue
        if not any(target in name for target in config.target_modules):
            continue
        parent, child_name = _get_parent(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout))
        replaced.append(name)
    return replaced


def mark_trainable_adapter_and_lora(model: nn.Module) -> list[str]:
    trainable: list[str] = []
    for name, param in model.named_parameters():
        should_train = "audio_adapter" in name or ".lora_a." in name or ".lora_b." in name
        param.requires_grad = should_train
        if should_train:
            trainable.append(name)
    return trainable


def adapter_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "audio_adapter" in name or ".lora_a." in name or ".lora_b." in name:
            out[name] = param.detach().cpu()
    return out


def load_adapter_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    relevant_missing = [k for k in missing if "audio_adapter" in k or ".lora_a." in k or ".lora_b." in k]
    return relevant_missing, list(unexpected)
