from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from .layers import LoRAConv1D, LoRALinear


@dataclass
class LoRAInjectionResult:
    replaced_modules: list[str]


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_to_model(
    model: nn.Module,
    target_module_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> LoRAInjectionResult:
    suffixes = tuple(target_module_suffixes)
    replaced = []

    for module_name, module in list(model.named_modules()):
        if not module_name.endswith(suffixes):
            continue

        parent, child_name = _get_parent_module(model, module_name)

        if isinstance(module, nn.Linear):
            replacement = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        elif isinstance(module, Conv1D):
            replacement = LoRAConv1D(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            continue

        setattr(parent, child_name, replacement)
        replaced.append(module_name)

    if not replaced:
        raise ValueError(
            f"No modules were replaced. Check target suffixes: {list(target_module_suffixes)}"
        )

    return LoRAInjectionResult(replaced_modules=replaced)


def freeze_non_lora_parameters(model: nn.Module, extra_trainable_keywords: Iterable[str] = ()) -> None:
    keep = tuple(extra_trainable_keywords)
    for name, p in model.named_parameters():
        p.requires_grad = False
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        if keep and any(k in name for k in keep):
            p.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
