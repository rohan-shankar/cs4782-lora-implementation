from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        low_rank = F.linear(self.dropout(x), self.lora_A)
        delta = F.linear(low_rank, self.lora_B)
        return base_out + delta * self.scaling


class LoRAConv1D(nn.Module):
    """
    LoRA wrapper for GPT-style Conv1D (weight shape [in_features, out_features]).
    """

    def __init__(self, base: Conv1D, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.in_features = base.weight.shape[0]
        self.out_features = base.weight.shape[1]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight.transpose(0, 1), self.bias)
        low_rank = F.linear(self.dropout(x), self.lora_A)
        delta = F.linear(low_rank, self.lora_B)
        return base_out + delta * self.scaling
