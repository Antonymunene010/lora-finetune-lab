"""
LoRA Linear Layer implementation from scratch.
Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022).
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA Linear Layer:
    h = W_0 * x + (alpha / r) * B * A * x

    - W_0: Frozen pretrained weights of shape [out_features, in_features]
    - A: Trainable low-rank matrix of shape [r, in_features], initialized with Kaiming uniform
    - B: Trainable low-rank matrix of shape [out_features, r], initialized with zeros
    - alpha: Scaling factor
    - r: Rank dimension
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Base linear layer (frozen during adaptation)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if bias and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self.merged = False

    def reset_parameters(self):
        """
        Initialization:
        - A is initialized with Kaiming uniform.
        - B is initialized with zeros so Delta W = B * A = 0 initially,
          preserving the exact behavior of pretrained model at step 0.
        """
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.05):
        """Construct a LoRALinear layer from an existing nn.Linear instance."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
        )
        layer.base_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.base_layer.bias.data.copy_(linear.bias.data)
        return layer

    def merge(self):
        """
        Merge adapter weights W = W_0 + (alpha / r) * (B @ A) directly into base weights.
        Eliminates inference latency overhead in production deployment.
        """
        if self.r > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Reverse weight merging to resume training or swap adapters."""
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        base_out = self.base_layer(x)

        if self.r == 0 or self.merged:
            return base_out

        # LoRA forward pass: (x @ A^T) @ B^T * scaling
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out
