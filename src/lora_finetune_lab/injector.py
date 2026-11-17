"""
Model adaptation utilities: Replace targeted Linear layers with LoRALinear and report parameter savings.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
from .lora_layer import LoRALinear


def apply_lora_to_model(
    model: nn.Module,
    target_names: List[str] = ["q_proj", "v_proj"],
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> nn.Module:
    """
    Recursively traverse model and replace linear layers matching target_names with LoRALinear.
    Also ensures base model parameters are frozen (requires_grad=False).
    """
    # First, freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False

    # Find and replace targeted modules
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear):
                # Check if matches target name (e.g. q_proj, v_proj, fc)
                if any(target in child_name for target in target_names):
                    lora_layer = LoRALinear.from_linear(
                        child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, child_name, lora_layer)

    return model


def get_trainable_parameters_report(model: nn.Module) -> Dict[str, Any]:
    """
    Return statistics on total vs trainable parameters and efficiency ratio.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable_params / all_params if all_params > 0 else 0.0

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": ratio,
        "frozen_params": all_params - trainable_params,
    }
