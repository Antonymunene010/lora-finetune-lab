"""
Unit tests for LoRA linear module, parameter freezing, and zero-latency merging.
"""

import pytest
import torch
import torch.nn as nn
from lora_finetune_lab.lora_layer import LoRALinear
from lora_finetune_lab.injector import apply_lora_to_model, get_trainable_parameters_report


def test_lora_initialization_zero_delta():
    """
    At initialization, LoRA B is zero, meaning Delta W = B @ A = 0.
    The layer output must be identical to the base layer.
    """
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 64)
    lora_linear = LoRALinear.from_linear(base_linear, r=4, lora_alpha=8.0)

    x = torch.randn(4, 32)
    with torch.no_grad():
        out_base = base_linear(x)
        out_lora = lora_linear(x)

    diff = (out_base - out_lora).abs().max().item()
    assert diff < 1e-6, f"LoRA modified outputs at step 0: diff={diff}"


def test_parameter_freezing_and_gradients():
    """
    Ensure base weights are frozen and only LoRA A/B receive gradients.
    """
    lora = LoRALinear(16, 16, r=4)
    x = torch.randn(2, 16)
    out = lora(x)
    loss = out.sum()
    loss.backward()

    # Base weights must have NO gradients
    assert lora.base_layer.weight.grad is None

    # LoRA parameters must have valid gradients
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_weight_merging_equivalence():
    """
    Merging weights W_merged = W_0 + (alpha/r)*B@A must yield identical outputs to unmerged forward pass.
    """
    torch.manual_seed(42)
    lora = LoRALinear(24, 24, r=4, lora_alpha=8.0)
    # Manually populate B with non-zero values
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)

    x = torch.randn(3, 24)

    # 1. Unmerged forward pass
    lora.eval()
    with torch.no_grad():
        unmerged_out = lora(x)

    # 2. Merge weights and re-evaluate
    lora.merge()
    assert lora.merged is True
    with torch.no_grad():
        merged_out = lora(x)

    diff = (unmerged_out - merged_out).abs().max().item()
    assert diff < 1e-5, f"Merged output deviates from unmerged forward pass: diff={diff}"

    # 3. Unmerge and verify consistency
    lora.unmerge()
    assert lora.merged is False
    with torch.no_grad():
        restored_out = lora(x)
    diff_restored = (unmerged_out - restored_out).abs().max().item()
    assert diff_restored < 1e-5


def test_injector_and_parameter_reduction():
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(128, 128)
            self.v_proj = nn.Linear(128, 128)
            self.out_proj = nn.Linear(128, 128)

        def forward(self, x):
            return self.out_proj(self.q_proj(x) + self.v_proj(x))

    model = DummyTransformer()
    apply_lora_to_model(model, target_names=["q_proj", "v_proj"], r=4)

    report = get_trainable_parameters_report(model)
    # Trainable parameters should be a small fraction of total
    assert report["trainable_percent"] < 15.0
    assert report["trainable_params"] > 0
