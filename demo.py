#!/usr/bin/env python3
"""
LoRA Fine-Tuning Demo:
Demonstrates parameter freezing, adapter training, parameter reduction metrics,
and zero-latency deployment via weight merging.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from lora_finetune_lab.injector import apply_lora_to_model, get_trainable_parameters_report
from lora_finetune_lab.trainer import LoRATrainer


class SmallEncoder(nn.Module):
    """Simple 4-layer model with multi-head linear projections."""
    def __init__(self, dim=256):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        h = torch.relu(self.q_proj(x) + self.v_proj(x))
        return self.fc(h)


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running LoRA Demo on device: {device}")

    # 1. Initialize base model
    model = SmallEncoder(dim=256)
    initial_report = get_trainable_parameters_report(model)
    print(f"[*] Base Model Parameters: {initial_report['all_params']:,} (100% trainable)")

    # 2. Inject LoRA adapters into attention projections
    print("\n[*] Injecting LoRA adapters (r=8, alpha=16) into ['q_proj', 'v_proj']...")
    apply_lora_to_model(model, target_names=["q_proj", "v_proj"], r=8, lora_alpha=16.0)

    lora_report = get_trainable_parameters_report(model)
    print("--------------------------------------------------")
    print(f"  Total Parameters:     {lora_report['all_params']:,}")
    print(f"  Trainable Parameters: {lora_report['trainable_params']:,}")
    print(f"  Frozen Parameters:    {lora_report['frozen_params']:,}")
    print(f"  Trainable Percentage: {lora_report['trainable_percent']:.2f}%")
    print(f"  Parameter Reduction:  {100 - lora_report['trainable_percent']:.2f}% frozen!")
    print("--------------------------------------------------")

    # 3. Create synthetic fine-tuning dataset
    x_train = torch.randn(200, 256)
    y_train = torch.randint(0, 10, (200,))
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 4. Train only LoRA adapters
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    trainer = LoRATrainer(model, optimizer, device=device)

    print("\n[*] Training LoRA adapters for 5 epochs...")
    for epoch in range(1, 6):
        loss = trainer.train_epoch(loader, criterion)
        print(f"    Epoch {epoch}/5 | Loss: {loss:.4f}")

    # 5. Save and reload adapter checkpoint
    adapter_path = "checkpoints/lora_adapter.pt"
    trainer.save_adapter(adapter_path)
    trainer.load_adapter(adapter_path)

    # 6. Benchmark Inference (Unmerged vs Merged)
    test_x = torch.randn(500, 256).to(device)
    model.eval()

    # Unmerged forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(test_x)
    unmerged_time = (time.perf_counter() - t0) * 1000

    # Merge weights into base layers
    for name, m in model.named_modules():
        if hasattr(m, "merge"):
            m.merge()

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(test_x)
    merged_time = (time.perf_counter() - t1) * 1000

    print("\n[*] Inference Benchmarking (50 batches):")
    print(f"    Unmerged Adapter Latency: {unmerged_time:.2f} ms")
    print(f"    Merged (Zero-Overhead) Latency: {merged_time:.2f} ms")
    print("[*] LoRA fine-tuning and weight merging demo complete.")


if __name__ == "__main__":
    main()
