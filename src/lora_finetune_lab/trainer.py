"""
Training loop with gradient accumulation, evaluation, and adapter state checkpointing.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .lora_layer import LoRALinear


class LoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(dataloader, start=1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets) / self.grad_accum_steps
            loss.backward()

            total_loss += loss.item() * self.grad_accum_steps

            if step % self.grad_accum_steps == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

        return total_loss / len(dataloader)

    def save_adapter(self, save_path: str):
        """Save only the trainable LoRA adapter weights (tiny footprint, typically <10MB)."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear) and module.r > 0:
                lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
                lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        torch.save(lora_state, save_path)
        print(f"[*] Saved {len(lora_state)} LoRA tensors to {save_path}")

    def load_adapter(self, load_path: str):
        """Load LoRA adapter weights into model."""
        lora_state = torch.load(load_path, map_location=self.device)
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear) and module.r > 0:
                a_key = f"{name}.lora_A"
                b_key = f"{name}.lora_B"
                if a_key in lora_state and b_key in lora_state:
                    module.lora_A.data.copy_(lora_state[a_key])
                    module.lora_B.data.copy_(lora_state[b_key])
        print(f"[*] Successfully loaded adapter from {load_path}")
