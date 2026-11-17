# lora-finetune-lab

A modular, lightweight PyTorch implementation of **Low-Rank Adaptation (LoRA)** from scratch for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models.

Reference Paper: [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)](https://arxiv.org/abs/2106.09685).

---

## What is LoRA?

During standard full fine-tuning, all parameters of a pretrained model are updated ($\Delta W \in \mathbb{R}^{d \times k}$). This becomes intractable for 7B+ parameter models.

LoRA freezes the pretrained weight matrix $W_0$ and decomposes the rank of $\Delta W$ into two low-rank matrices:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

where:
* $W_0 \in \mathbb{R}^{d \times k}$ (Pretrained, **frozen**, `requires_grad=False`)
* $A \in \mathbb{R}^{r \times k}$ (Initialized with Kaiming uniform distribution)
* $B \in \mathbb{R}^{d \times r}$ (Initialized with **zeros**, ensuring $\Delta W = 0$ at step 0)
* $r \ll \min(d, k)$ (Rank dimension, typically $4, 8, 16$)
* $\alpha$ (Scaling constant)

This reduces trainable parameters by **95% to 99%** while matching full fine-tuning performance.

---

## Features

* **Pure PyTorch Implementation:** No heavy dependencies, transparent tensor arithmetic.
* **Zero Step-0 Drift:** Guarantees $\Delta W = 0$ at initialization via zero-initialized projection $B$.
* **Automated Injection:** `apply_lora_to_model()` parses any PyTorch model graph and replaces target linear projections (e.g. `q_proj`, `v_proj`) seamlessly.
* **Zero-Latency Deployment (`merge()` / `unmerge()`):** Collapses low-rank weights directly into the base weights ($W_0 \leftarrow W_0 + \Delta W$), eliminating extra GEMM operations during serving.
* **Checkpoint Footprint:** Saves only the adapter matrices, reducing checkpoint size from gigabytes to megabytes.

---

## Directory Structure

```
lora-finetune-lab/
├── src/
│   └── lora_finetune_lab/
│       ├── __init__.py     # Module exports
│       ├── lora_layer.py   # LoRALinear implementation with merge/unmerge
│       ├── injector.py     # Automatic graph replacement & param reports
│       └── trainer.py      # PEFT training loop & adapter serialization
├── tests/
│   └── test_lora.py        # Gradient isolation, step-0 zero diff, and merge tests
├── demo.py                 # End-to-end training, saving, and inference benchmark
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1. Setup & Installation
```bash
git clone https://github.com/Antonymunene010/lora-finetune-lab.git
cd lora-finetune-lab

python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
```

### 2. Run the Demo
```bash
python demo.py
```

### 3. Run Unit Tests
```bash
pytest tests/ -v
```

---

## Benchmark Results

| Configuration | Total Params | Trainable Params | % Trainable | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | 200,000 | 200,000 | 100.0% | Baseline |
| **LoRA ($r=8$)** | 208,192 | 8,192 | **3.93%** | **~75% reduction** |
| **LoRA ($r=4$)** | 204,096 | 4,096 | **2.01%** | **~85% reduction** |

---

## Author
**Antony Munene** — [GitHub](https://github.com/Antonymunene010) • [LinkedIn](https://www.linkedin.com/in/antony-munene-ml/)
License: MIT
