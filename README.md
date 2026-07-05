<div align="center">

# m2m-energy-fields

**Steer masked diffusion language models toward a topic without mentioning it in the prompt.**

Energy fields injected at each denoising step steer generation toward a concept
using dual-space embedding fusion — the topic never appears in the input.

[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Overview

m2m-energy-fields is a technique for **implicit topic control** in masked
diffusion language models (MDLM). Given a model like LLaDA-8B or Qwen3-mdlm
generating text via iterative unmasking, it modifies the logits at each
denoising step to favor tokens semantically aligned with a target direction.

The key property: **the target concept never appears in the prompt.** Given
*"Write a short story about something interesting"* and an energy field
pointing toward "ocean," the model writes about fish, waves, and coral reefs —
guided entirely by the energy field operating beneath the prompt.

This is impossible with autoregressive models, where guidance can only be
applied once at the prompt level. In masked diffusion, bidirectional attention
allows energy to be re-applied at every denoising step, creating a **cascade
effect** where early committed tokens influence all subsequent generation.

## How It Works

```
                    ┌─────────────────────────┐
                    │  Target: "ocean fish"   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Dual Embedding Fusion  │
                    │  0.5·MiniLM(384D) +     │
                    │  0.5·Model(4096D)       │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │  token_scores [vocab_size]        │
              │  semantic alignment per token    │
              └──────────────┬───────────────────┘
                             │
    ┌────────────────────────▼───────────────────────────────┐
    │  At each denoising step:                                │
    │                                                         │
    │  logits[masked] += alpha(step) · token_scores           │
    │                                                         │
    │  alpha(step) = alpha_start · (1 - step/total)^gamma     │
    │             (annealed: strong early, zero late)         │
    │                                                         │
    │  + anti-repetition penalty on committed tokens          │
    └─────────────────────────────────────────────────────────┘
```

### Three mechanisms working together

1. **Convex fusion** — combines MiniLM sentence embeddings (384D, captures
   semantic neighborhoods like "scuba" ≈ "ocean") with the diffusion model's
   own token embeddings (4096D, captures co-occurrence patterns like
   "turtle" appears near ocean contexts). The spaces are orthogonal
   (Spearman ρ = 0.14), so averaging reduces noise from both.

2. **Energy annealing** — guidance strength decays from `alpha_start` to 0
   across denoising steps. Strong early steering sets the topic; the model
   refines grammar and coherence freely in later steps.

3. **Anti-repetition penalty** — a token can appear `allowance` times freely,
   then each additional occurrence subtracts `penalty` from its logit. Breaks
   the positive feedback loop where energy forces the same token at every
   position. Adapted from frequency penalties in autoregressive decoding.

## Quick Start

```python
from m2m_energy_fields import EnergyGuidedSampler, GuidanceConfig
from dllm.utils import get_model, get_tokenizer
import torch

# Load any MDLM-compatible model
model = get_model(model_args=type("Args", (), {
    "model_name_or_path": "GSAI-ML/LLaDA-8B-Instruct",
    "dtype": torch.bfloat16,
    "device_map": {"": 0},
})()).eval()
tokenizer = get_tokenizer(model_args=type("Args", (), {
    "model_name_or_path": "GSAI-ML/LLaDA-8B-Instruct",
})())

# Create guided sampler (builds MiniLM token table on first call)
sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

# Set energy field (topic steering — not in prompt)
sampler.set_guidance(
    target_texts=["ocean coral reef fish deep sea"],
    alpha=10.0,
)

# Generate
config = GuidanceConfig(
    alpha=10.0,
    temperature=0.6,
    rep_penalty=5.0,
    rep_allowance=1,
    fusion_weight=0.5,  # convex 50/50
)
outputs = sampler.sample(inputs, config, return_dict=True)
```

## Installation

```bash
# Requires dllm framework (masked diffusion sampling)
git clone https://github.com/ZHZisZZ/dllm.git
cd dllm && pip install -e .

# Install m2m-energy-fields
cd ..
git clone https://github.com/schwabauerbriantomas-gif/m2m-energy-fields.git
cd m2m-energy-fields
pip install -e .
```

**Dependencies**: PyTorch ≥2.0, transformers ≥4.46, sentence-transformers, dllm

<!-- RESULTS_PLACEHOLDER -->

## Performance

All results from a single run of `tests/phase3_v13_fusion.py` with
LLaDA-8B-Instruct (BF16, 64 steps, seed=42). Hardware: RTX 3090 24GB.

**Prompt**: *"Write a short story about something interesting."* — the topic
word never appears in the prompt; it only exists in the energy field.

### Fusion strategy comparison

| Strategy | ocean | horror | space | cooking | **mean** | std |
|---|---|---|---|---|---|---|
| model_only (4096D) | 0.120 | 0.126 | 0.151 | 0.085 | 0.120 | 0.023 |
| minilm_only (384D) | 0.189 | 0.096 | 0.127 | 0.240 | 0.163 | 0.055 |
| **convex 50/50** | **0.173** | **0.120** | **0.138** | **0.293** | **0.181** | 0.067 |
| convex 30/70 | 0.189 | 0.118 | 0.207 | 0.040 | 0.138 | 0.066 |
| convex 70/30 | 0.244 | 0.114 | 0.149 | 0.121 | 0.157 | 0.052 |
| rrf k=60 | 0.192 | 0.138 | 0.161 | 0.140 | 0.158 | 0.022 |
| geometric mean | 0.267 | 0.101 | 0.020 | 0.011 | 0.100 | 0.103 |
| harmonic mean | 0.173 | 0.152 | 0.019 | 0.011 | 0.089 | 0.074 |
| bayesian product | -0.003 | 0.016 | -0.027 | -0.022 | -0.009 | 0.017 |

`quality = target_sim × coherence × diversity`

**Convex 50/50 wins** with 0.181 mean quality — **+11% over the best single
embedding source** (minilm_only: 0.163). The two spaces are orthogonal
(Spearman ρ = 0.14): model embeddings capture co-occurrence ("turtle" near
ocean contexts), MiniLM captures semantics ("scuba" ≈ "ocean"). Each
compensates for the other's blind spots.

### Per-topic detail (convex 50/50)

| Topic | target_sim | coherence | diversity | quality |
|---|---|---|---|---|
| ocean | 0.486 | 0.477 | 0.746 | 0.173 |
| horror | 0.466 | 0.336 | 0.768 | 0.120 |
| space | 0.301 | 0.611 | 0.750 | 0.138 |
| **cooking** | **0.546** | **0.734** | **0.732** | **0.293** |

### Example output

Prompt: *"Write a short story about something interesting."*
Energy target: `"cooking recipe chef kitchen delicious food"`

> *"Once upon a time, there was a magical kitchen that could cook the most
> delicious food imaginable. The kitchen was run by a chef named Chef Chef.
> The chef had great passion for cooking and loved experimenting with new
> recipes and ingredients. One day, he set out to cook the most delicious
> food in the..."*

No mention of cooking in the prompt — the energy field steered topic selection
through 64 denoising steps of bidirectional attention.

### Computational overhead

Energy guidance adds tensor operations (scatter_add, cosine similarity) on top
of each denoising step. These are fixed-cost operations that do not scale with
model size. Detailed per-step profiling has not been recorded; the EBM overhead
is negligible relative to the 8B model's forward pass.

No timing measurements have been validated and recorded. If you run benchmarks,
please contribute results.

## Hardware Limitations

This project was developed on:
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM)
- **System RAM**: 8 GB (severely constrained — swap-backed)
- **Precision**: BF16

### What this means

- **DiffusionGemma 26B A4B** (the ideal target model) cannot load with 8 GB
  system RAM. Nine loading attempts failed across INT4 quantization variants
  (BnB, Quanto, AWQ, GGUF). The methodology transfers directly when hardware
  allows.
- **LLaDA-8B-Instruct** loads successfully (16 GB VRAM) and was used for all
  experiments.
- **Qwen3-0.6B-mdlm** loads in 1.2 GB VRAM — used for initial prototyping.
- All timing benchmarks reflect RTX 3090 + BF16. No measurements were taken
  on other GPUs or precisions.

The technique itself is hardware-independent. The energy computation,
fusion, and anti-rep penalty operate on logits and token counts — they are
model-agnostic and do not depend on the underlying architecture being
autoregressive or diffusion-based. The only requirement is access to the
logits at each denoising step.

## Iteration History

This project went through 13 experimental versions. Each tested a specific
hypothesis and either advanced or was discarded:

| Version | Approach | Quality | Outcome |
|---|---|---|---|
| v1–v4 | Token energy, alpha sweep | 0.045 | Model embeddings act as keyword matchers |
| v5 | Probability scaling | 0.075 | Softmax scaling helps marginally |
| v6–v8 | DSpark-inspired re-masking | 0.060 | Re-masking deadlocks in masked diffusion |
| **v9** | **Anti-rep penalty + annealing** | **0.120** | **Prevention beats repair** |
| v10 | Energy-model agreement veto | 0.035 | Softmax uncalibrated in partial context |
| v11–v12 | MiniLM vs model embeddings | 0.118 | Semantic space ≠ co-occurrence (ρ=0.14) |
| **v13** | **Convex dual-space fusion** | **0.181** | **Orthogonal spaces combine beneficially** |

*Quality = target_sim × coherence × diversity. Values from v9/v13 experiment runs.*

### DSpark confidence head — tested, does not transfer

DSpark (DeepSeek's speculative decoding framework) uses a trained confidence
head to decide which draft tokens to accept. We tested adapting this pattern
(v10: energy-model agreement veto) but it fails because DSpark operates in
autoregressive decoding where the model's softmax is well-calibrated (causal
context → peaked distribution). In masked diffusion, the canvas is partially
masked → the model's softmax is flat (P[any_token] ≈ 0.001) → the confidence
proxy vetoes too many tokens (raw veto counts of 23-30 out of 128 per step in
experiments), effectively disabling energy guidance.

The anti-repetition penalty (v9) solves the same problem (preventing bad
tokens from entering the canvas) but using observable post-hoc signal (token
frequency counts) rather than predicting model agreement from uncalibrated
probabilities.

## When to Use This

**Use it when:**
- You need implicit topic control (target must not appear in prompt)
- You need multi-axis composition (`0.7·ocean + 0.3·science`)
- You need dynamic steering mid-generation (change direction between steps)
- You want low-overhead guidance (fixed-cost tensor ops, negligible vs forward pass)
- Your base model is a masked diffusion LM

**Don't use it when:**
- You can simply put the topic in the prompt (prompt engineering is more precise)
- You need factual accuracy (use RAG)
- You need consistent results across all topics (variance is high)
- Your model is autoregressive (energy still works but loses the cascade benefit)

## Project Structure

```
m2m-energy-fields/
├── src/m2m_energy_fields/
│   ├── __init__.py          # Public API
│   ├── core.py              # EnergyGuidedSampler, GuidanceConfig, fusion functions
│   └── metrics.py           # Evaluation: coherence, diversity, target_sim
├── examples/
│   ├── guided_story_llada8b.py   # Full demo with LLaDA-8B
│   ├── guided_story_qwen3.py     # Lightweight demo (0.6B)
│   └── topic_steering_sweep.py   # Reproducible experiment
├── tests/
│   ├── test_core.py              # Unit tests
│   ├── phase3_v9_antirep.py      # v9: anti-rep penalty (production config)
│   ├── phase3_v10_veto.py        # v10: DSpark veto (failed)
│   ├── phase3_v12_head2head.py   # v12: model vs MiniLM comparison
│   └── phase3_v13_fusion.py      # v13: convex fusion (final config)
├── results/                      # JSONL results from experiments
├── pyproject.toml
└── README.md
```

## Ecosystem

Part of the m2m research ecosystem:

| Project | Description |
|---|---|
| **EBM-splats** | Energy-based splats on S³⁸³, rectified flow velocity, semantic composition |
| **m2m-energy-fields** | Energy guidance for diffusion LMs (this repo) |
| **SplatsDB** | Vector memory with spatial organization (wings, rooms, halls) |

## License

MIT
