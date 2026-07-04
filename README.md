# m2m-energy-fields

<div align="center">

**Energy-based guidance fields for steering masked diffusion language models.**

Inject semantic energy directions into each denoising step of discrete text
diffusion — steering generation toward a concept without mentioning it in the prompt.

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

## Performance

### Quality vs. baseline diffusion (no guidance)

**Model**: LLaDA-8B-Instruct · **Prompt**: *"Write a short story about something interesting."*
**Hardware**: RTX 3090 24GB BF16

| Config | target_sim | coherence | diversity | quality |
|---|---|---|---|---|
| Baseline (no guidance) | — | 0.55 | 0.78 | — |
| Ocean + model embeddings | 0.61 | 0.86 | 0.22 | 0.120 |
| Ocean + MiniLM embeddings | 0.46 | 0.55 | 0.75 | 0.189 |
| **Ocean + convex fusion** | 0.55 | 0.48 | 0.75 | **0.173** |
| Cooking + convex fusion | 0.55 | 0.73 | 0.73 | **0.293** |
| Space + convex fusion | 0.30 | 0.61 | 0.75 | 0.138 |
| Horror + convex fusion | 0.47 | 0.34 | 0.77 | 0.120 |

`quality = target_sim × coherence × diversity`

### Convex fusion vs. single embedding sources

| Topic | Model only | MiniLM only | **Convex 50/50** |
|---|---|---|---|
| ocean | 0.120 | 0.189 | 0.173 |
| horror | 0.126 | 0.096 | 0.120 |
| space | 0.151 | 0.127 | 0.138 |
| cooking | 0.085 | 0.240 | **0.293** |
| **mean** | **0.120** | **0.163** | **0.181** |

Convex fusion beats the best single source by **+11%** on average quality.

### Computational overhead

| Component | Per-step cost | % of step |
|---|---|---|
| Model forward pass (8B BF16) | 85.0 ms | 82.4% |
| Softmax + argmax + topk | 16.2 ms | 15.7% |
| Energy guidance (EBM) | 1.4 ms | 1.4% |
| Anti-rep penalty | 0.7 ms | 0.7% |

**The EBM adds 2.1 ms per step — 2% overhead.** The bottleneck is entirely
the diffusion model's forward pass, not the energy computation.

### Throughput

All measurements are from a single RTX 3090 with LLaDA-8B-Instruct (BF16,
64 denoising steps, 200-token canvas):

- **Total generation time**: 6.6 s (102.9 ms × 64 steps)
- **Effective throughput**: ~30 TPS (tokens committed per second of wall time)
- **EBM overhead per step**: 2.1 ms (1.4 ms energy + 0.7 ms anti-rep penalty)
- **EBM as fraction of step**: 2.0%

The bottleneck is the model forward pass at 85 ms/step (82% of wall time).
Scaling to faster hardware or sparser models (MoE, FP8) would reduce the
forward pass proportionally — the EBM's 2.1 ms is fixed-cost tensor ops
(scatter_add, cosine similarity) that do not scale with model size.

No measurements were taken on hardware other than the RTX 3090.

### Example output

Prompt: *"Write a short story about something interesting."*
Energy target: `"cooking recipe chef kitchen delicious food"`

> *"Once upon a time, there was a magical kitchen that could cook the most
> delicious food imaginable. The kitchen was run by a chef named Chef Chef.
> The chef had great passion for cooking and loved experimenting with new
> recipes every day..."*

No mention of cooking in the prompt — the energy field steered topic selection
through 64 denoising steps of bidirectional attention.

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

# Create guided sampler
sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

# Set energy field (topic steering — not in prompt)
sampler.set_guidance(
    target_texts=["ocean coral reef fish deep sea"],
    alpha=10.0,
)

# Generate
config = GuidanceConfig(alpha=10.0, temperature=0.6)
with sampler:  # patches forward, auto-restores on exit
    outputs = sampler.sample(inputs, config)
```

## Installation

```bash
# Requires dllm framework (masked diffusion sampling)
git clone https://github.com/ZHZisZZ/dllm.git
cd dllm && pip install -e .

# Install m2m-energy-fields
cd ..
git clone https://github.com/nousresearch/m2m-energy-fields.git
cd m2m-energy-fields
pip install -e .
```

**Dependencies**: PyTorch ≥2.0, transformers ≥4.46, sentence-transformers, dllm

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

### DSpark confidence head — tested, does not transfer

DSpark (DeepSeek's speculative decoding framework) uses a trained confidence
head to decide which draft tokens to accept. We tested adapting this pattern
(v10: energy-model agreement veto) but it fails because DSpark operates in
autoregressive decoding where the model's softmax is well-calibrated (causal
context → peaked distribution). In masked diffusion, the canvas is partially
masked → the model's softmax is flat (P[any_token] ≈ 0.001) → the confidence
proxy vetoes 43% of all tokens, effectively disabling energy guidance.

The anti-repetition penalty (v9) solves the same problem (preventing bad
tokens from entering the canvas) but using observable post-hoc signal (token
frequency counts) rather than predicting model agreement from uncalibrated
probabilities.

## When to Use This

**Use it when:**
- You need implicit topic control (target must not appear in prompt)
- You need multi-axis composition (`0.7·ocean + 0.3·science`)
- You need dynamic steering mid-generation (change direction between steps)
- You want zero-overhead guidance (2% of forward pass)
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
│   ├── core.py              # EnergyGuidedSampler, EnergyField, GuidanceConfig
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
├── results/                      # JSONL results from all experiments
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
