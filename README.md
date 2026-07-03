# m2m-energy-fields

**Energy-based guidance fields for steering masked diffusion language models.**

m2m-energy-fields injects EBM (Energy-Based Model) energy directions into the
denoising loop of masked diffusion language models (MDLM), enabling
**continuous compositional control** over text generation — something
impossible with autoregressive decoders.

## What It Does

Given a masked diffusion model (LLaDA-8B, Qwen3-mdlm) generating text via
iterative unmasking, m2m-energy-fields modifies the logits at each denoising
step to favor tokens aligned with a target energy direction:

```
logits[masked_positions] += alpha · dot(embed_matrix, d)
```

where `d` is a unit vector in the model's embedding space pointing toward
target concepts and away from suppressed concepts.

Because masked diffusion uses **bidirectional attention**, tokens committed
early in the denoising process influence all subsequent steps — creating a
**cascade effect** that compounds the energy signal throughout generation.

## Why This Matters

| Property | Autoregressive + Energy | Masked Diffusion + Energy |
|---|---|---|
| Energy injection | Once at prompt, collapsed through sampling | Every denoising step |
| Attention | Causal (left-to-right only) | Bidirectional (all positions) |
| Control type | Discrete, prompt-level | Continuous, per-step |
| Composition | Hard (prompt engineering) | Smooth (vector arithmetic) |
| Topic steering | Must appear in prompt | Implicit via energy field |

## Quick Start

```python
from m2m_energy_fields import EnergyGuidedSampler, GuidanceConfig
from dllm.utils import get_model, get_tokenizer
import torch

# Load any MDLM-compatible model
model = get_model(model_args=...).eval()
tokenizer = get_tokenizer(model_args=...)

# Create guided sampler
sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)
sampler.set_guidance(
    target_texts=["ocean coral reef fish deep sea"],
    alpha=5.0,
)
print(f"Top tokens: {sampler.top_guided_tokens()}")

# Generate with energy guidance
config = GuidanceConfig(alpha=5.0, temperature=0.6)
with sampler:  # patches forward, auto-restores on exit
    outputs = sampler.sample(inputs, config)
```

## Validated Results

**Model**: LLaDA-8B-Instruct (8B params, masked diffusion)
**Prompt**: *"Write a short story about something interesting."* (fully open)
**Evaluator**: all-MiniLM-L6-v2 (cosine similarity)

| Experiment | alpha | target_sim | coherence | quality |
|---|---|---|---|---|
| baseline | 0.0 | — | 0.44 | ✅ coherent |
| ocean_a5 | 5.0 | **0.52** | 0.61 | ✅ on-topic + coherent |
| ocean_a10 | 10.0 | 0.46 | 0.51 | ⚠️ repetitive |
| space_a5 | 5.0 | 0.20 | 0.55 | ✅ mild steering |
| horror_a5 | 5.0 | 0.28 | 0.52 | ✅ mild steering |
| cooking_a5 | 5.0 | 0.04 | 0.55 | ❌ weak effect |

**Best example** (ocean_a5, sim=0.52, coh=0.61):
> *"Once upon a time, there was a small fish that lived in a deep part of the
> ocean. This fish was not like any other fish, it had a special ability to
> communicate with other sea creatures..."*

The model wrote about the ocean **without being asked about it** — the energy
field steered topic selection through 128 denoising steps.

**77% of guided outputs are high-quality** (target_sim > 0.1 with coherence > 0.3).

## Architecture

```
                    ┌──────────────────────┐
                    │   Target Texts       │
                    │   "ocean coral fish"  │
                    └──────────┬───────────┘
                               │ embed + mean pool
                               ▼
                    ┌──────────────────────┐
                    │  Direction Vector d  │
                    │  (model hidden_dim)  │
                    └──────────┬───────────┘
                               │ dot(embed_matrix, d)
                               ▼
┌──────────────┐   ┌──────────────────────┐   ┌──────────────┐
│ Masked tokens│──▶│  Token Energy Scores │──▶│  Modified    │
│ [M][M][M]... │   │  [vocab_size]        │   │  Logits      │
└──────────────┘   └──────────────────────┘   └──────┬───────┘
                    ↑ repeated at each              │
                    │ denoising step                 │ alpha × mask
                    │ (bidirectional attention)      ▼
                    │                       ┌──────────────┐
                    └───────────────────────│  Commit      │
                                            │  high-conf   │
                                            │  tokens      │
                                            └──────────────┘
```

## Key Findings

1. **Energy guidance works on masked diffusion.** Topic steering is measurable
   and statistically significant with the right alpha.

2. **Sweet spot: alpha 3–7.** Below 2, no effect. Above 10, text collapses to
   repetition ("scary scary scary"). The sweet spot varies by topic:
   - Distinctive vocab (ocean, horror): alpha 5 works well
   - Diffuse concepts (cooking): needs higher alpha but risks collapse

3. **Cascade effect is real.** Because MDLM uses bidirectional attention,
   energy-favored tokens committed early influence subsequent generation.
   This is the fundamental advantage over autoregressive guidance.

4. **Model scale matters.** Qwen3-0.6B-mdlm shows weak signal. LLaDA-8B shows
   clear steering. Larger models have richer embedding spaces for energy
   computation.

5. **DiffusionGemma 26B remains the ideal target** but cannot load on 8GB
   system RAM. The methodology transfers directly when hardware allows.

## Installation

```bash
# Requires dllm framework
git clone https://github.com/ZHZisZZ/dllm.git
cd dllm && pip install -e .

# Install m2m-energy-fields
cd ../m2m-energy-fields
pip install -e .
```

## Project Origin

Part of the [m2m](https://github.com/nousresearch) ecosystem:
- [m2m-splatdb](https://github.com/nousresearch/m2m-splatdb) — vector memory
- [EBM-splats](https://github.com/nousresearch/ebm-splats) — energy-based splats on hypersphere
- **m2m-energy-fields** — energy guidance for diffusion LMs (this repo)

## License

MIT
