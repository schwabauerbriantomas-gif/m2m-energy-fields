"""
m2m_energy_fields — Energy-based guidance for masked diffusion language models.

This package provides the EnergyGuidedSampler, which injects energy fields
derived from dual-space embedding fusion (MiniLM 384D + model 4096D) into the
denoising loop of masked diffusion language models (LLaDA, Qwen3-mdlm),
enabling compositional topic steering over text generation.

Validated architecture (v13):
- Convex fusion: 0.5 · MiniLM + 0.5 · Model embeddings
- Energy annealing: alpha decays from 10 → 0 over denoising steps
- Anti-repetition penalty: frequency_penalty adapted for masked diffusion
"""

from .core import (
    EnergyGuidedSampler,
    EnergyField,
    GuidanceConfig,
    compute_model_scores,
    compute_minilm_scores,
    fuse_convex,
    fuse_rrf,
    alpha_schedule,
    penalty_schedule,
)
from .metrics import evaluate_guidance, coherence_score, repetition_ratio

__version__ = "0.1.0"
__all__ = [
    "EnergyGuidedSampler",
    "EnergyField",
    "GuidanceConfig",
    "compute_model_scores",
    "compute_minilm_scores",
    "fuse_convex",
    "fuse_rrf",
    "alpha_schedule",
    "penalty_schedule",
    "evaluate_guidance",
    "coherence_score",
    "repetition_ratio",
]
