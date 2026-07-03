"""
m2m_energy_fields — Energy-based guidance for masked diffusion language models.

This package provides the EnergyGuidedSampler, which injects energy fields
derived from EBM splat directions into the denoising loop of masked diffusion
language models (MDLM, LLaDA, DiffusionGemma), enabling continuous
compositional control over text generation.
"""

from .core import EnergyGuidedSampler, EnergyField, GuidanceConfig
from .metrics import evaluate_guidance, coherence_score, repetition_ratio

__version__ = "0.1.0"
__all__ = [
    "EnergyGuidedSampler",
    "EnergyField",
    "GuidanceConfig",
    "evaluate_guidance",
    "coherence_score",
    "repetition_ratio",
]
