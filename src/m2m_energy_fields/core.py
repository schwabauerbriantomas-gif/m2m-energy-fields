"""
Core energy guidance module.

The EnergyGuidedSampler wraps any MDLM-compatible sampler from the dllm
framework and injects energy fields at each denoising step.

ENERGY FIELD MECHANISM:
  A masked diffusion model (LLaDA, Qwen3-mdlm) generates text by iteratively
  unmasking tokens. At each step, the model produces logits [B, T, vocab] for
  ALL positions simultaneously via bidirectional attention.

  We compute an energy direction d in the model's own embedding space from
  target/suppress texts. Then at each denoising step:

      logits[masked_positions] += alpha * dot(embed_matrix, d)

  This steers token selection toward energy-aligned tokens. Because the model
  uses bidirectional attention, tokens committed early influence subsequent
  steps — creating a cascade effect that compounds the energy signal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

try:
    from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
except ImportError:
    raise ImportError(
        "dllm framework is required. Install: pip install -e .[dllm]\n"
        "Or clone: https://github.com/ZHZisZZ/dllm"
    )


@dataclass
class GuidanceConfig:
    """Configuration for energy-guided generation."""
    alpha: float = 5.0
    """Guidance strength. Sweet spot is typically 3–7.
    Too low (<2): no effect. Too high (>10): text collapses to repetition."""

    temperature: float = 0.6
    """Sampling temperature. Must be >0 for energy to have effect."""

    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    remasking: str = "low_confidence"


@dataclass
class EnergyField:
    """An energy field defined by target and suppress directions."""
    target_texts: list[str] = field(default_factory=list)
    suppress_texts: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return bool(self.target_texts or self.suppress_texts)


class EnergyGuidedSampler(MDLMSampler):
    """
    MDLM sampler with EBM energy field injection at each denoising step.

    Usage:
        sampler = EnergyGuidedSampler(model, tokenizer)
        sampler.set_energy_field(
            target_texts=["ocean coral reef fish"],
            alpha=5.0,
        )
        outputs = sampler.sample(inputs, config, return_dict=True)

    The sampler monkey-patches model.forward to intercept logits and add
    energy scores before token selection. The original forward is restored
    after sampling.
    """

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self._field: Optional[EnergyField] = None
        self._alpha: float = 0.0
        self._token_scores: Optional[torch.Tensor] = None
        self._original_forward = None
        self._patched = False

        # Model's embedding matrix: [vocab, hidden_dim]
        self.embed_matrix = model.get_input_embeddings().weight.data.float()
        self.hidden_dim = self.embed_matrix.shape[1]
        self.vocab_size = self.embed_matrix.shape[0]

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts using the model's own embedding layer + mean pooling."""
        embs = []
        for text in texts:
            tokens = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            )
            ids = tokens["input_ids"].to(self.embed_matrix.device)
            with torch.no_grad():
                pooled = self.embed_matrix[ids].mean(dim=1).squeeze(0)
            embs.append(pooled)
        return torch.stack(embs)

    def set_energy_field(
        self,
        field: EnergyField,
        alpha: float = 5.0,
    ) -> None:
        """
        Activate energy guidance.

        Args:
            field: EnergyField with target/suppress text lists.
            alpha: Guidance strength (3–7 recommended).
        """
        if not field.is_active():
            self._field = None
            self._token_scores = None
            self.restore()
            return

        d = torch.zeros(self.hidden_dim, device=self.embed_matrix.device)
        if field.target_texts:
            d = d + self._embed_texts(field.target_texts).mean(dim=0)
        if field.suppress_texts:
            d = d - self._embed_texts(field.suppress_texts).mean(dim=0)
        d = F.normalize(d, dim=-1)

        with torch.no_grad():
            scores = torch.mv(self.embed_matrix, d)
            scores = scores / (scores.abs().max() + 1e-8)

        self._token_scores = scores.to(self.embed_matrix.device)
        self._alpha = alpha
        self._field = field

    def set_guidance(
        self,
        target_texts: list[str] | None = None,
        suppress_texts: list[str] | None = None,
        alpha: float = 5.0,
    ) -> None:
        """Convenience method — creates EnergyField internally."""
        field = EnergyField(
            target_texts=target_texts or [],
            suppress_texts=suppress_texts or [],
        )
        self.set_energy_field(field, alpha=alpha)

    def clear_guidance(self) -> None:
        """Disable energy guidance."""
        self._field = None
        self._token_scores = None
        self.restore()

    @property
    def is_guided(self) -> bool:
        return self._field is not None and self._field.is_active()

    def top_guided_tokens(self, k: int = 10) -> list[str]:
        """Show the top-k tokens that energy guidance favors."""
        if self._token_scores is None:
            return []
        idx = self._token_scores.topk(k).indices.tolist()
        return [self.tokenizer.decode([i]).strip() for i in idx]

    # ── Forward patching ──

    def _patched_forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Intercept logits and apply energy guidance at masked positions."""
        out = self._original_forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        if self.is_guided:
            mask_id = self.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_id)
            scores = self._token_scores.unsqueeze(0).unsqueeze(0) * self._alpha
            mask = mask_positions.unsqueeze(-1).float()
            out.logits = out.logits.float() + mask * scores
        return out

    def patch(self) -> None:
        """Patch the model's forward to inject energy guidance."""
        if not self._patched:
            self._original_forward = self.model.forward
            self.model.forward = self._patched_forward
            self._patched = True

    def restore(self) -> None:
        """Restore the original model forward."""
        if self._patched:
            self.model.forward = self._original_forward
            self._original_forward = None
            self._patched = False

    def __enter__(self):
        self.patch()
        return self

    def __exit__(self, *args):
        self.restore()
        return False

    def sample_guided(
        self,
        inputs,
        config: GuidanceConfig | None = None,
        target_texts: list[str] | None = None,
        suppress_texts: list[str] | None = None,
        alpha: float = 5.0,
    ):
        """
        Convenience: set guidance, sample, and restore in one call.

        Returns BaseSamplerOutput with .sequences and .histories.
        """
        config = config or GuidanceConfig()
        mdlm_config = MDLMSamplerConfig(
            steps=config.steps,
            max_new_tokens=config.max_new_tokens,
            block_size=config.block_size,
            temperature=config.temperature,
            remasking=config.remasking,
        )

        self.set_guidance(
            target_texts=target_texts,
            suppress_texts=suppress_texts,
            alpha=alpha,
        )

        with self:  # patches forward, restores on exit
            return self.sample(inputs, mdlm_config, return_dict=True)
