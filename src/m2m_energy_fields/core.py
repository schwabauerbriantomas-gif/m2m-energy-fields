"""
Core energy guidance module.

The EnergyGuidedSampler wraps any MDLM-compatible model and injects energy
fields at each denoising step, using three mechanisms:

1. Convex fusion of two orthogonal embedding spaces (MiniLM 384D semantic +
   model 4096D co-occurrence) to compute per-token energy scores.
2. Energy annealing (alpha decays from alpha_start to alpha_end over steps).
3. Anti-repetition penalty (frequency_penalty style) to prevent collapse.

ENERGY FIELD MECHANISM:
  A masked diffusion model (LLaDA, Qwen3-mdlm) generates text by iteratively
  unmasking tokens. At each step, the model produces logits [B, T, vocab] for
  ALL positions simultaneously via bidirectional attention.

  We compute an energy direction d from target/suppress texts in two spaces,
  fuse them into a single token_scores vector, then at each denoising step:

      logits[masked_positions] += alpha(step) * token_scores
      logits[:, :, token] -= penalty(step) * max(0, count[token] - allowance)

  Because the model uses bidirectional attention, tokens committed early
  influence subsequent steps — creating a cascade effect that compounds the
  energy signal.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


try:
    import dllm
    from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
    from dllm.core.samplers.utils import get_num_transfer_tokens, add_gumbel_noise
    from dllm.core.schedulers import LinearAlphaScheduler
except ImportError:
    raise ImportError(
        "dllm framework is required.\n"
        "Install: git clone https://github.com/ZHZisZZ/dllm.git && cd dllm && pip install -e ."
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. Install: pip install sentence-transformers"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class GuidanceConfig:
    """Configuration for energy-guided generation."""
    alpha: float = 10.0
    """Initial guidance strength. Decays to 0 over steps via annealing.
    Sweet spot: 10.0 (strong early steering, decays to let model refine)."""

    alpha_end: float = 0.0
    """Final alpha value after annealing. Typically 0."""

    gamma: float = 1.0
    """Annealing decay rate. gamma=1 → linear, gamma=2 → quadratic."""

    temperature: float = 0.6
    """Gumbel noise temperature. Must be >0 for energy to have effect."""

    steps: int = 64
    """Total denoising steps."""

    max_new_tokens: int = 64
    """Tokens to generate."""

    block_size: int = 32
    """Tokens per denoising block."""

    rep_penalty: float = 5.0
    """Anti-repetition penalty strength. Subtracted from logits per excess
    occurrence. 5.0 is the validated sweet spot."""

    rep_allowance: int = 1
    """How many times a token can appear before penalty kicks in."""

    fusion_weight: float = 0.5
    """Convex fusion weight: w*model + (1-w)*minilm. 0.5 is optimal (v13)."""


@dataclass
class EnergyField:
    """An energy field defined by target and suppress directions."""
    target_texts: list[str] = field(default_factory=list)
    suppress_texts: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return bool(self.target_texts or self.suppress_texts)


# ── Fusion strategies ──

def fuse_convex(model_scores: torch.Tensor, minilm_scores: torch.Tensor, w: float = 0.5) -> torch.Tensor:
    """Convex combination: w*model + (1-w)*minilm."""
    return w * model_scores + (1 - w) * minilm_scores


def fuse_rrf(model_scores: torch.Tensor, minilm_scores: torch.Tensor, k: int = 60) -> torch.Tensor:
    """Reciprocal Rank Fusion."""
    rank_m = model_scores.argsort(descending=True).argsort().float()
    rank_n = minilm_scores.argsort(descending=True).argsort().float()
    return 1.0 / (k + rank_m + 1) + 1.0 / (k + rank_n + 1)


# ── Score computation ──

def compute_model_scores(
    embed_matrix: torch.Tensor, tokenizer, target_text: str
) -> torch.Tensor:
    """Compute token scores using model's own embeddings (4096D, co-occurrence space)."""
    tokens = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=128)
    ids = tokens["input_ids"].to(embed_matrix.device)
    with torch.no_grad():
        d = F.normalize(embed_matrix[ids].mean(dim=1).squeeze(0), dim=-1)
        scores = torch.mv(embed_matrix, d)
        scores = scores / (scores.abs().max() + 1e-8)
    return scores


def compute_minilm_scores(
    minilm_table: torch.Tensor, embedder, target_text: str
) -> torch.Tensor:
    """Compute token scores using MiniLM embeddings (384D, semantic space)."""
    d = embedder.encode([target_text], convert_to_tensor=True,
                        normalize_embeddings=True, device=str(minilm_table.device))
    d = F.normalize(d.squeeze(0), dim=-1)
    with torch.no_grad():
        scores = torch.mv(minilm_table, d)
        scores = scores / (scores.abs().max() + 1e-8)
    return scores


class EnergyGuidedSampler:
    """
    MDLM sampler with dual-space energy field injection at each denoising step.

    Implements v13 architecture: convex fusion of MiniLM (384D) + model (4096D)
    embeddings, energy annealing, and anti-repetition penalty.

    Usage:
        sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)
        sampler.set_guidance(
            target_texts=["ocean coral reef fish"],
            alpha=10.0,
        )
        config = GuidanceConfig()
        outputs = sampler.sample_guided(inputs, config)
    """

    def __init__(self, model, tokenizer, minilm_name: str = MINILM_MODEL):
        self.model = model
        self.tokenizer = tokenizer
        self._field: Optional[EnergyField] = None
        self._token_scores: Optional[torch.Tensor] = None

        # Model's embedding matrix: [vocab, hidden_dim]
        self.embed_matrix = model.get_input_embeddings().weight.data.float().to(DEVICE)
        self.vocab_size = self.embed_matrix.shape[0]

        # MiniLM embedder + precomputed token table
        self.embedder = SentenceTransformer(minilm_name, device=DEVICE)
        self._minilm_table: Optional[torch.Tensor] = None

    def _build_minilm_table(self) -> torch.Tensor:
        """Build MiniLM embedding table for all vocab tokens. Cached after first call."""
        if self._minilm_table is not None:
            return self._minilm_table

        token_texts = [
            self.tokenizer.decode([i], skip_special_tokens=True).strip() or "<pad>"
            for i in range(self.vocab_size)
        ]
        self._minilm_table = self.embedder.encode(
            token_texts, batch_size=1024, show_progress_bar=False,
            convert_to_tensor=True, normalize_embeddings=True, device=DEVICE,
        )
        return self._minilm_table

    def set_guidance(
        self,
        target_texts: list[str] | None = None,
        suppress_texts: list[str] | None = None,
        alpha: float = 10.0,
        fusion_weight: float = 0.5,
    ) -> None:
        """
        Compute and activate energy guidance.

        Args:
            target_texts: Texts defining the target direction (topic to steer toward).
            suppress_texts: Texts defining the suppress direction (topic to steer away).
            alpha: Initial guidance strength.
            fusion_weight: Convex fusion weight (0=all-MiniLM, 1=all-model, 0.5=balanced).
        """
        field = EnergyField(
            target_texts=target_texts or [],
            suppress_texts=suppress_texts or [],
        )
        if not field.is_active():
            self._field = None
            self._token_scores = None
            return

        # ── Compute target direction text ──
        target_text = " ".join(field.target_texts)
        suppress_text = " ".join(field.suppress_texts) if field.suppress_texts else None

        # ── Model embedding scores ──
        model_scores = compute_model_scores(self.embed_matrix, self.tokenizer, target_text)
        if suppress_text:
            sup_scores = compute_model_scores(self.embed_matrix, self.tokenizer, suppress_text)
            model_scores = model_scores - sup_scores * 0.5

        # ── MiniLM scores ──
        minilm_table = self._build_minilm_table()
        minilm_scores = compute_minilm_scores(minilm_table, self.embedder, target_text)
        if suppress_text:
            sup_minilm = compute_minilm_scores(minilm_table, self.embedder, suppress_text)
            minilm_scores = minilm_scores - sup_minilm * 0.5

        # ── Convex fusion ──
        fused = fuse_convex(model_scores, minilm_scores, w=fusion_weight)
        fused = fused / (fused.abs().max() + 1e-8)
        self._token_scores = fused.to(DEVICE)
        self._field = field

    def clear_guidance(self) -> None:
        """Disable energy guidance."""
        self._field = None
        self._token_scores = None

    @property
    def is_guided(self) -> bool:
        return self._token_scores is not None

    def top_guided_tokens(self, k: int = 10) -> list[str]:
        """Show the top-k tokens that energy guidance favors."""
        if self._token_scores is None:
            return []
        idx = self._token_scores.topk(k).indices.tolist()
        return [self.tokenizer.decode([i]).strip() for i in idx]

    # ── Guided sampling ──

    def sample_guided(
        self,
        inputs,
        config: GuidanceConfig | None = None,
    ) -> torch.Tensor:
        """
        Run energy-guided masked diffusion sampling.

        This is the core algorithm (v9/v13 validated):
        - Energy annealing: alpha decays from config.alpha to alpha_end
        - Anti-rep penalty: ramps up over generation, subtracts from logits
          for tokens that exceed rep_allowance occurrences.
        - Token selection via Gumbel noise + confidence-based transfer.

        Args:
            inputs: List of token id lists or tensors, one per sequence.
            config: GuidanceConfig with alpha, penalty, steps, etc.

        Returns:
            Tensor of shape [B, T] with committed token ids.
        """
        config = config or GuidanceConfig()
        assert self._token_scores is not None, "Call set_guidance() first"

        token_scores = self._token_scores
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ── Build canvas ──
        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=DEVICE) for p in inputs]
        prompt_lens = [p.shape[0] for p in inputs]
        max_length = config.max_new_tokens + max(prompt_lens)
        B, T = len(inputs), max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=DEVICE)
        for i, p in enumerate(inputs):
            x[i, :prompt_lens[i]] = p
            x[i, prompt_lens[i]:prompt_lens[i] + config.max_new_tokens] = mask_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
        for i, pl in enumerate(prompt_lens):
            attention_mask[i, :min(pl + config.max_new_tokens, T)] = 1

        # ── Scheduling ──
        scheduler = LinearAlphaScheduler()
        num_blocks = math.ceil(config.max_new_tokens / config.block_size)
        steps_per_block = max(1, math.ceil(config.steps / num_blocks))
        total_steps = num_blocks * steps_per_block
        token_counts = torch.zeros(B, self.vocab_size, device=DEVICE)

        def alpha_at(step):
            progress = step / max(total_steps - 1, 1)
            decay = max(1.0 - progress, 0.0) ** config.gamma
            return config.alpha * decay + config.alpha_end * (1.0 - decay)

        def penalty_at(step):
            # Linear ramp from 0 to rep_penalty over generation
            progress = step / max(total_steps - 1, 1)
            return config.rep_penalty * progress

        # ── Denoising loop ──
        global_step = 0
        for b in range(num_blocks):
            block_mask = torch.zeros((B, config.block_size), dtype=torch.bool, device=x.device)
            for j in range(B):
                s = prompt_lens[j] + b * config.block_size
                e = min(s + config.block_size, prompt_lens[j] + config.max_new_tokens, T)
                if s < e:
                    block_mask[j, :e - s] = x[j, s:e] == mask_id

            num_transfer = get_num_transfer_tokens(
                mask_index=block_mask, steps=steps_per_block,
                scheduler=scheduler, stochastic=False,
            )
            eff_steps = num_transfer.size(1)

            for i in range(eff_steps):
                mask_index = x == mask_id
                a = alpha_at(global_step)
                pen = penalty_at(global_step)

                with torch.no_grad():
                    logits = self.model(x, attention_mask=attention_mask).logits.float()

                # Energy guidance at masked positions
                energy_scores = token_scores.unsqueeze(0).unsqueeze(0) * a
                logits = logits + mask_index.unsqueeze(-1).float() * energy_scores

                # Anti-repetition penalty (ramps up over generation)
                if pen > 0 and global_step > 2:
                    excess = (token_counts - config.rep_allowance).clamp(min=0)
                    logits = logits - (pen * excess).unsqueeze(1)

                # Token selection
                logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                p = F.softmax(logits, dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

                # Mask out positions beyond current block
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * config.block_size:] = -math.inf

                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.tensor(-math.inf, device=x0.device))

                # Confidence-based token transfer
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer[j, i].item())
                    if k > 0:
                        _, sel = torch.topk(conf[j], k=k)
                        transfer_index[j, sel] = True

                # Update frequency counts
                for j in range(B):
                    committed = x0[j][transfer_index[j]]
                    token_counts[j].scatter_add_(
                        0, committed,
                        torch.ones_like(committed, dtype=token_counts.dtype),
                    )

                x[transfer_index] = x0[transfer_index]
                global_step += 1

        return x

    def sample(
        self,
        inputs,
        config: GuidanceConfig | None = None,
        return_dict: bool = False,
    ):
        """
        Unified sample: if guidance is set, runs guided sampling; otherwise
        falls back to standard MDLM sampling via dllm.
        """
        config = config or GuidanceConfig()

        if self.is_guided:
            sequences = self.sample_guided(inputs, config)
            if return_dict:
                from types import SimpleNamespace
                return SimpleNamespace(sequences=sequences)
            return sequences

        # Standard MDLM (no guidance)
        mdlm_config = MDLMSamplerConfig(
            steps=config.steps,
            max_new_tokens=config.max_new_tokens,
            block_size=config.block_size,
            temperature=config.temperature,
        )
        sampler = MDLMSampler(model=self.model, tokenizer=self.tokenizer)
        return sampler.sample(inputs, mdlm_config, return_dict=return_dict)
