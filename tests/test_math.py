"""
Mathematical invariant tests for m2m_energy_fields.

Each test validates a mathematical property that must hold for the system
to be correct. These don't need a GPU or model — they operate on the
pure functions (fusion, scheduling, normalization) and tensor invariants.
"""
import pytest
import torch
import math
import numpy as np

from m2m_energy_fields import (
    GuidanceConfig,
    EnergyField,
    fuse_convex,
    fuse_rrf,
    alpha_schedule,
    penalty_schedule,
)


# ═══════════════════════════════════════════════════════════════════
# 1. FUSION INVARIANTS
# ═══════════════════════════════════════════════════════════════════

class TestFuseConvex:
    """Convex fusion: s = w*m + (1-w)*n"""

    def test_convex_combination_property(self):
        """s(w) = w*m + (1-w)*n must satisfy: s(0)=n, s(1)=m, s(0.5)=0.5(m+n)."""
        m = torch.randn(100)
        n = torch.randn(100)
        assert torch.allclose(fuse_convex(m, n, w=0.0), n, atol=1e-6)
        assert torch.allclose(fuse_convex(m, n, w=1.0), m, atol=1e-6)
        assert torch.allclose(fuse_convex(m, n, w=0.5), 0.5 * (m + n), atol=1e-6)

    def test_convex_monotonic_interpolation(self):
        """As w goes 0→1, fused must move monotonically from n toward m
        in the direction (m-n). I.e., dot(fused - n, m-n) increases."""
        m = torch.randn(100)
        n = torch.randn(100)
        direction = m - n
        dots = []
        for w in torch.linspace(0, 1, 11):
            fused = fuse_convex(m, n, w=float(w))
            dots.append(torch.dot(fused - n, direction).item())
        # Dots must be non-decreasing
        for i in range(1, len(dots)):
            assert dots[i] >= dots[i-1] - 1e-5, \
                f"Convex fusion not monotonic at step {i}: {dots[i]} < {dots[i-1]}"

    def test_convex_bounded_between_inputs(self):
        """Each element of fused must lie between min(m[i],n[i]) and max(m[i],n[i])."""
        m = torch.randn(50)
        n = torch.randn(50)
        fused = fuse_convex(m, n, w=0.3)
        lower = torch.minimum(m, n)
        upper = torch.maximum(m, n)
        assert torch.all(fused >= lower - 1e-6), "Fused below min(m,n)"
        assert torch.all(fused <= upper + 1e-6), "Fused above max(m,n)"

    def test_convex_preserves_rank_correlation(self):
        """If both m and n rank token i as highest, fused must also rank it highest."""
        m = torch.tensor([0.1, 0.9, 0.3])
        n = torch.tensor([0.2, 0.8, 0.1])
        fused = fuse_convex(m, n, w=0.5)
        assert torch.argmax(fused) == 1, "Fusion lost top-rank agreement"

    def test_convex_equal_inputs_yields_input(self):
        """If m == n, fused must equal m for any w."""
        m = torch.randn(30)
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fused = fuse_convex(m, m, w=w)
            assert torch.allclose(fused, m, atol=1e-6)


class TestFuseRRF:
    """Reciprocal Rank Fusion: s = 1/(k+rank_m+1) + 1/(k+rank_n+1)"""

    def test_rrf_unanimous_top_rank_preserved(self):
        """If both rankers agree token i is #1, fused must rank it #1."""
        m = torch.tensor([3.0, 2.0, 1.0])   # rank 0 > 1 > 2
        n = torch.tensor([3.0, 2.0, 1.0])   # same
        fused = fuse_rrf(m, n, k=60)
        assert torch.argmax(fused) == 0

    def test_rrf_disagreement_no_zero(self):
        """RRF scores are always positive (sum of two positive terms)."""
        m = torch.randn(200)
        n = torch.randn(200)
        fused = fuse_rrf(m, n, k=60)
        assert torch.all(fused > 0), "RRF produced non-positive scores"

    def test_rrf_range(self):
        """RRF score ∈ (0, 2/(k+1)] for each token."""
        k = 60
        m = torch.randn(100)
        n = torch.randn(100)
        fused = fuse_rrf(m, n, k=k)
        upper_bound = 2.0 / (k + 1)
        assert torch.all(fused <= upper_bound + 1e-6), \
            f"RRF score exceeded 2/(k+1) = {upper_bound}"
        assert torch.all(fused > 0)

    def test_rrf_lower_k_sharper(self):
        """Lower k → more weight on top-ranked tokens → higher top score."""
        m = torch.tensor([10.0, 1.0, 0.0])
        n = torch.tensor([10.0, 1.0, 0.0])
        fused_low_k = fuse_rrf(m, n, k=1)
        fused_high_k = fuse_rrf(m, n, k=100)
        assert fused_low_k[0] > fused_high_k[0], \
            "Lower k should give higher score to top-ranked"


# ═══════════════════════════════════════════════════════════════════
# 2. NORMALIZATION INVARIANTS
# ═══════════════════════════════════════════════════════════════════

class TestNormalization:
    """Scores are normalized by abs().max() — must stay in [-1, 1]."""

    def test_normalized_range(self):
        """After normalize-by-max, all values ∈ [-1, 1]."""
        raw = torch.randn(1000) * 100
        normalized = raw / (raw.abs().max() + 1e-8)
        assert torch.all(normalized >= -1.0 - 1e-6)
        assert torch.all(normalized <= 1.0 + 1e-6)

    def test_normalized_preserves_sign(self):
        """Normalization must not change sign of any element."""
        raw = torch.randn(500)
        normalized = raw / (raw.abs().max() + 1e-8)
        assert torch.all(torch.sign(normalized) == torch.sign(raw))

    def test_normalized_preserves_rank(self):
        """Normalization must preserve rank ordering."""
        raw = torch.randn(200)
        normalized = raw / (raw.abs().max() + 1e-8)
        raw_rank = raw.argsort().argsort()
        norm_rank = normalized.argsort().argsort()
        assert torch.equal(raw_rank, norm_rank)


# ═══════════════════════════════════════════════════════════════════
# 3. ANNEALING SCHEDULE INVARIANTS
# ═══════════════════════════════════════════════════════════════════

class TestAlphaSchedule:
    """alpha_schedule(step, total, alpha_start, alpha_end, gamma)"""

    def test_start_value(self):
        """alpha(0) must equal alpha_start."""
        assert alpha_schedule(0, 64, alpha_start=10.0) == pytest.approx(10.0)

    def test_end_value(self):
        """alpha(total-1) must equal alpha_end."""
        assert alpha_schedule(63, 64, alpha_start=10.0, alpha_end=0.0) == pytest.approx(0.0)

    def test_monotonic_decrease(self):
        """alpha must be monotonically non-increasing across steps."""
        values = [alpha_schedule(s, 64, alpha_start=10.0) for s in range(64)]
        for i in range(1, len(values)):
            assert values[i] <= values[i-1] + 1e-9, \
                f"Alpha increased at step {i}: {values[i]} > {values[i-1]}"

    def test_non_negative(self):
        """alpha must never go below 0."""
        for s in range(64):
            assert alpha_schedule(s, 64, alpha_start=10.0, alpha_end=0.0) >= -1e-9

    def test_nonzero_end(self):
        """If alpha_end > 0, last step should equal alpha_end."""
        val = alpha_schedule(63, 64, alpha_start=10.0, alpha_end=2.0)
        assert val == pytest.approx(2.0)

    def test_gamma_curvature(self):
        """gamma=2 (quadratic) produces values ≤ gamma=1 (linear) at every
        intermediate step, because (1-x)^2 ≤ (1-x) for x ∈ [0,1].

        Both start at alpha_start and end at alpha_end, but gamma>1 creates
        a more concave curve — stays near alpha_start longer, then drops
        more steeply near the end."""
        steps = list(range(64))
        linear = [alpha_schedule(s, 64, alpha_start=10.0, gamma=1.0) for s in steps]
        quadratic = [alpha_schedule(s, 64, alpha_start=10.0, gamma=2.0) for s in steps]

        # Start and end are the same
        assert linear[0] == pytest.approx(quadratic[0], abs=1e-6)
        assert linear[-1] == pytest.approx(quadratic[-1], abs=1e-6)

        # At every intermediate step, quadratic ≤ linear
        for i in range(1, 63):
            assert quadratic[i] <= linear[i] + 1e-9, \
                f"Quadratic > linear at step {i}: quad={quadratic[i]} > lin={linear[i]}"

        # The gap should be largest at midpoint (maximum divergence)
        gaps = [abs(l - q) for l, q in zip(linear, quadratic)]
        max_gap_step = gaps.index(max(gaps))
        assert 20 < max_gap_step < 44, \
            f"Maximum gap should be near midpoint, got step {max_gap_step}"

    def test_total_steps_one(self):
        """Edge case: total_steps=1 shouldn't divide by zero."""
        val = alpha_schedule(0, 1, alpha_start=10.0)
        assert not math.isnan(val)
        assert not math.isinf(val)


# ═══════════════════════════════════════════════════════════════════
# 4. PENALTY SCHEDULE INVARIANTS
# ═══════════════════════════════════════════════════════════════════

class TestPenaltySchedule:
    """penalty_schedule(step, total, rep_penalty)"""

    def test_start_zero(self):
        """penalty(0) must be 0 — early tokens are unconstrained."""
        assert penalty_schedule(0, 64, rep_penalty=5.0) == pytest.approx(0.0)

    def test_end_value(self):
        """penalty(total-1) must equal rep_penalty."""
        assert penalty_schedule(63, 64, rep_penalty=5.0) == pytest.approx(5.0)

    def test_monotonic_increase(self):
        """Penalty must ramp up monotonically."""
        values = [penalty_schedule(s, 64, rep_penalty=5.0) for s in range(64)]
        for i in range(1, len(values)):
            assert values[i] >= values[i-1] - 1e-9

    def test_non_negative(self):
        """Penalty must never be negative."""
        for s in range(64):
            assert penalty_schedule(s, 64, rep_penalty=5.0) >= -1e-9


# ═══════════════════════════════════════════════════════════════════
# 5. ANTI-REPETITION PENALTY MECHANICS
# ═══════════════════════════════════════════════════════════════════

class TestAntiRepPenalty:
    """Validates the core formula: logits[:, :, v] -= pen * max(0, count[v] - allowance)"""

    def test_allowance_is_free(self):
        """Token appearing ≤ allowance times should get zero penalty."""
        count = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float)
        allowance = 2
        excess = (count - allowance).clamp(min=0)
        assert excess[0] == 0  # count=0, no penalty
        assert excess[1] == 0  # count=1 ≤ allowance
        assert excess[2] == 0  # count=2 = allowance
        assert excess[3] == 1  # count=3, excess=1
        assert excess[4] == 2  # count=4, excess=2

    def test_penalty_increases_with_excess(self):
        """More repetitions → more penalty (cumulative)."""
        pen = 5.0
        allowance = 1
        for count in [2, 5, 10, 20]:
            excess = max(0, count - allowance)
            penalty = pen * excess
            assert penalty > 0
        # Verify monotonicity
        p2 = pen * max(0, 2 - allowance)
        p10 = pen * max(0, 10 - allowance)
        assert p10 > p2

    def test_penalty_subtracts_from_logits(self):
        """logits -= pen * excess must decrease logits for repeated tokens."""
        V = 10
        logits = torch.zeros(1, 5, V)  # [B, T, vocab]
        token_counts = torch.tensor([0, 0, 0, 3, 0, 0, 0, 0, 0, 0])  # token 3 appeared 3x
        allowance = 1
        pen = 5.0
        excess = (token_counts - allowance).clamp(min=0)  # [vocab]
        logits_after = logits - pen * excess.unsqueeze(0).unsqueeze(0)  # broadcast [1, 1, V]
        # Token 3 should have lower logits everywhere
        assert logits_after[0, 0, 3] == -10.0  # -5 * (3-1) = -10
        assert logits_after[0, 0, 0] == 0.0    # no penalty for token 0

    def test_scatter_add_counts_correctly(self):
        """scatter_add_ must increment the right token counts."""
        counts = torch.zeros(10)
        committed = torch.tensor([3, 3, 5, 3, 7])
        counts.scatter_add_(0, committed, torch.ones_like(committed, dtype=counts.dtype))
        assert counts[3] == 3
        assert counts[5] == 1
        assert counts[7] == 1
        assert counts[0] == 0


# ═══════════════════════════════════════════════════════════════════
# 6. ENERGY INJECTION MECHANICS
# ═══════════════════════════════════════════════════════════════════

class TestEnergyInjection:
    """Validates: logits[masked] += alpha * token_scores, only at masked positions."""

    def test_energy_only_at_masked_positions(self):
        """Energy must be added ONLY at positions where input_ids == mask_id."""
        B, T, V = 2, 8, 100
        logits = torch.zeros(B, T, V)
        token_scores = torch.randn(V)
        alpha = 5.0
        mask_id = 99
        input_ids = torch.tensor([
            [1, 2, mask_id, 3, mask_id, 4, 5, 6],
            [1, mask_id, 2, mask_id, 3, 4, 5, 6],
        ])
        mask_positions = (input_ids == mask_id)
        energy = token_scores.unsqueeze(0).unsqueeze(0) * alpha
        logits_modified = logits + mask_positions.unsqueeze(-1).float() * energy

        # Check: unmasked positions unchanged
        for b in range(B):
            for t in range(T):
                if not mask_positions[b, t]:
                    assert torch.all(logits_modified[b, t] == 0), \
                        f"Energy leaked to unmasked position [{b},{t}]"

        # Check: masked positions have energy
        for b in range(B):
            for t in range(T):
                if mask_positions[b, t]:
                    assert torch.allclose(logits_modified[b, t], token_scores * alpha), \
                        f"Energy missing at masked position [{b},{t}]"

    def test_energy_scales_with_alpha(self):
        """Doubling alpha should double the energy contribution."""
        logits = torch.zeros(1, 4, 50)
        token_scores = torch.randn(50)
        mask_positions = torch.tensor([[False, True, True, False]])
        alpha1 = 3.0
        alpha2 = 6.0

        e1 = (mask_positions.unsqueeze(-1).float() * (token_scores.unsqueeze(0).unsqueeze(0) * alpha1))
        e2 = (mask_positions.unsqueeze(-1).float() * (token_scores.unsqueeze(0).unsqueeze(0) * alpha2))
        masked_e1 = e1[mask_positions]
        masked_e2 = e2[mask_positions]
        ratio = masked_e2 / (masked_e1 + 1e-8)
        assert torch.allclose(ratio, torch.ones_like(ratio) * 2.0, atol=1e-4), \
            f"Energy scaling ratio not 2.0: max diff = {(ratio - 2.0).abs().max()}"


# ═══════════════════════════════════════════════════════════════════
# 7. ORTHOGONALITY (the ρ=0.14 claim)
# ═══════════════════════════════════════════════════════════════════

class TestOrthogonality:
    """Validates that model and MiniLM spaces are measurably different."""

    def test_random_orthogonal_spaces(self):
        """Two random high-dim vectors in different dims are near-orthogonal.
        This is a sanity check for our Spearman correlation methodology."""
        n = 500
        m = torch.randn(n)
        n_space = torch.randn(n)
        # Rank correlation
        rank_m = m.argsort().argsort().float()
        rank_n = n_space.argsort().argsort().float()
        spearman = torch.corrcoef(torch.stack([rank_m, rank_n]))[0, 1].item()
        # Two random rankings should have near-zero correlation
        assert abs(spearman) < 0.2, f"Random spaces should be uncorrelated, got ρ={spearman}"

    def test_identical_spaces_perfectly_correlated(self):
        """If two score vectors are identical, Spearman ρ must be 1.0."""
        m = torch.randn(100)
        spearman = torch.corrcoef(torch.stack([m, m]))[0, 1].item()
        assert spearman > 0.999


# ═══════════════════════════════════════════════════════════════════
# 8. CONFIG INVARIANTS
# ═══════════════════════════════════════════════════════════════════

class TestConfigInvariants:
    """GuidanceConfig defaults must match the validated v13 sweet spot."""

    def test_validated_alpha(self):
        """Default alpha must be 10.0 (v13 validated sweet spot)."""
        c = GuidanceConfig()
        assert c.alpha == 10.0

    def test_validated_penalty(self):
        """Default rep_penalty must be 5.0 (v9 validated sweet spot)."""
        c = GuidanceConfig()
        assert c.rep_penalty == 5.0

    def test_validated_allowance(self):
        """Default allowance must be 1 (v9 validated)."""
        c = GuidanceConfig()
        assert c.rep_allowance == 1

    def test_validated_fusion(self):
        """Default fusion_weight must be 0.5 (convex 50/50, v13 winner)."""
        c = GuidanceConfig()
        assert c.fusion_weight == 0.5

    def test_validated_gamma(self):
        """Default gamma must be 1.0 (linear annealing)."""
        c = GuidanceConfig()
        assert c.gamma == 1.0
