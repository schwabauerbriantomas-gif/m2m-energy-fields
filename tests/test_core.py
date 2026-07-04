"""Tests for m2m_energy_fields."""
import pytest
import torch
from m2m_energy_fields import EnergyField, GuidanceConfig, fuse_convex, fuse_rrf


def test_energy_field_active():
    f = EnergyField(target_texts=["ocean"])
    assert f.is_active()

    f_empty = EnergyField()
    assert not f_empty.is_active()

    f_suppress = EnergyField(suppress_texts=["horror"])
    assert f_suppress.is_active()


def test_guidance_config_defaults():
    c = GuidanceConfig()
    assert c.alpha == 10.0
    assert c.alpha_end == 0.0
    assert c.temperature == 0.6
    assert c.steps == 64
    assert c.rep_penalty == 5.0
    assert c.rep_allowance == 1
    assert c.fusion_weight == 0.5


def test_guidance_config_custom():
    c = GuidanceConfig(alpha=5.0, temperature=0.3, rep_penalty=3.0)
    assert c.alpha == 5.0
    assert c.temperature == 0.3
    assert c.rep_penalty == 3.0


def test_fuse_convex():
    m = torch.tensor([1.0, 0.0, 0.5])
    n = torch.tensor([0.0, 1.0, 0.5])
    # w=0.5: [0.5, 0.5, 0.5]
    fused = fuse_convex(m, n, w=0.5)
    expected = torch.tensor([0.5, 0.5, 0.5])
    assert torch.allclose(fused, expected, atol=1e-6)

    # w=1.0: pure model
    fused_model = fuse_convex(m, n, w=1.0)
    assert torch.allclose(fused_model, m, atol=1e-6)

    # w=0.0: pure minilm
    fused_minilm = fuse_convex(m, n, w=0.0)
    assert torch.allclose(fused_minilm, n, atol=1e-6)


def test_fuse_rrf():
    m = torch.tensor([3.0, 2.0, 1.0])
    n = torch.tensor([1.0, 3.0, 2.0])
    fused = fuse_rrf(m, n, k=60)
    # Both rank token 0 and 1 highly → they should get highest fused scores
    assert fused[0] > fused[2]
    assert fused[1] > fused[2]
