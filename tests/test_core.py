"""Tests for m2m_energy_fields."""
import pytest
from m2m_energy_fields import EnergyField, GuidanceConfig


def test_energy_field_active():
    f = EnergyField(target_texts=["ocean"])
    assert f.is_active()

    f_empty = EnergyField()
    assert not f_empty.is_active()

    f_suppress = EnergyField(suppress_texts=["horror"])
    assert f_suppress.is_active()


def test_guidance_config_defaults():
    c = GuidanceConfig()
    assert c.alpha == 5.0
    assert c.temperature == 0.6
    assert c.steps == 128


def test_guidance_config_custom():
    c = GuidanceConfig(alpha=3.0, temperature=0.3)
    assert c.alpha == 3.0
    assert c.temperature == 0.3
