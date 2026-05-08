from __future__ import annotations

from dataclasses import asdict

import pytest

from openlifu.seg.material import (
    CORTICAL_BONE,
    MATERIALS,
    MATERIALS_TWO_CLASS_BONE,
    SKULL,
    TRABECULAR_BONE,
    Material,
)

# Mock PARAM_INFO for tests
PARAM_INFO = {
    "name": {"label": "Material name", "description": "Name for the material"},
    "sound_speed": {"label": "Sound speed (m/s)", "description": "Speed of sound in the material (m/s)"},
    "density": {"label": "Density (kg/m^3)", "description": "Mass density of the material (kg/m^3)"},
    "attenuation": {"label": "Attenuation (dB/cm/MHz)", "description": "Ultrasound attenuation in the material (dB/cm/MHz)"},
    "specific_heat": {"label": "Specific heat (J/kg/K)", "description": "Specific heat capacity of the material (J/kg/K)"},
    "thermal_conductivity": {"label": "Thermal conductivity (W/m/K)", "description": "Thermal conductivity of the material (W/m/K)"},
}

@pytest.fixture()
def default_material():
    return Material()

def test_default_material_values(default_material):
    assert default_material.name == "Material"
    assert default_material.sound_speed == 1500.0
    assert default_material.density == 1000.0
    assert default_material.attenuation == 0.0
    assert default_material.specific_heat == 4182.0
    assert default_material.thermal_conductivity == 0.598

def test_material_to_dict(default_material):
    expected = asdict(default_material)
    assert default_material.to_dict() == expected

def test_material_from_dict():
    data = {
        "name": "test",
        "sound_speed": 1234.0,
        "density": 999.0,
        "attenuation": 1.2,
        "specific_heat": 4000.0,
        "thermal_conductivity": 0.5
    }
    material = Material.from_dict(data)
    assert material.to_dict() == data

def test_get_param_valid(default_material):
    assert default_material.get_param("density") == 1000.0

def test_get_param_invalid(default_material):
    with pytest.raises(ValueError, match="Parameter fake_param not found."):
        default_material.get_param("fake_param")

def test_param_info_valid(monkeypatch):
    monkeypatch.setattr("openlifu.seg.material.PARAM_INFO", PARAM_INFO)
    info = Material.param_info("density")
    assert info["label"] == "Density (kg/m^3)"

def test_param_info_invalid(monkeypatch):
    monkeypatch.setattr("openlifu.seg.material.PARAM_INFO", PARAM_INFO)
    with pytest.raises(ValueError, match="Parameter unknown not found."):
        Material.param_info("unknown")


# ---- Two-class bone material tests ----


class TestCorticalBone:
    """Tests for the CORTICAL_BONE material definition."""

    def test_name(self):
        assert CORTICAL_BONE.name == "cortical_bone"

    def test_itrusst_properties(self):
        """Verify ITRUSST benchmark values: c=2800, rho=1850, alpha=4.0."""
        assert CORTICAL_BONE.sound_speed == 2800.0
        assert CORTICAL_BONE.density == 1850.0
        assert CORTICAL_BONE.attenuation == 4.0

    def test_thermal_properties_positive(self):
        assert CORTICAL_BONE.specific_heat > 0
        assert CORTICAL_BONE.thermal_conductivity > 0

    def test_serialization_roundtrip(self):
        d = CORTICAL_BONE.to_dict()
        restored = Material.from_dict(d)
        assert restored.sound_speed == CORTICAL_BONE.sound_speed
        assert restored.density == CORTICAL_BONE.density
        assert restored.attenuation == CORTICAL_BONE.attenuation
        assert restored.name == CORTICAL_BONE.name


class TestTrabecularBone:
    """Tests for the TRABECULAR_BONE material definition."""

    def test_name(self):
        assert TRABECULAR_BONE.name == "trabecular_bone"

    def test_itrusst_properties(self):
        """Verify ITRUSST benchmark values: c=2300, rho=1700, alpha=8.0."""
        assert TRABECULAR_BONE.sound_speed == 2300.0
        assert TRABECULAR_BONE.density == 1700.0
        assert TRABECULAR_BONE.attenuation == 8.0

    def test_thermal_properties_positive(self):
        assert TRABECULAR_BONE.specific_heat > 0
        assert TRABECULAR_BONE.thermal_conductivity > 0

    def test_serialization_roundtrip(self):
        d = TRABECULAR_BONE.to_dict()
        restored = Material.from_dict(d)
        assert restored.sound_speed == TRABECULAR_BONE.sound_speed
        assert restored.density == TRABECULAR_BONE.density
        assert restored.attenuation == TRABECULAR_BONE.attenuation
        assert restored.name == TRABECULAR_BONE.name


class TestTwoClassBoneRelationships:
    """Tests for physical relationships between bone material types."""

    def test_cortical_faster_than_trabecular(self):
        """Cortical bone has higher speed of sound than trabecular."""
        assert CORTICAL_BONE.sound_speed > TRABECULAR_BONE.sound_speed

    def test_cortical_denser_than_trabecular(self):
        """Cortical bone is denser than trabecular."""
        assert CORTICAL_BONE.density > TRABECULAR_BONE.density

    def test_trabecular_more_attenuating_than_cortical(self):
        """Trabecular bone has higher attenuation than cortical."""
        assert TRABECULAR_BONE.attenuation > CORTICAL_BONE.attenuation

    def test_both_slower_than_single_skull(self):
        """Both bone types have lower speed of sound than the homogeneous skull model."""
        assert CORTICAL_BONE.sound_speed < SKULL.sound_speed
        assert TRABECULAR_BONE.sound_speed < SKULL.sound_speed

    def test_materials_dict_contains_bone_types(self):
        """The two-class materials dict replaces skull with cortical + trabecular."""
        assert "cortical_bone" in MATERIALS_TWO_CLASS_BONE
        assert "trabecular_bone" in MATERIALS_TWO_CLASS_BONE
        assert "skull" not in MATERIALS_TWO_CLASS_BONE

    def test_standard_materials_unchanged(self):
        """The standard MATERIALS dict still has 'skull' and no bone subtypes."""
        assert "skull" in MATERIALS
        assert "cortical_bone" not in MATERIALS
        assert "trabecular_bone" not in MATERIALS

    def test_two_class_has_all_non_bone_materials(self):
        """The two-class dict preserves all non-bone materials from the standard set."""
        for key in ("water", "tissue", "air", "standoff"):
            assert key in MATERIALS_TWO_CLASS_BONE
            assert MATERIALS_TWO_CLASS_BONE[key].sound_speed == MATERIALS[key].sound_speed
