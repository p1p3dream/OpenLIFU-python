from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xa

from openlifu.seg import SegmentationMethod
from openlifu.seg.material import AIR, SKULL, TISSUE, WATER
from openlifu.seg.seg_methods.nnunet_seg import (
    NNUNetSegmentation,
    LABEL_MAP_FULLHEAD,
    LABEL_MAP_SKULL,
    PATCH_SIZES,
)
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_synthetic_volume(
    shape=(64, 64, 64), spacing=1.0, intensity=100.0,
) -> xa.DataArray:
    nz, ny, nx = shape
    x = np.arange(nx) * spacing - (nx - 1) * spacing / 2
    y = np.arange(ny) * spacing - (ny - 1) * spacing / 2
    z = np.arange(nz) * spacing - (nz - 1) * spacing / 2
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    head_radius = min(shape) * spacing * 0.4
    data = np.where(dist <= head_radius, intensity, 0.0)
    return xa.DataArray(data, dims=["z", "y", "x"], coords={"z": z, "y": y, "x": x})


def make_mock_session(n_classes, output_shape):
    """Create a mock onnxruntime.InferenceSession."""
    session = MagicMock()
    # Mock get_inputs and get_outputs for shape info
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 1, *output_shape[2:]]
    session.get_inputs.return_value = [mock_input]

    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.shape = list(output_shape)
    session.get_outputs.return_value = [mock_output]

    # Mock run to return random logits
    def mock_run(output_names, input_feed):
        batch = input_feed["input"]
        spatial = batch.shape[2:]
        result = np.random.randn(1, n_classes, *spatial).astype(np.float32)
        return [result]

    session.run = mock_run
    return session


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestNNUNetSegmentationConstruction:

    def test_default_construction(self):
        seg = NNUNetSegmentation()
        assert isinstance(seg, NNUNetSegmentation)
        assert isinstance(seg, SegmentationMethod)
        assert seg.model_type == "fullhead"
        assert seg.model_path == ""
        assert seg.use_gpu is False
        assert seg.use_mirroring is True
        assert seg.ref_material == "water"

    def test_skull_construction(self):
        seg = NNUNetSegmentation(model_type="skull")
        assert seg.model_type == "skull"
        assert "tissue" in seg.materials
        assert "csf" not in seg.materials

    def test_fullhead_materials_auto_swap(self):
        seg = NNUNetSegmentation(model_type="fullhead")
        assert "csf" in seg.materials
        assert "gray_matter" in seg.materials
        assert "white_matter" in seg.materials
        assert "tissue" not in seg.materials

    def test_skull_materials(self):
        seg = NNUNetSegmentation(model_type="skull")
        assert "water" in seg.materials
        assert "skull" in seg.materials
        assert "tissue" in seg.materials
        assert "air" in seg.materials

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            NNUNetSegmentation(model_type="invalid_model")


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestNNUNetSegmentationSerialization:

    def test_dict_roundtrip_skull(self):
        seg = NNUNetSegmentation(model_type="skull")
        d = seg.to_dict()
        assert d["class"] == "NNUNetSegmentation"
        reconstructed = SegmentationMethod.from_dict(d)
        assert isinstance(reconstructed, NNUNetSegmentation)
        assert reconstructed.model_type == "skull"

    def test_dict_roundtrip_fullhead(self):
        seg = NNUNetSegmentation(model_type="fullhead", use_gpu=True)
        d = seg.to_dict()
        reconstructed = SegmentationMethod.from_dict(d)
        assert isinstance(reconstructed, NNUNetSegmentation)
        assert reconstructed.model_type == "fullhead"
        assert reconstructed.use_gpu is True
        assert "csf" in reconstructed.materials

    def test_to_table(self):
        seg = NNUNetSegmentation()
        table = seg.to_table()
        assert len(table) >= 4
        names = set(table["Name"].tolist())
        assert "Type" in names
        assert "Model Type" in names


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_zscore_known_data(self):
        """Z-score on known data produces mean near 0, std near 1."""
        data = np.random.RandomState(42).randn(50, 50, 50).astype(np.float32) * 10 + 50
        mask = data > 0
        mean = data[mask].mean()
        std = data[mask].std()
        normalized = (data - mean) / std
        assert abs(normalized[mask].mean()) < 0.01
        assert abs(normalized[mask].std() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Gaussian kernel tests
# ---------------------------------------------------------------------------

class TestGaussianKernel:

    def test_shape_and_symmetry(self):
        shape = (16, 24, 32)
        seg = NNUNetSegmentation()
        g = seg._build_gaussian_kernel(shape)
        assert g.shape == shape
        np.testing.assert_allclose(g, g[::-1, :, :], atol=1e-12)
        np.testing.assert_allclose(g, g[:, ::-1, :], atol=1e-12)
        np.testing.assert_allclose(g, g[:, :, ::-1], atol=1e-12)
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
        assert g[center] == g.max()


# ---------------------------------------------------------------------------
# Segment with mocked ONNX
# ---------------------------------------------------------------------------

class TestSegmentMocked:

    def test_segment_returns_valid_shape(self):
        """Mock ONNX session and verify _segment returns correct shape and valid labels."""
        seg = NNUNetSegmentation(model_type="skull", use_mirroring=False)
        volume = create_synthetic_volume(shape=(32, 32, 32))
        n_classes = len(LABEL_MAP_SKULL)
        patch_size = PATCH_SIZES["skull"]

        mock_session = make_mock_session(n_classes, (1, n_classes, *patch_size))

        with patch.object(NNUNetSegmentation, "_get_session", return_value=mock_session):
            result = seg._segment(volume)

        assert result.shape == volume.shape
        unique = set(np.unique(result.to_numpy()))
        material_idx = seg._material_indices()
        valid_indices = set(material_idx.values())
        assert unique.issubset(valid_indices)

    def test_segment_fullhead_returns_valid_labels(self):
        """Fullhead model should produce valid 7-class material indices."""
        seg = NNUNetSegmentation(model_type="fullhead", use_mirroring=False)
        volume = create_synthetic_volume(shape=(32, 32, 32))
        n_classes = len(LABEL_MAP_FULLHEAD)
        patch_size = PATCH_SIZES["fullhead"]

        mock_session = make_mock_session(n_classes, (1, n_classes, *patch_size))

        with patch.object(NNUNetSegmentation, "_get_session", return_value=mock_session):
            result = seg._segment(volume)

        assert result.shape == volume.shape
        material_idx = seg._material_indices()
        valid_indices = set(material_idx.values())
        unique = set(np.unique(result.to_numpy()))
        assert unique.issubset(valid_indices)
