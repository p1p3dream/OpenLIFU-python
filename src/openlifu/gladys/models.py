"""ONNX model download, caching, and integrity validation for GLADYS.

Models are cached under ``~/.openlifu/models/`` and validated by SHA-256
checksum. On first use, the requested model is automatically downloaded
from the configured URL (GitHub Releases once published).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR: Path = Path.home() / ".openlifu" / "models"
"""Default directory for cached ONNX model files."""

# ---------------------------------------------------------------------------
# Model registry
#
# Each entry maps a short model name to its download URL and expected
# SHA-256 digest. URLs are placeholders until the models are published
# on GitHub Releases.
# ---------------------------------------------------------------------------

ModelInfo = Dict[str, str]

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "fullhead_seg_v1": {
        "url": "https://github.com/OpenwaterHealth/OpenLIFU-python/releases/download/models-v1/fullhead_seg_v1.onnx",
        "sha256": "placeholder_sha256_will_be_updated_on_release",
        "filename": "fullhead_seg_v1.onnx",
    },
    "skull_seg_v1": {
        "url": "https://github.com/OpenwaterHealth/OpenLIFU-python/releases/download/models-v1/skull_seg_v1.onnx",
        "sha256": "placeholder_sha256_will_be_updated_on_release",
        "filename": "skull_seg_v1.onnx",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model_path(
    model_name: str,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Return the local path to a cached ONNX model, downloading if needed.

    Args:
        model_name: Key into ``MODEL_REGISTRY`` (e.g. ``"fullhead_seg_v1"``).
        cache_dir: Override for the cache directory. Defaults to
            ``~/.openlifu/models/``.
        force_download: When True, re-download even if the file already
            exists locally.

    Returns:
        Absolute path to the validated ONNX file on disk.

    Raises:
        KeyError: If ``model_name`` is not in the registry.
        RuntimeError: If the download fails or the checksum does not match.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {sorted(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[model_name]
    cache = cache_dir or DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / info["filename"]

    if local_path.exists() and not force_download:
        if _validate_checksum(local_path, info["sha256"]):
            logger.debug("Model '%s' found in cache: %s", model_name, local_path)
            return local_path
        logger.warning(
            "Cached model '%s' failed checksum validation. Re-downloading.",
            model_name,
        )

    _download_model(info["url"], local_path)

    if not _validate_checksum(local_path, info["sha256"]):
        local_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded model '{model_name}' failed SHA-256 validation. "
            "The file has been removed. Please try again or verify the URL."
        )

    logger.info("Model '%s' downloaded and validated: %s", model_name, local_path)
    return local_path


def list_cached_models(cache_dir: Optional[Path] = None) -> list[Path]:
    """List all ONNX files present in the cache directory.

    Args:
        cache_dir: Override for the cache directory. Defaults to
            ``~/.openlifu/models/``.

    Returns:
        Sorted list of paths to cached ``.onnx`` files.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    if not cache.exists():
        return []
    return sorted(cache.glob("*.onnx"))


def clear_cache(cache_dir: Optional[Path] = None) -> int:
    """Remove all cached model files.

    Args:
        cache_dir: Override for the cache directory. Defaults to
            ``~/.openlifu/models/``.

    Returns:
        Number of files removed.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    if not cache.exists():
        return 0
    removed = 0
    for f in cache.glob("*.onnx"):
        f.unlink()
        removed += 1
        logger.info("Removed cached model: %s", f)
    return removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_model(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest*, creating parent dirs as needed.

    Raises RuntimeError on any download failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from %s ...", url)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc


def _validate_checksum(path: Path, expected_sha256: str) -> bool:
    """Return True if the SHA-256 of *path* matches *expected_sha256*.

    If the expected hash is the placeholder string, validation is skipped
    (always returns True) so development can proceed before models are
    published.
    """
    if expected_sha256.startswith("placeholder"):
        logger.debug(
            "Skipping checksum validation for %s (placeholder hash).", path.name
        )
        return True

    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != expected_sha256:
        logger.error(
            "Checksum mismatch for %s: expected %s, got %s",
            path.name,
            expected_sha256,
            digest,
        )
        return False
    return True
