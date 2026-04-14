"""GLADYS: Guided Localization and Acoustic Delivery System.

This module provides the GLADYS workflow components for OpenLIFU,
including threshold-based MRI segmentation, simulation-corrected
beamforming, and integrated treatment planning utilities.

Typical usage::

    from openlifu.gladys import GLADYSPipeline

    pipeline = GLADYSPipeline(transducer=my_transducer)
    solution, sim_result, analysis = pipeline.plan_treatment(
        mri_path="T1w.nii.gz",
        target_position=[10.0, -5.0, 45.0],
    )
"""

from openlifu.gladys.config import (
    DEFAULT_CFL,
    DEFAULT_SPEED_OF_SOUND,
    DEFAULT_VOLTAGE,
    SDT_FREQUENCY_HZ,
    SDT_PULSE_DURATION_S,
    TREATMENT_DURATION_S,
    default_delay_method,
    default_protocol,
    default_pulse,
    default_seg_method,
    default_sequence,
    default_sim_setup,
)
from openlifu.gladys.models import (
    clear_cache,
    get_model_path,
    list_cached_models,
)
from openlifu.gladys.pipeline import GLADYSPipeline

__all__ = [
    # Pipeline
    "GLADYSPipeline",
    # Config constants
    "SDT_FREQUENCY_HZ",
    "SDT_PULSE_DURATION_S",
    "TREATMENT_DURATION_S",
    "DEFAULT_SPEED_OF_SOUND",
    "DEFAULT_CFL",
    "DEFAULT_VOLTAGE",
    # Config factory functions
    "default_protocol",
    "default_pulse",
    "default_sequence",
    "default_seg_method",
    "default_delay_method",
    "default_sim_setup",
    # Model management
    "get_model_path",
    "list_cached_models",
    "clear_cache",
]
