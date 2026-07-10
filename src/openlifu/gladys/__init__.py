"""GLADYS: Guided Localization and Acoustic Delivery System.

This module provides the GLADYS workflow components for OpenLIFU,
including nnU-Net deep learning segmentation, ComplexWeighted phase
correction, tissue-masked focal analysis, array targeting, and
integrated treatment planning utilities.

Typical usage::

    from openlifu.gladys import GLADYSPipeline, create_gladys_transducer

    pipeline = GLADYSPipeline(transducer=create_gladys_transducer())
    solution, sim_result, analysis = pipeline.plan_treatment(
        mri_path="T1w.nii.gz",
        target_position=[10.0, -5.0, 45.0],
        entry_point=[10.0, -5.0, 95.0],
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
from openlifu.gladys.focal_analysis import (
    FOCAL_ROI_RADIUS_MM,
    SKULL_MARGIN_MM,
    FocalStats,
    find_focal_peak,
)
from openlifu.gladys.models import (
    clear_cache,
    get_model_path,
    list_cached_models,
)
from openlifu.gladys.pipeline import GLADYSPipeline
from openlifu.gladys.targeting import position_transducer
from openlifu.gladys.transducer import create_gladys_transducer

__all__ = [
    # Pipeline
    "GLADYSPipeline",
    # Transducer
    "create_gladys_transducer",
    # Targeting
    "position_transducer",
    # Focal analysis
    "find_focal_peak",
    "FocalStats",
    "FOCAL_ROI_RADIUS_MM",
    "SKULL_MARGIN_MM",
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
