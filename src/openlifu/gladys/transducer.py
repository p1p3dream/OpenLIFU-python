"""GLADYS reference transducer definition.

Defines the GLADYS Guided Localization and Acoustic Delivery System
transducer: a 64-element hemispherical array operating at 500 kHz with a
90 mm radius of curvature. This is the software transducer definition that
matches the geometry used in the N=180 transcranial validation study
(64-element hemispherical array, 500 kHz, R=90 mm).

The element placement reproduces the construction in
``gladys-validation-batch-2026-05-17/sweep_bone_attenuation.py`` (function
``load_array``): elements are distributed over a spherical cap on the
positive-z side of a sphere of radius ``radius_mm`` using a half-angle
``theta_max`` derived from the aperture, with polar angle spaced by a
cosine-weighted rule and azimuthal angle advanced by the golden angle.

The transducer is defined in its own local coordinate frame with the
geometric focus (sphere center) at the origin, concave side facing -z. Any
targeting rotation or translation onto a subject is applied externally via
transforms at planning time; it is not baked into this definition.
"""

from __future__ import annotations

import numpy as np

from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

# Reference geometry constants (N=180 transcranial validation study).
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQUENCY_HZ = 500e3
ELEMENT_SIZE_MM = 5.0


def create_gladys_transducer(
    n_elements: int = N_ELEMENTS,
    radius_mm: float = RADIUS_MM,
    aperture_mm: float = APERTURE_MM,
    freq_hz: float = FREQUENCY_HZ,
    element_size_mm: float = ELEMENT_SIZE_MM,
    transducer_id: str = "gladys_v1",
    name: str = "GLADYS v1 64-element 500 kHz hemispherical array",
) -> Transducer:
    """Create the GLADYS reference hemispherical transducer.

    Reproduces the N=180 validation geometry: a spherical-cap array of
    ``n_elements`` point-like square elements on a sphere of radius
    ``radius_mm``, spanning a half-angle ``theta_max`` set by ``aperture_mm``,
    driven at ``freq_hz``.

    Element placement math, matching ``load_array`` in
    ``sweep_bone_attenuation.py``:

        half_aperture = aperture_mm / 2.0
        theta_max = arcsin(half_aperture / radius_mm)
        golden_angle = pi * (3.0 - sqrt(5.0))
        for i in range(n_elements):
            cos_theta = 1.0 - (1.0 - cos(theta_max)) * (i + 0.5) / n_elements
            theta = arccos(cos_theta)
            phi = golden_angle * i
            x = radius_mm * sin(theta) * cos(phi)
            y = radius_mm * sin(theta) * sin(phi)
            z = radius_mm * cos(theta)

    Each element's orientation points inward toward the focus (origin), with
    azimuth/elevation derived from the inward normal. Units are millimeters
    for both the elements and the transducer.

    Args:
        n_elements: Number of array elements (default 64).
        radius_mm: Sphere radius / radius of curvature in mm (default 90.0).
        aperture_mm: Spherical-cap aperture diameter in mm; sets the angular
            extent ``theta_max`` via ``arcsin(aperture_mm/2 / radius_mm)``
            (default 80.0).
        freq_hz: Nominal driving frequency in Hz (default 500e3).
        element_size_mm: Square element side length in mm (default 5.0).
        transducer_id: Transducer identifier (default "gladys_v1").
        name: Human-readable transducer name.

    Returns:
        A :class:`openlifu.xdc.Transducer` encoding the GLADYS array in its
        local coordinate frame (focus at the origin).
    """
    half_aperture = aperture_mm / 2.0
    theta_max = np.arcsin(half_aperture / radius_mm)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    elements: list[Element] = []
    for i in range(n_elements):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)
        pos = np.array([x, y, z])

        # Inward normal pointing from the element toward the focus at origin.
        normal = -pos
        normal = normal / np.linalg.norm(normal)
        az = np.arctan2(normal[0], normal[2])
        el = -np.arctan2(normal[1], np.sqrt(normal[0] ** 2 + normal[2] ** 2))

        elements.append(Element(
            index=i + 1,
            pin=i + 1,
            position=pos,
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))

    return Transducer(
        id=transducer_id,
        name=name,
        elements=elements,
        frequency=freq_hz,
        units="mm",
    )
