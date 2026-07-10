"""Array positioning and targeting for the GLADYS hemispherical transducer.

Provides functions to rotate and translate a GLADYS transducer so that its
geometric focus lands on a specified brain target and the array faces a
chosen skull entry point.

The default transducer definition (see ``transducer.py``) has its geometric
focus at the origin with elements on the +z hemisphere, concave side facing
-z. Positioning therefore requires:

    1. A rotation that maps the default beam direction [0, 0, -1] onto the
       desired direction (from entry point toward target).
    2. A translation that moves the focus from the origin to the target.

The rotation is computed with the Rodrigues formula so that arbitrary (not
just axis-aligned) entry points are supported.
"""

from __future__ import annotations

import numpy as np

from openlifu.xdc import Transducer

# Default beam direction in the local transducer frame: concave side faces -z.
_DEFAULT_BEAM_DIR = np.array([0.0, 0.0, -1.0])


def _rodrigues_rotation_matrix(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute a 3x3 rotation matrix that maps unit vector *source* to *target*.

    Uses the Rodrigues rotation formula:

        R = I * cos(theta)
            + (1 - cos(theta)) * (k outer k)
            + sin(theta) * K

    where *k* is the unit rotation axis (cross product of source and target),
    *theta* is the angle between them, and *K* is the skew-symmetric cross-
    product matrix of *k*.

    Edge cases:
        - If source and target are (nearly) parallel, returns the identity.
        - If source and target are (nearly) antiparallel, returns a 180-degree
          rotation about an arbitrary axis perpendicular to source.

    Args:
        source: Unit vector (3,) to rotate from.
        target: Unit vector (3,) to rotate to.

    Returns:
        A 3x3 orthonormal rotation matrix.
    """
    cross = np.cross(source, target)
    sin_theta = np.linalg.norm(cross)
    cos_theta = float(np.dot(source, target))

    # Nearly parallel: no rotation needed.
    if sin_theta < 1e-8:
        if cos_theta > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees around any perpendicular axis.
        # Pick the coordinate axis least aligned with source.
        abs_src = np.abs(source)
        min_axis = int(np.argmin(abs_src))
        perp = np.zeros(3)
        perp[min_axis] = 1.0
        # Gram-Schmidt to get a true perpendicular.
        perp = perp - np.dot(perp, source) * source
        perp = perp / np.linalg.norm(perp)
        # 180-degree rotation: R = 2 * (k outer k) - I
        return 2.0 * np.outer(perp, perp) - np.eye(3)

    k = cross / sin_theta  # unit rotation axis

    # Skew-symmetric cross-product matrix of k.
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])

    R = (
        np.eye(3) * cos_theta
        + (1.0 - cos_theta) * np.outer(k, k)
        + sin_theta * K
    )
    return R


def position_transducer(
    transducer: Transducer,
    target_mm: np.ndarray,
    entry_point_mm: np.ndarray,
) -> Transducer:
    """Create a positioned copy of a transducer aimed at a brain target.

    Takes a transducer defined in its local frame (focus at origin, concave
    side facing -z) and returns a new Transducer whose element positions and
    orientations have been transformed so that:

    - The geometric focus coincides with *target_mm*.
    - The array faces the skull entry point, with the beam axis aligned along
      the direction from *entry_point_mm* toward *target_mm*.

    The input transducer is not modified.

    The transformation is:

        p_world = R @ p_local + target_mm

    where R is the Rodrigues rotation that maps the default beam direction
    [0, 0, -1] onto ``normalize(target_mm - entry_point_mm)``.

    After positioning, each element's orientation (az, el, roll) is recomputed
    so that it points inward toward the focus (target), matching the convention
    used throughout the OpenLIFU pipeline.

    Args:
        transducer: A Transducer in its local coordinate frame (as produced
            by ``create_gladys_transducer``). Must use ``units="mm"``.
        target_mm: 3-element array giving the treatment target position in
            world coordinates (mm).
        entry_point_mm: 3-element array giving the skull entry point in
            world coordinates (mm). The array will be oriented so the beam
            travels from this point toward the target.

    Returns:
        A new Transducer with element positions in world coordinates (mm)
        and orientations pointing toward the target.

    Raises:
        ValueError: If the entry point and target are coincident.
    """
    target_mm = np.asarray(target_mm, dtype=float)
    entry_point_mm = np.asarray(entry_point_mm, dtype=float)

    if target_mm.shape != (3,):
        raise ValueError(
            f"target_mm must have 3 elements, got shape {target_mm.shape}"
        )
    if entry_point_mm.shape != (3,):
        raise ValueError(
            f"entry_point_mm must have 3 elements, got shape {entry_point_mm.shape}"
        )

    # Desired beam direction: from entry point toward target.
    beam_vec = target_mm - entry_point_mm
    beam_len = np.linalg.norm(beam_vec)
    if beam_len < 1e-6:
        raise ValueError(
            "entry_point_mm and target_mm are coincident (or nearly so); "
            "cannot determine a beam direction."
        )
    desired_dir = beam_vec / beam_len

    # Rotation from default beam direction to the desired direction.
    R = _rodrigues_rotation_matrix(_DEFAULT_BEAM_DIR, desired_dir)

    # Build a deep copy so the original is untouched.
    positioned = transducer.copy()

    for el in positioned.elements:
        # Transform position: rotate then translate to target.
        local_pos = el.position  # already in mm (same units as transducer)
        world_pos = R @ local_pos + target_mm
        el.position = world_pos

        # Recompute orientation so the element faces the focus (target).
        direction = target_mm - world_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            n = direction / dist
            az = np.arctan2(n[0], n[2])
            el_angle = -np.arctan2(n[1], np.sqrt(n[0] ** 2 + n[2] ** 2))
            el.orientation = np.array([az, el_angle, 0.0])

    return positioned
