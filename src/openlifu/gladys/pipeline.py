"""GLADYS treatment planning pipeline.

Orchestrates the complete workflow from a raw MRI volume to a hardware-
ready sonication solution:

    1. Load MRI (NIfTI) as an xarray.DataArray
    2. Construct a target Point from coordinates
    3. Run tissue segmentation (ThresholdMRI with brain classification)
    4. Compute aberration-corrected transmit delays (SimulationCorrected)
    5. Run k-wave simulation and scale to target pressure
    6. Upload the resulting Solution to the LIFU hardware

Usage::

    from openlifu.gladys import GLADYSPipeline

    pipeline = GLADYSPipeline()
    solution, sim_result, analysis = pipeline.plan_treatment(
        mri_path="/path/to/T1w.nii.gz",
        target_position=[10.0, -5.0, 45.0],
    )
    pipeline.upload_to_hardware(solution, interface)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xa

from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.protocol import Protocol
from openlifu.plan.solution import Solution
from openlifu.plan.solution_analysis import SolutionAnalysis
from openlifu.xdc import Transducer

from .config import DEFAULT_VOLTAGE, default_protocol

logger = logging.getLogger(__name__)


class GLADYSPipeline:
    """End-to-end treatment planning pipeline for the GLADYS platform.

    Wraps OpenLIFU's Protocol/Solution machinery with GLADYS-specific
    defaults (500 kHz SDT, 86 ms pulse, ThresholdMRI segmentation,
    SimulationCorrected delays) so a treatment plan can be produced
    from just an MRI path and a target coordinate.

    Args:
        transducer: The Transducer to plan for. Required for simulation.
            If None, must be provided later via ``plan_treatment``.
        protocol: A fully configured Protocol. If None, the GLADYS
            default protocol is used.
        voltage: Transmit voltage (V) for the solution. Overridden by
            simulation scaling when ``simulate=True``.
    """

    def __init__(
        self,
        transducer: Optional[Transducer] = None,
        protocol: Optional[Protocol] = None,
        voltage: float = DEFAULT_VOLTAGE,
    ) -> None:
        self.transducer = transducer
        self.protocol = protocol if protocol is not None else default_protocol()
        self.voltage = voltage

    # ------------------------------------------------------------------
    # MRI loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_mri(mri_path: Union[str, Path]) -> xa.DataArray:
        """Load a NIfTI MRI volume and convert it to an xarray.DataArray.

        The returned DataArray has dimensions ``('x', 'y', 'z')`` with
        coordinates in millimeters derived from the NIfTI affine. This
        is the coordinate convention expected by the OpenLIFU simulation
        and segmentation pipeline.

        Args:
            mri_path: Path to a NIfTI file (``.nii`` or ``.nii.gz``).

        Returns:
            An xarray.DataArray with spatial coordinates in mm.

        Raises:
            FileNotFoundError: If *mri_path* does not exist.
            ImportError: If nibabel is not installed.
        """
        mri_path = Path(mri_path)
        if not mri_path.exists():
            raise FileNotFoundError(f"MRI file not found: {mri_path}")

        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "nibabel is required for MRI loading. "
                "Install it with: pip install nibabel"
            ) from exc

        img = nib.load(str(mri_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine

        # Build mm-scale coordinate arrays from the affine.
        # Assumes an axis-aligned affine (diagonal voxel-to-world mapping)
        # which is the common case after reorientation to RAS/LPS.
        coords = {}
        dim_names = ("x", "y", "z")
        for axis, dim in enumerate(dim_names):
            n_voxels = data.shape[axis]
            origin = affine[axis, 3]
            spacing = affine[axis, axis]
            coord_values = origin + np.arange(n_voxels) * spacing
            coords[dim] = xa.Variable(
                dim,
                coord_values,
                attrs={"units": "mm"},
            )

        volume = xa.DataArray(
            data,
            dims=dim_names,
            coords=coords,
            attrs={"source": str(mri_path)},
        )
        logger.info(
            "Loaded MRI volume from %s: shape=%s, spacing=[%.2f, %.2f, %.2f] mm",
            mri_path.name,
            data.shape,
            abs(affine[0, 0]),
            abs(affine[1, 1]),
            abs(affine[2, 2]),
        )
        return volume

    # ------------------------------------------------------------------
    # Treatment planning
    # ------------------------------------------------------------------

    def plan_treatment(
        self,
        mri_path: Union[str, Path],
        target_position: Sequence[float],
        transducer: Optional[Transducer] = None,
        target_units: str = "mm",
        simulate: bool = True,
        scale: bool = True,
    ) -> Tuple[Solution, xa.Dataset, SolutionAnalysis]:
        """Run the full GLADYS treatment planning pipeline.

        Loads the MRI, segments the volume, computes transmit delays with
        skull aberration correction, runs a k-wave pressure simulation,
        and (optionally) scales the solution to achieve the protocol's
        target pressure.

        Args:
            mri_path: Path to the T1-weighted NIfTI MRI volume.
            target_position: 3-element sequence of ``[x, y, z]``
                coordinates for the treatment target, in *target_units*.
            transducer: Override the pipeline's transducer for this run.
            target_units: Spatial units for *target_position*
                (default ``"mm"``).
            simulate: Whether to run the k-wave simulation
                (default True).
            scale: Whether to rescale the solution to the protocol's
                target pressure after simulation (default True).

        Returns:
            A 3-tuple of ``(solution, simulation_result, analysis)``:

            - **solution**: The computed Solution with delays,
              apodizations, and (if simulated) pressure fields.
            - **simulation_result**: Aggregated simulation output as an
              xarray.Dataset, or None if ``simulate=False``.
            - **analysis**: A SolutionAnalysis with acoustic safety
              metrics, or None if ``simulate=False``.

        Raises:
            ValueError: If no transducer is available.
            FileNotFoundError: If the MRI file does not exist.
        """
        arr = transducer or self.transducer
        if arr is None:
            raise ValueError(
                "A Transducer must be provided either at pipeline construction "
                "or as an argument to plan_treatment."
            )

        # Step 1: Load MRI volume
        logger.info("GLADYS pipeline: loading MRI from %s", mri_path)
        volume = self.load_mri(mri_path)

        # Step 2: Build the target Point
        pos = np.asarray(target_position, dtype=float)
        if pos.shape != (3,):
            raise ValueError(
                f"target_position must have exactly 3 elements, got shape {pos.shape}"
            )
        target = Point(
            position=pos,
            id="gladys_target",
            name="GLADYS Target",
            units=target_units,
        )
        logger.info(
            "GLADYS pipeline: target at [%.2f, %.2f, %.2f] %s",
            pos[0], pos[1], pos[2], target_units,
        )

        # Step 3: Compute the solution (segments, beamforms, simulates, scales)
        logger.info("GLADYS pipeline: computing solution via Protocol.calc_solution")
        solution, sim_result, analysis = self.protocol.calc_solution(
            target=target,
            transducer=arr,
            volume=volume,
            simulate=simulate,
            scale=scale,
            voltage=self.voltage,
        )

        logger.info("GLADYS pipeline: solution '%s' computed successfully", solution.id)
        if analysis is not None:
            logger.info(
                "  PNP=%.3f MPa, MI=%.3f, ISPTA=%.3f mW/cm^2",
                max(analysis.mainlobe_pnp_MPa) if analysis.mainlobe_pnp_MPa else 0.0,
                analysis.MI,
                analysis.global_ispta_mWcm2,
            )

        return solution, sim_result, analysis

    # ------------------------------------------------------------------
    # Hardware upload
    # ------------------------------------------------------------------

    @staticmethod
    def upload_to_hardware(
        solution: Solution,
        interface: LIFUInterface,
        profile_index: int = 1,
    ) -> None:
        """Upload a computed solution to the LIFU hardware.

        Validates the solution against hardware safety limits, programs
        the transmit delays and apodizations, and sets the HV voltage.

        Args:
            solution: A Solution object (typically from ``plan_treatment``).
            interface: An initialized LIFUInterface connected to hardware.
            profile_index: Hardware profile slot to program (default 1).

        Raises:
            ValueError: If the solution violates hardware voltage or
                duty-cycle limits.
        """
        logger.info(
            "GLADYS pipeline: uploading solution '%s' to hardware (profile %d)",
            solution.id,
            profile_index,
        )
        interface.set_solution(
            solution=solution,
            profile_index=profile_index,
        )
        logger.info("GLADYS pipeline: solution uploaded successfully")
