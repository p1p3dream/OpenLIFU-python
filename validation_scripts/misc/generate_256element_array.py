#!/usr/bin/env python3
"""Generate and validate a 256-element hemispherical transducer array for transcranial FUS at 500 kHz."""

import json
import sys
from pathlib import Path

import numpy as np

# Ensure openlifu is importable from the source tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openlifu.xdc.element import Element
from openlifu.xdc.transducer import Transducer

SCRIPT_DIR = Path(__file__).resolve().parent

# Array parameters
N_ELEMENTS = 256
RADIUS_MM = 100.0       # radius of curvature
APERTURE_MM = 110.0     # full aperture diameter
ELEMENT_SIZE_MM = 5.0
FREQ_HZ = 500e3


def create_hemispherical_array(
    n_elements: int = N_ELEMENTS,
    radius_mm: float = RADIUS_MM,
    aperture_mm: float = APERTURE_MM,
    freq_hz: float = FREQ_HZ,
    element_size_mm: float = ELEMENT_SIZE_MM,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
    if half_aperture > radius_mm:
        raise ValueError(
            f"Aperture radius ({half_aperture}mm) exceeds sphere radius ({radius_mm}mm)"
        )
    theta_max = np.arcsin(half_aperture / radius_mm)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    elements = []
    for i in range(n_elements):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)
        nx, ny, nz = -x, -y, -z
        az = np.arctan2(nx, nz)
        el = -np.arctan2(ny, np.sqrt(nx**2 + nz**2))
        elements.append(Element(
            index=i + 1, pin=i + 1,
            position=np.array([x, y, z]),
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))
    return Transducer(
        id="hemi256",
        name=f"Hemispherical {n_elements}-element array",
        elements=elements, frequency=freq_hz, units="mm",
    )


def compute_inter_element_distances(positions: np.ndarray) -> np.ndarray:
    """Return the nearest-neighbour distance for each element."""
    n = positions.shape[0]
    nn_dists = np.empty(n)
    for i in range(n):
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        nn_dists[i] = dists.min()
    return nn_dists


def print_summary(tx: Transducer, radius_mm: float, aperture_mm: float) -> None:
    positions = np.array([e.position for e in tx.elements])
    nn_dists = compute_inter_element_distances(positions)

    # Polar angles (theta) from +z axis
    r = np.linalg.norm(positions, axis=1)
    thetas = np.arccos(positions[:, 2] / r)

    # Spherical cap solid angle: 2*pi*(1 - cos(theta_max))
    theta_max = thetas.max()
    cap_solid_angle = 2.0 * np.pi * (1.0 - np.cos(theta_max))

    f_number = radius_mm / aperture_mm

    print("=" * 55)
    print(f"  256-Element Hemispherical Array Summary")
    print("=" * 55)
    print(f"  Elements:           {len(tx.elements)}")
    print(f"  Frequency:          {tx.frequency / 1e3:.0f} kHz")
    print(f"  ROC:                {radius_mm:.1f} mm")
    print(f"  Aperture diameter:  {aperture_mm:.1f} mm")
    print(f"  F#:                 {f_number:.3f}")
    print(f"  Element size:       {ELEMENT_SIZE_MM:.1f} mm")
    print("-" * 55)
    print(f"  Theta range:        {np.degrees(thetas.min()):.2f} - {np.degrees(thetas.max()):.2f} deg")
    print(f"  Cap solid angle:    {cap_solid_angle:.4f} sr")
    print(f"  Element density:    {len(tx.elements) / cap_solid_angle:.1f} elements/sr")
    print("-" * 55)
    print(f"  Inter-element spacing (nearest neighbour):")
    print(f"    Mean:  {nn_dists.mean():.2f} mm")
    print(f"    Min:   {nn_dists.min():.2f} mm")
    print(f"    Max:   {nn_dists.max():.2f} mm")
    print("=" * 55)


def save_plot(tx: Transducer, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skipping plot save to {path}")
        return

    positions = np.array([e.position for e in tx.elements])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=positions[:, 2], cmap="viridis", s=12, edgecolors="k", linewidths=0.3,
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Hemispherical {len(tx.elements)}-element array")
    fig.colorbar(sc, ax=ax, label="Z (mm)", shrink=0.6)
    ax.set_box_aspect([1, 1, 1])
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {path}")


def main() -> None:
    tx = create_hemispherical_array()
    print_summary(tx, RADIUS_MM, APERTURE_MM)

    # Save JSON
    json_path = SCRIPT_DIR / "hemi256_500khz.json"
    with open(json_path, "w") as f:
        json.dump(tx.to_dict(), f, indent=2)
    print(f"  JSON saved to {json_path}")

    # Save plot (optional, requires matplotlib)
    plot_path = SCRIPT_DIR / "hemi256_layout.png"
    save_plot(tx, plot_path)


if __name__ == "__main__":
    main()
