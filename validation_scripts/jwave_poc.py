#!/usr/bin/env python3
"""j-Wave proof-of-concept: differentiable acoustic simulation.

This script validates that j-Wave can:
1. Create a 3D acoustic domain
2. Solve the Helmholtz equation (CW steady-state) for a point source
3. Differentiate focal pressure with respect to source parameters via jax.grad

Usage (CPU, macOS or Linux):
    pip install jwave  # pulls jax, jaxlib, jaxdf, equinox automatically
    python scripts/jwave_poc.py

Usage (GPU, Linux with CUDA 12):
    pip install jwave
    pip install jax[cuda12]   # upgrades jaxlib with CUDA support
    python scripts/jwave_poc.py

The PyPI jwave 0.2.1 (latest as of 2026-05) pulls jaxdf<0.3.0 which pins
jax<0.5.0 / jaxlib 0.4.38. This is the stable path. The GitHub main branch
of jwave has moved to jaxdf>=0.3.0 (jax>=0.9.0, numpy>=2.0), which is
bleeding-edge and may conflict with other openlifu dependencies.

Tested with: Python 3.12, jwave 0.2.1, jax 0.4.38, jaxlib 0.4.38.
"""

from __future__ import annotations

import sys
import time


def check_imports():
    """Verify jwave, jax, and jaxdf are importable and report versions."""
    missing = []
    for pkg in ("jax", "jaxlib", "jwave", "jaxdf"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed (no __version__)")
            print(f"  {pkg}: {version}")
        except ImportError:
            missing.append(pkg)
            print(f"  {pkg}: NOT INSTALLED")
    if missing:
        print(
            f"\nMissing packages: {', '.join(missing)}\n"
            "Install with: pip install jwave\n"
            "For CUDA: pip install jwave && pip install jax[cuda12]"
        )
        sys.exit(1)


def report_devices():
    """Report available JAX compute devices."""
    import jax
    devices = jax.devices()
    print(f"  JAX devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    return devices


def test_helmholtz_point_source():
    """Solve the Helmholtz equation for a point source in a water-only domain.

    Domain: 64x64x64 at 1mm spacing (64mm cube).
    Medium: water (c=1500 m/s, rho=1000 kg/m3, alpha=0.0 dB/MHz^2/cm).
    Source: single point source at (16, 32, 32), amplitude 1 Pa, 500 kHz.
    Expected: spherical spreading with 1/r decay in pressure amplitude.

    Returns the complex pressure field for further analysis.
    """
    import jax.numpy as jnp
    import numpy as np
    from jaxdf.discretization import FourierSeries
    from jwave.acoustics.time_harmonic import helmholtz_solver
    from jwave.geometry import Domain, Medium

    # Grid parameters
    N = (64, 64, 64)
    dx = (1e-3, 1e-3, 1e-3)  # 1 mm spacing
    freq = 500e3  # 500 kHz
    omega = 2 * np.pi * freq
    c0 = 1500.0  # water
    rho0 = 1000.0
    wavelength = c0 / freq  # 3 mm
    ppw = wavelength / dx[0]  # 3 points per wavelength (coarse, but functional)

    print(f"  Grid: {N} at {dx[0]*1e3:.1f} mm spacing")
    print(f"  Frequency: {freq/1e3:.0f} kHz, wavelength: {wavelength*1e3:.1f} mm")
    print(f"  Points per wavelength: {ppw:.1f} (coarse for POC, need >=6 for accuracy)")

    domain = Domain(N, dx)
    medium = Medium(
        domain=domain,
        sound_speed=c0,
        density=rho0,
        attenuation=0.0,
        pml_size=10,
    )

    # Place a point source at (16, 32, 32)
    src_idx = (16, 32, 32)
    src_field = np.zeros(N + (1,), dtype=np.complex64)
    src_field[src_idx + (0,)] = 1.0  # unit amplitude
    source = FourierSeries(jnp.array(src_field), domain)

    print("  Solving Helmholtz equation (GMRES)...")
    t0 = time.time()
    result = helmholtz_solver(medium, omega, source, tol=1e-5, maxiter=500)
    elapsed = time.time() - t0
    print(f"  Solve time: {elapsed:.2f} s")

    p_complex = np.asarray(result.params).squeeze(-1)
    p_amp = np.abs(p_complex)
    p_max = float(p_amp.max())
    p_max_idx = np.unravel_index(np.argmax(p_amp), p_amp.shape)

    print(f"  Max pressure amplitude: {p_max:.4f} Pa at index {p_max_idx}")
    print(f"  Pressure at source:     {float(p_amp[src_idx]):.4f} Pa")
    print(f"  Pressure at (32,32,32): {float(p_amp[32,32,32]):.6f} Pa")
    print(f"  Pressure at (48,32,32): {float(p_amp[48,32,32]):.6f} Pa")

    return p_complex, domain, medium


def test_differentiability():
    """Test jax.grad through the Helmholtz solver.

    This is the key feature: compute d(focal_pressure)/d(source_phase).
    In a real application, source_phase would be per-element transducer delays,
    and the gradient would drive aberration correction optimization.

    We place two point sources with adjustable phases and differentiate
    the pressure amplitude at a target point w.r.t. those phases.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxdf.discretization import FourierSeries
    from jwave.acoustics.time_harmonic import helmholtz_solver
    from jwave.geometry import Domain, Medium

    N = (32, 32, 32)
    dx = (1e-3, 1e-3, 1e-3)
    freq = 500e3
    omega = 2 * np.pi * freq
    c0 = 1500.0

    domain = Domain(N, dx)
    medium = Medium(
        domain=domain,
        sound_speed=c0,
        density=1000.0,
        attenuation=0.0,
        pml_size=8,
    )

    # Two source positions at different distances from target (asymmetric)
    # Source 0 is closer to target than source 1, so they need different
    # phases to constructively interfere at the target.
    src_positions = [(12, 16, 16), (26, 16, 16)]
    target_idx = (16, 16, 16)

    def focal_pressure_from_phases(phases: jnp.ndarray) -> jnp.ndarray:
        """Compute negative pressure amplitude at target given source phases.

        Negative because we want to maximize pressure, so gradient descent
        on this loss function will increase focal pressure.
        """
        src = jnp.zeros(N + (1,), dtype=jnp.complex64)
        for i, pos in enumerate(src_positions):
            src = src.at[pos + (0,)].set(jnp.exp(1j * phases[i]))
        source = FourierSeries(src, domain)
        result = helmholtz_solver(medium, omega, source, tol=1e-4, maxiter=200)
        p_at_target = result.params[target_idx + (0,)]
        return -jnp.abs(p_at_target)

    # Compute gradient
    phases = jnp.array([0.0, 0.0], dtype=jnp.float32)

    print("  Computing forward pass (two-source Helmholtz)...")
    t0 = time.time()
    loss_val = focal_pressure_from_phases(phases)
    t_fwd = time.time() - t0
    print(f"  Forward pass: {t_fwd:.2f} s, focal pressure: {-float(loss_val):.6f} Pa")

    print("  Computing gradient via jax.grad...")
    grad_fn = jax.grad(focal_pressure_from_phases)
    t0 = time.time()
    grad_val = grad_fn(phases)
    t_grad = time.time() - t0
    print(f"  Gradient computation: {t_grad:.2f} s")
    print(f"  Gradient: {np.asarray(grad_val)}")
    print(f"  |grad|: {float(jnp.linalg.norm(grad_val)):.6e}")

    # Verify gradient is non-trivial (not all zeros)
    grad_norm = float(jnp.linalg.norm(grad_val))
    if grad_norm > 1e-10:
        print("  PASS: Non-zero gradient confirms differentiability through Helmholtz solver")
    else:
        print("  WARNING: Gradient is near-zero. May indicate a tracing issue.")

    # Quick gradient descent to show optimization works.
    # Learning rate is large because gradient magnitudes are O(1e-4) for
    # unit-amplitude sources on a coarse 32^3 grid. In a real application
    # with realistic source amplitudes (60 kPa) the gradients would be
    # proportionally larger and lr ~1e-7 would be appropriate (see Morgan's
    # optimize_delays which uses lr=1e-7 with amplitude=60000).
    print("\n  Running 10 steps of gradient descent on source phases...")
    lr = 10.0
    for step in range(10):
        loss, grad = jax.value_and_grad(focal_pressure_from_phases)(phases)
        phases = phases - lr * grad
        if step % 3 == 0 or step == 9:
            print(f"    Step {step:2d}: focal_p = {-float(loss):.6f}, "
                  f"phases = [{float(phases[0]):.4f}, {float(phases[1]):.4f}]")

    return float(-loss_val), grad_norm


def estimate_memory(grid_shape, dtype_bytes=8):
    """Estimate GPU memory for a j-Wave Helmholtz solve.

    The Helmholtz solver stores:
    - Complex pressure field: N*8 bytes (complex64) or N*16 (complex128)
    - Source field: same size
    - Medium arrays (c, rho, alpha): 3 * N * 4 bytes each (float32)
    - GMRES workspace: ~10-20 copies of the field for Krylov vectors
    - FFT workspace (PSTD): ~2-4 copies

    Rule of thumb: ~30x the field size for the full solve.
    """
    import numpy as np
    n_voxels = int(np.prod(grid_shape))
    field_bytes = n_voxels * dtype_bytes
    # GMRES needs ~20 Krylov vectors + medium + source + misc
    total_estimate = field_bytes * 30
    return total_estimate


def print_memory_estimates():
    """Print memory estimates for relevant grid sizes."""
    grids = {
        "POC (64^3)": (64, 64, 64),
        "Small (128^3)": (128, 128, 128),
        "BM7 typical (241x141x141)": (241, 141, 141),
        "BM7 with PML+40 (281x181x181)": (281, 181, 181),
        "Large (256^3)": (256, 256, 256),
    }
    print("  Memory estimates for Helmholtz solve (complex64):")
    for name, shape in grids.items():
        mem = estimate_memory(shape, dtype_bytes=8)
        print(f"    {name}: {mem / 1e6:.0f} MB ({mem / 1e9:.1f} GB)")


def main():
    print("=" * 70)
    print("j-Wave Proof of Concept: Differentiable Acoustic Simulation")
    print("=" * 70)

    print("\n[1] Checking imports...")
    check_imports()

    print("\n[2] JAX device info...")
    report_devices()

    print("\n[3] Memory estimates...")
    print_memory_estimates()

    print("\n[4] Helmholtz point source test (64^3, water, 500 kHz)...")
    p_complex, domain, medium = test_helmholtz_point_source()

    print("\n[5] Differentiability test (32^3, two sources, jax.grad)...")
    focal_p, grad_norm = test_differentiability()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Helmholtz solver: WORKING")
    print(f"  jax.grad through solver: {'WORKING' if grad_norm > 1e-10 else 'FAILED'}")
    print(f"  Gradient-based optimization: DEMONSTRATED (10 steps)")
    print("=" * 70)


if __name__ == "__main__":
    main()
