"""GPU coordination via filesystem lock (granular, around k-wave calls only).

Usage:
    from _gpu_flock import gpu_flock
    with gpu_flock():
        result = run_simulation(...)

The context manager acquires an exclusive flock on /tmp/stonkbot_gpu.lock
for the duration of the `with` block. This is meant to wrap ONLY the k-wave
CUDA binary invocation (direct kspaceFirstOrder3D calls, run_simulation,
and the k-wave-based delay/apodization methods) so that CPU-bound Python
setup and analysis can run in parallel across multiple agent invocations.

A single 4090 (24 GB VRAM) can only run one ~21 GB kspaceFirstOrder-CUDA
sim at a time, hence the serialization. But everything else (MRI loading,
segmentation resampling, sim_params construction, probe analysis, sidecar
writing) should overlap freely.

Migration note: /tmp/stonkbot_gpu.lock is a DIFFERENT path from the legacy
/tmp/stonkbot_gladys.lock used by outer-script flock wrappers, so there is
no deadlock risk if some agent prompts still wrap the whole Python run in
outer flock while others use only this inner lock.
"""
import contextlib
import fcntl
import os

GPU_LOCK_PATH = "/tmp/stonkbot_gpu.lock"


@contextlib.contextmanager
def gpu_flock(lock_path: str = GPU_LOCK_PATH):
    """Acquire an exclusive flock around GPU-using code.

    Ensures k-wave CUDA binary invocations do not overlap, since each uses
    ~21 GB VRAM on a 24 GB card. CPU-bound Python setup and analysis runs
    outside the lock so multiple agents can overlap there.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
