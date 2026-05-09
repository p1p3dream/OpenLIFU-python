#!/usr/bin/env python3
"""Export a trained nnU-Net model to ONNX format.

Reads the nnU-Net plans.json and dataset.json to auto-detect architecture
parameters, reconstructs the PlainConvUNet, loads trained weights, disables
deep supervision, and exports to ONNX with validation.

Prerequisites (install into your nnU-Net environment):
    pip install onnx onnxruntime

Usage:
    python export_onnx.py --dataset 1 --fold 0 --output skull_seg.onnx
    python export_onnx.py --dataset 2 --fold 0 --output fullhead_seg.onnx

Environment variables:
    nnUNet_results  Path to nnU-Net results directory (or use --nnunet-results)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def find_results_dir(dataset_id: int, nnunet_results: str) -> Path:
    """Locate the nnU-Net results directory for a given dataset ID.

    Scans the results root for a directory matching the pattern
    DatasetNNN_* where NNN is the zero-padded dataset ID.
    """
    results_root = Path(nnunet_results)
    if not results_root.is_dir():
        sys.exit(f"nnUNet_results directory not found: {results_root}")

    prefix = f"Dataset{dataset_id:03d}_"
    matches = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(prefix)]

    if len(matches) == 0:
        sys.exit(f"No dataset directory found matching {prefix}* in {results_root}")
    if len(matches) > 1:
        names = [m.name for m in matches]
        sys.exit(f"Multiple dataset directories found for {prefix}*: {names}")

    return matches[0]


def load_json(path: Path) -> dict:
    """Load a JSON file and return its contents as a dict."""
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_network(plans: dict, dataset_json: dict, config_name: str) -> torch.nn.Module:
    """Reconstruct the nnU-Net network from plans with deep supervision disabled.

    Uses the official nnunetv2 utility to resolve class names, import
    operators, and instantiate the architecture. Deep supervision is
    explicitly disabled so the model returns a single output tensor
    rather than a list of multi-scale predictions.
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    if config_name not in plans["configurations"]:
        available = list(plans["configurations"].keys())
        sys.exit(f"Configuration '{config_name}' not found in plans. Available: {available}")

    config = plans["configurations"][config_name]
    arch = config["architecture"]

    num_input_channels = len(dataset_json["channel_names"])
    num_output_classes = len(dataset_json["labels"])

    print(f"Architecture: {arch['network_class_name']}")
    print(f"Input channels: {num_input_channels}")
    print(f"Output classes: {num_output_classes}")
    print(f"Patch size: {config['patch_size']}")
    print(f"Features per stage: {arch['arch_kwargs']['features_per_stage']}")
    print(f"Strides: {arch['arch_kwargs']['strides']}")

    network = get_network_from_plans(
        arch_class_name=arch["network_class_name"],
        arch_kwargs=arch["arch_kwargs"],
        arch_kwargs_req_import=arch["_kw_requires_import"],
        input_channels=num_input_channels,
        output_channels=num_output_classes,
        allow_init=False,  # We will load trained weights, skip random init
        deep_supervision=False,
    )

    return network


def load_weights(network: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load trained weights from an nnU-Net checkpoint into the network.

    The checkpoint stores weights under the 'network_weights' key.
    Weights are loaded with strict=True to catch any architecture mismatch.
    """
    if not checkpoint_path.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "network_weights" not in checkpoint:
        sys.exit(f"Checkpoint does not contain 'network_weights'. Keys: {list(checkpoint.keys())}")

    network.load_state_dict(checkpoint["network_weights"])
    print(f"Loaded weights successfully ({len(checkpoint['network_weights'])} parameter tensors)")


def export_to_onnx(
    network: torch.nn.Module,
    patch_size: list[int],
    num_input_channels: int,
    output_path: Path,
) -> None:
    """Export the network to ONNX format.

    Creates a dummy input matching the training patch size, traces the
    model with torch.onnx.export, and writes the ONNX file. Uses opset 17
    for broad compatibility with InstanceNorm and ConvTranspose3d operators.
    The batch dimension is marked as dynamic.
    """
    network.eval()

    # Dummy input: batch=1, channels=num_input_channels, spatial=patch_size
    dummy_input = torch.randn(1, num_input_channels, *patch_size)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Verify forward pass works before export
    with torch.no_grad():
        test_output = network(dummy_input)
    print(f"Forward pass output shape: {test_output.shape}")

    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        network,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")


def validate_onnx(
    network: torch.nn.Module,
    onnx_path: Path,
    patch_size: list[int],
    num_input_channels: int,
    atol: float = 5e-3,
    rtol: float = 1e-3,
) -> None:
    """Validate the ONNX model against the PyTorch model.

    Runs both models on the same random input and checks that outputs
    match within tolerance. Also runs the ONNX checker for structural
    validity.

    Note on tolerances: InstanceNorm3d computes per-instance statistics
    from the input tensor. Float32 accumulation order differs between
    PyTorch and ONNX Runtime, producing small numerical differences
    (typically max ~1e-3). The default tolerances account for this.
    """
    try:
        import onnx
    except ImportError:
        print("WARNING: 'onnx' package not installed. Skipping ONNX structural check.")
        print("  Install with: pip install onnx")
        onnx = None

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("'onnxruntime' package is required for validation. Install with: pip install onnxruntime")

    # Structural validation
    if onnx is not None:
        print("Running ONNX structural check...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX structural check passed")

    # Numerical validation
    print("Running numerical validation (PyTorch vs ONNX Runtime)...")

    test_input = torch.randn(1, num_input_channels, *patch_size)
    test_input_np = test_input.numpy()

    # PyTorch reference output
    network.eval()
    with torch.no_grad():
        pytorch_output = network(test_input).numpy()

    # ONNX Runtime output
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"input": test_input_np})[0]

    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  ORT output shape:     {ort_output.shape}")
    print(f"  PyTorch output range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"  ORT output range:     [{ort_output.min():.6f}, {ort_output.max():.6f}]")

    max_abs_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_abs_diff = np.mean(np.abs(pytorch_output - ort_output))
    print(f"  Max absolute diff:    {max_abs_diff:.2e}")
    print(f"  Mean absolute diff:   {mean_abs_diff:.2e}")

    if np.allclose(pytorch_output, ort_output, atol=atol, rtol=rtol):
        print(f"  PASSED: Outputs match within atol={atol}, rtol={rtol}")
    else:
        mismatched = ~np.isclose(pytorch_output, ort_output, atol=atol, rtol=rtol)
        n_bad = int(mismatched.sum())
        pct = 100.0 * n_bad / mismatched.size
        print(f"  FAILED: {n_bad}/{mismatched.size} elements ({pct:.4f}%) exceed tolerance")
        print(f"  Tolerances: atol={atol}, rtol={rtol}")
        print("  This may be acceptable for inference; consider relaxing tolerances.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained nnU-Net model to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_onnx.py --dataset 1 --fold 0 --output skull_seg.onnx\n"
            "  python export_onnx.py --dataset 2 --fold 0 --output fullhead_seg.onnx\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=int,
        required=True,
        help="nnU-Net dataset ID (e.g., 1 for Dataset001_SkullSeg)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Training fold to export (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the ONNX file (e.g., skull_seg.onnx)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="3d_fullres",
        help="nnU-Net configuration name (default: 3d_fullres)",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name (default: nnUNetTrainer)",
    )
    parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetPlans",
        help="Plans name (default: nnUNetPlans)",
    )
    parser.add_argument(
        "--nnunet-results",
        type=str,
        default=None,
        help="Path to nnUNet_results directory (default: $nnUNet_results env var)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint filename (default: checkpoint_final.pth)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX validation step",
    )
    args = parser.parse_args()

    # Resolve nnUNet_results directory
    nnunet_results = args.nnunet_results or os.environ.get("nnUNet_results")
    if nnunet_results is None:
        sys.exit(
            "nnUNet_results not specified. Set the nnUNet_results environment variable "
            "or pass --nnunet-results."
        )

    # Locate dataset results directory
    dataset_dir = find_results_dir(args.dataset, nnunet_results)
    print(f"Dataset directory: {dataset_dir}")

    # Build the trainer subdirectory path
    trainer_subdir = f"{args.trainer}__{args.plans}__{args.config}"
    trainer_dir = dataset_dir / trainer_subdir
    if not trainer_dir.is_dir():
        sys.exit(f"Trainer directory not found: {trainer_dir}")

    # Load plans and dataset JSON
    plans = load_json(trainer_dir / "plans.json")
    dataset_json = load_json(trainer_dir / "dataset.json")
    print(f"Dataset: {plans['dataset_name']}")
    print(f"Labels: {dataset_json['labels']}")

    # Locate checkpoint
    fold_dir = trainer_dir / f"fold_{args.fold}"
    checkpoint_path = fold_dir / args.checkpoint
    if not checkpoint_path.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    # Build network with deep supervision disabled
    print("\n--- Building network ---")
    network = build_network(plans, dataset_json, args.config)

    # Load trained weights
    print("\n--- Loading weights ---")
    load_weights(network, checkpoint_path)

    # Export to ONNX
    config = plans["configurations"][args.config]
    patch_size = config["patch_size"]
    num_input_channels = len(dataset_json["channel_names"])

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Exporting to ONNX ---")
    export_to_onnx(network, patch_size, num_input_channels, output_path)

    # Validate
    if not args.skip_validation:
        print("\n--- Validating ONNX model ---")
        validate_onnx(network, output_path, patch_size, num_input_channels)
    else:
        print("\nSkipping validation (--skip-validation)")

    print(f"\nDone. ONNX model written to: {output_path}")


if __name__ == "__main__":
    main()
