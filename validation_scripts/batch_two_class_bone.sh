#!/usr/bin/env bash
# Batch two-class bone simulation for 11 outlier subjects (>15 dB attenuation).
#
# Steps per subject:
#   1. Generate two-class label NIfTI from nnU-Net single-class labels
#   2. Run GLADYS simulation with --bone-model two_class
#
# Prerequisites:
#   - nnU-Net labels exist: ~/Data/openlifu-validation/results/<subj>_nnunet_labels.nii.gz
#   - OpenLIFU installed in editable mode with two-class bone support
#   - GPU available (k-Wave CUDA)
#
# Usage:
#   bash scripts/batch_two_class_bone.sh
#   bash scripts/batch_two_class_bone.sh --cortical-thickness 3.0
#   bash scripts/batch_two_class_bone.sh --dry-run

set -uo pipefail

SUBJECTS=(
    NC024   # 16.28 dB
    NC011   # 16.37 dB
    GU006   # 16.53 dB
    NC015   # 16.69 dB
    GU035   # 17.42 dB
    NYU005  # 17.44 dB
    NC017   # 18.18 dB
    NC033   # 18.19 dB
    NC029   # 18.96 dB
    NC012   # 20.60 dB
    NYU008  # 20.77 dB
)

RESULTS_DIR="${HOME}/Data/openlifu-validation/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORTICAL_THICKNESS="2.5"
DRY_RUN=0
OUTPUT_TAG="two_class_bone"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --cortical-thickness) CORTICAL_THICKNESS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Two-Class Bone Batch Simulation"
echo "========================================"
echo "Subjects:           ${#SUBJECTS[@]}"
echo "Cortical thickness: ${CORTICAL_THICKNESS} mm"
echo "Results dir:        ${RESULTS_DIR}"
echo "Output tag:         ${OUTPUT_TAG}"
echo "Dry run:            ${DRY_RUN}"
echo "========================================"

# Step 1: Check that all single-class labels exist
echo ""
echo "[Step 1] Checking single-class label NIfTIs..."
MISSING=0
for subj in "${SUBJECTS[@]}"; do
    label_path="${RESULTS_DIR}/${subj}_nnunet_labels.nii.gz"
    if [ ! -f "$label_path" ]; then
        echo "  MISSING: $label_path"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo "ERROR: ${MISSING} label NIfTIs missing. Run nnU-Net segmentation first."
    echo "  Use: python scripts/run_gladys_nnunet_subject.py --subject <ID>"
    exit 1
fi
echo "  All ${#SUBJECTS[@]} label NIfTIs found."

# Step 2: Generate two-class labels
echo ""
echo "[Step 2] Generating two-class bone labels (cortical thickness=${CORTICAL_THICKNESS} mm)..."
for subj in "${SUBJECTS[@]}"; do
    input="${RESULTS_DIR}/${subj}_nnunet_labels.nii.gz"
    output="${RESULTS_DIR}/${subj}_two_class_labels.nii.gz"
    if [ -f "$output" ]; then
        echo "  ${subj}: two-class labels already exist, skipping."
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] Would generate: $output"
        continue
    fi
    python "${SCRIPT_DIR}/split_skull_two_class.py" \
        --input "$input" \
        --output "$output" \
        --cortical-thickness "$CORTICAL_THICKNESS"
    echo ""
done

# Step 3: Run simulations
echo ""
echo "[Step 3] Running GLADYS simulations with two-class bone model..."
COMPLETED=0
FAILED=0
TOTAL=${#SUBJECTS[@]}

for subj in "${SUBJECTS[@]}"; do
    echo "----------------------------------------"
    echo "Subject: ${subj} ($(( COMPLETED + FAILED + 1 ))/${TOTAL})"
    echo "----------------------------------------"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] Would run simulation for ${subj}"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    label_path="${RESULTS_DIR}/${subj}_two_class_labels.nii.gz"
    if [ ! -f "$label_path" ]; then
        echo "  ERROR: two-class labels not found: $label_path"
        FAILED=$((FAILED + 1))
        continue
    fi

    OUTPUT_TAG="${OUTPUT_TAG}" python "${SCRIPT_DIR}/run_gladys_nnunet_subject.py" \
        --subject "$subj" \
        --label-path "$label_path" \
        --output-tag "$OUTPUT_TAG" \
        --bone-model two_class \
        2>&1 | tee "${RESULTS_DIR}/${OUTPUT_TAG}_${subj}.log"

    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  FAILED: ${subj}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "BATCH COMPLETE"
echo "========================================"
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Failed:    ${FAILED}/${TOTAL}"
echo "Results:   ${RESULTS_DIR}/${OUTPUT_TAG}_*.log"
echo "========================================"
