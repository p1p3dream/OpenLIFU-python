#!/usr/bin/env bash
# Sweep cortical thickness parameter for two-class bone simulations.
#
# Tests 3 representative subjects at 4 cortical thickness values
# (skipping 2.5mm which was already run in the full batch).
#
# For each (subject, thickness) pair:
#   1. Generate two-class label NIfTI with the given cortical thickness
#   2. Run GLADYS simulation with --bone-model two_class
#
# Total: 3 subjects x 4 thicknesses = 12 simulation runs (~5 hours GPU time)
#
# Usage:
#   bash scripts/sweep_cortical_thickness.sh
#   bash scripts/sweep_cortical_thickness.sh --dry-run

set -uo pipefail

SUBJECTS=(NC024 GU035 NC012)
THICKNESSES=(1.5 2.0 3.0 3.5)

RESULTS_DIR="${HOME}/Data/openlifu-validation/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TOTAL=$(( ${#SUBJECTS[@]} * ${#THICKNESSES[@]} ))
echo "========================================================"
echo "Cortical Thickness Sweep"
echo "========================================================"
echo "Subjects:    ${SUBJECTS[*]}"
echo "Thicknesses: ${THICKNESSES[*]} mm"
echo "Total runs:  ${TOTAL}"
echo "Results dir: ${RESULTS_DIR}"
echo "Dry run:     ${DRY_RUN}"
echo "Started:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# Pre-check: all single-class labels must exist
echo ""
echo "[Pre-check] Verifying source label NIfTIs..."
MISSING=0
for subj in "${SUBJECTS[@]}"; do
    label_path="${RESULTS_DIR}/${subj}_nnunet_labels.nii.gz"
    if [ ! -f "$label_path" ]; then
        echo "  MISSING: $label_path"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo "ERROR: ${MISSING} label NIfTIs missing. Cannot proceed."
    exit 1
fi
echo "  All ${#SUBJECTS[@]} source labels found."

COMPLETED=0
FAILED=0
RUN_NUM=0
SWEEP_START=$(date +%s)

for ct in "${THICKNESSES[@]}"; do
    for subj in "${SUBJECTS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        TAG="two_class_ct${ct}_"
        LABEL_OUT="${RESULTS_DIR}/${subj}_two_class_ct${ct}_labels.nii.gz"
        LOG_FILE="${RESULTS_DIR}/sweep_ct${ct}_${subj}.log"

        echo ""
        echo "========================================================"
        echo "[${RUN_NUM}/${TOTAL}] subject=${subj}  cortical_thickness=${ct}mm"
        echo "  tag:   ${TAG}"
        echo "  label: ${LABEL_OUT}"
        echo "  log:   ${LOG_FILE}"
        echo "  time:  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================"

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  [DRY RUN] Would generate labels and run simulation."
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        # Step 1: Generate two-class labels for this thickness
        INPUT_LABEL="${RESULTS_DIR}/${subj}_nnunet_labels.nii.gz"
        if [ -f "$LABEL_OUT" ]; then
            echo "  Labels already exist: ${LABEL_OUT}, skipping generation."
        else
            echo "  Generating two-class labels (cortical_thickness=${ct}mm)..."
            python "${SCRIPT_DIR}/split_skull_two_class.py" \
                --input "$INPUT_LABEL" \
                --output "$LABEL_OUT" \
                --cortical-thickness "$ct"
            if [ $? -ne 0 ]; then
                echo "  FAILED: label generation for ${subj} at ct=${ct}mm"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi

        # Step 2: Run simulation
        echo "  Running simulation..."
        RUN_START=$(date +%s)

        python "${SCRIPT_DIR}/run_gladys_nnunet_subject.py" \
            --subject "$subj" \
            --label-path "$LABEL_OUT" \
            --output-tag "${TAG}" \
            --bone-model two_class \
            2>&1 | tee "$LOG_FILE"

        SIM_EXIT="${PIPESTATUS[0]}"
        RUN_END=$(date +%s)
        RUN_ELAPSED=$(( RUN_END - RUN_START ))
        RUN_MIN=$(( RUN_ELAPSED / 60 ))

        if [ "$SIM_EXIT" -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
            echo "  DONE: ${subj} ct=${ct}mm in ${RUN_MIN}min (${RUN_ELAPSED}s)"
        else
            FAILED=$((FAILED + 1))
            echo "  FAILED: ${subj} ct=${ct}mm (exit code ${SIM_EXIT})"
        fi

        # Progress estimate
        REMAINING=$(( TOTAL - RUN_NUM ))
        if [ "$COMPLETED" -gt 0 ]; then
            SWEEP_ELAPSED=$(( RUN_END - SWEEP_START ))
            AVG_SEC=$(( SWEEP_ELAPSED / RUN_NUM ))
            ETA_SEC=$(( AVG_SEC * REMAINING ))
            ETA_MIN=$(( ETA_SEC / 60 ))
            echo "  Progress: ${RUN_NUM}/${TOTAL} done, ~${ETA_MIN}min remaining"
        fi
    done
done

SWEEP_END=$(date +%s)
SWEEP_TOTAL=$(( SWEEP_END - SWEEP_START ))
SWEEP_HOURS=$(( SWEEP_TOTAL / 3600 ))
SWEEP_MINS=$(( (SWEEP_TOTAL % 3600) / 60 ))

echo ""
echo "========================================================"
echo "SWEEP COMPLETE"
echo "========================================================"
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Failed:    ${FAILED}/${TOTAL}"
echo "Wall time: ${SWEEP_HOURS}h ${SWEEP_MINS}m (${SWEEP_TOTAL}s)"
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Log files: ${RESULTS_DIR}/sweep_ct*_*.log"
echo ""
echo "To extract SUBJECT_SUMMARY lines:"
echo "  grep SUBJECT_SUMMARY ${RESULTS_DIR}/sweep_ct*.log"
echo "========================================================"
