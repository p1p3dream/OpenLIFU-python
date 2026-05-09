#!/usr/bin/env bash
# Batch j-Wave Helmholtz comparison for 11 outlier subjects.
#
# Runs 4 Helmholtz solves per subject (~10 min each on RTX 4090):
#   1. Water baseline
#   2. Skull geometric
#   3. Reciprocity phase correction
#   4. Skull phase-corrected
#
# Results go to a shared CSV for cross-subject analysis.
#
# Usage (on stonkbot):
#   source ~/envs/jwave-absorption/bin/activate
#   bash scripts/batch_jwave_helmholtz.sh
#   bash scripts/batch_jwave_helmholtz.sh --dry-run

set -uo pipefail

SUBJECTS=(
    NC024   # 13.55 dB (k-Wave geo)
    NC011   # 15.83 dB
    NC015   # 16.01 dB
    NYU005  # 17.52 dB
    NYU008  # 18.06 dB
    GU006   # 18.24 dB
    NC017   # 18.21 dB
    NC012   # 19.00 dB
    NC029   # 20.71 dB
    NC033   # 20.55 dB
    GU035   # 24.66 dB
)

RESULTS_DIR="${HOME}/Data/openlifu-validation/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_PATH="${RESULTS_DIR}/jwave_helmholtz_batch.csv"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --csv) CSV_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "j-Wave Helmholtz Batch Comparison"
echo "========================================"
echo "Subjects:    ${#SUBJECTS[@]}"
echo "CSV output:  ${CSV_PATH}"
echo "Results dir: ${RESULTS_DIR}"
echo "Dry run:     ${DRY_RUN}"
echo "========================================"

COMPLETED=0
FAILED=0
SKIPPED=0
TOTAL=${#SUBJECTS[@]}
START_TIME=$(date +%s)

for subj in "${SUBJECTS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Subject: ${subj} ($((COMPLETED + FAILED + SKIPPED + 1))/${TOTAL})"
    echo "----------------------------------------"

    label_path="${RESULTS_DIR}/${subj}_two_class_labels.nii.gz"
    if [ ! -f "$label_path" ]; then
        echo "  MISSING: $label_path"
        FAILED=$((FAILED + 1))
        continue
    fi

    if [ -f "$CSV_PATH" ] && grep -q "^${subj}," "$CSV_PATH" 2>/dev/null; then
        echo "  Already in CSV, skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] Would run Helmholtz for ${subj}"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    LOG_PATH="${RESULTS_DIR}/jwave_helmholtz_${subj}.log"
    echo "  Log: ${LOG_PATH}"

    python3 -u "${SCRIPT_DIR}/jwave_kwave_compare.py" \
        --subject "$subj" \
        --label-path "$label_path" \
        --csv "$CSV_PATH" \
        2>&1 | tee "$LOG_PATH"

    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  FAILED: ${subj}"
        FAILED=$((FAILED + 1))
    fi

    ELAPSED=$(( $(date +%s) - START_TIME ))
    DONE=$((COMPLETED + FAILED + SKIPPED))
    if [ "$DONE" -gt 0 ] && [ "$DONE" -lt "$TOTAL" ]; then
        REMAINING=$(( TOTAL - DONE ))
        PER_SUBJ=$(( ELAPSED / DONE ))
        ETA=$(( PER_SUBJ * REMAINING ))
        echo "  Elapsed: $((ELAPSED / 60))m, ETA: $((ETA / 60))m remaining"
    fi
done

TOTAL_TIME=$(( $(date +%s) - START_TIME ))

echo ""
echo "========================================"
echo "BATCH COMPLETE"
echo "========================================"
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Skipped:   ${SKIPPED}/${TOTAL}"
echo "Failed:    ${FAILED}/${TOTAL}"
echo "Total time: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo "CSV: ${CSV_PATH}"

if [ -f "$CSV_PATH" ]; then
    echo ""
    echo "Summary:"
    echo "--------------------------------------------------------------"
    column -t -s',' "$CSV_PATH"
    echo "--------------------------------------------------------------"
fi
echo "========================================"
