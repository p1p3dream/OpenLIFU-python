#!/usr/bin/env bash
# Batch j-Wave Helmholtz comparison for all 64 subjects (single-class labels).
#
# Skips subjects already in the output CSV.
# Uses nnunet single-class labels for consistency with k-Wave reference.
#
# Usage (on stonkbot):
#   source ~/envs/jwave-absorption/bin/activate
#   bash scripts/batch_jwave_all.sh

set -uo pipefail

SUBJECTS=(
    GU002 GU006 GU008 GU009 GU010 GU011 GU012 GU015 GU016 GU017
    GU018 GU019 GU020 GU021 GU024 GU025 GU026 GU027 GU029 GU030
    GU032 GU033 GU035 GU036 GU038 GU039 GU040 GU041
    NC004 NC008 NC010 NC011 NC012 NC013 NC014 NC015 NC016 NC017
    NC018 NC019 NC020 NC021 NC022 NC023 NC024 NC025 NC027 NC029
    NC031 NC033
    NYU002 NYU003 NYU004 NYU005 NYU006 NYU007 NYU008 NYU009 NYU010 NYU011
    subj1 subj2 subj3 subj4
)

RESULTS_DIR="${HOME}/Data/openlifu-validation/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_PATH="${RESULTS_DIR}/jwave_helmholtz_all.csv"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --csv) CSV_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "j-Wave Helmholtz Batch (All Subjects)"
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

    # Use single-class nnunet labels
    label_path="${RESULTS_DIR}/${subj}_nnunet_labels.nii.gz"
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
