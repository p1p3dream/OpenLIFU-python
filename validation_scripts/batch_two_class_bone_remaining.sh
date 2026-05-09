#!/usr/bin/env bash
# Batch two-class bone simulation for all remaining subjects (49 of 60).
# The original 11 outlier subjects were run previously.
#
# Usage:
#   nohup bash scripts/batch_two_class_bone_remaining.sh > log 2>&1 &

set -uo pipefail

# k-Wave CUDA binary needs libsz.so.2 from the virtualenv lib dir
export LD_LIBRARY_PATH="${HOME}/openlifu-env/lib:${LD_LIBRARY_PATH:-}"

SUBJECTS=(
    GU002 GU008 GU009 GU010 GU011 GU012 GU015 GU016 GU017 GU018
    GU019 GU020 GU021 GU024 GU025 GU026 GU027 GU029 GU030 GU032
    GU033 GU036 GU038 GU039 GU040 GU041
    NC004 NC008 NC010 NC013 NC014 NC016 NC018 NC019 NC020 NC021
    NC022 NC023 NC025 NC027 NC031
    NYU002 NYU003 NYU004 NYU006 NYU007 NYU009 NYU010 NYU011
)

RESULTS_DIR="${HOME}/Data/openlifu-validation/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_TAG="two_class_bone"

echo "========================================"
echo "Two-Class Bone Batch Simulation (Remaining 49)"
echo "Started: $(date)"
echo "========================================"
echo "Subjects:         ${#SUBJECTS[@]}"
echo "Results dir:      ${RESULTS_DIR}"
echo "Output tag:       ${OUTPUT_TAG}"
echo "LD_LIBRARY_PATH:  ${LD_LIBRARY_PATH}"
echo "========================================"

COMPLETED=0
SKIPPED=0
FAILED=0
TOTAL=${#SUBJECTS[@]}

for subj in "${SUBJECTS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] Subject: ${subj} ($(( COMPLETED + SKIPPED + FAILED + 1 ))/${TOTAL})"
    echo "----------------------------------------"

    # Skip if results already exist (defensive check)
    if ls "${RESULTS_DIR}/${subj}_two_class_bonegladys_nnunet_"*_pmax.nii.gz 1>/dev/null 2>&1; then
        echo "  SKIP: two-class bone results already exist for ${subj}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Check that two-class labels exist
    label_path="${RESULTS_DIR}/${subj}_two_class_labels.nii.gz"
    if [ ! -f "$label_path" ]; then
        echo "  ERROR: two-class labels not found: $label_path"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "  Running simulation..."
    OUTPUT_TAG="${OUTPUT_TAG}" python -u "${SCRIPT_DIR}/run_gladys_nnunet_subject.py" \
        --subject "$subj" \
        --label-path "$label_path" \
        --output-tag "$OUTPUT_TAG" \
        --bone-model two_class

    if [ $? -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "  [$(date +%H:%M:%S)] SUCCESS: ${subj} (${COMPLETED} completed so far)"
    else
        FAILED=$((FAILED + 1))
        echo "  [$(date +%H:%M:%S)] FAILED: ${subj}"
    fi
done

echo ""
echo "========================================"
echo "BATCH COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Skipped:   ${SKIPPED}/${TOTAL}"
echo "Failed:    ${FAILED}/${TOTAL}"
echo "========================================"
