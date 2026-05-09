#!/usr/bin/env bash
set -euo pipefail

# Full pipeline: nnU-Net segment -> optimize placement -> CW sim
# Usage: bash scripts/batch_pipeline.sh GU024 GU030 GU035 ...

REPO=/home/brandon/OpenLIFU-python
DATA=~/Data/openlifu-validation
MRI_DIR="$DATA/datasets/birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI"
RESULTS_DIR="$DATA/results"
NNUNET_ENV=~/openlifu-env/bin
NNUNET_RESULTS=~/nnUNet_results
TMPBASE=/mnt/data/tmp/brandon
LOGBASE=/mnt/data/tmp/brandon

export PYTHONPATH=$REPO/src
export LD_LIBRARY_PATH=~/openlifu-env/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
PYTHON=$NNUNET_ENV/python3

SUBJECTS=("$@")
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Usage: $0 SUBJ1 SUBJ2 ..."
    exit 1
fi

echo "==============================================="
echo "Batch pipeline: ${#SUBJECTS[@]} subjects"
echo "Subjects: ${SUBJECTS[*]}"
echo "==============================================="

for SUBJ in "${SUBJECTS[@]}"; do
    echo ""
    echo "========================================"
    echo "[$SUBJ] Starting pipeline $(date '+%H:%M:%S')"
    echo "========================================"

    MRI_PATH="$MRI_DIR/${SUBJ}_deface.nii"
    LABEL_PATH="$RESULTS_DIR/${SUBJ}_nnunet_labels.nii.gz"

    if [ ! -f "$MRI_PATH" ]; then
        echo "[$SUBJ] ERROR: MRI not found: $MRI_PATH"
        continue
    fi

    # Step 1: nnU-Net segmentation (if labels don't exist)
    if [ -f "$LABEL_PATH" ]; then
        echo "[$SUBJ] Labels exist, skipping segmentation"
    else
        echo "[$SUBJ] Running nnU-Net segmentation..."
        NNUNET_TMP=$(mktemp -d "$TMPBASE/nnunet_${SUBJ}_XXXXXX")
        NNUNET_INPUT="$NNUNET_TMP/input"
        NNUNET_OUTPUT="$NNUNET_TMP/output"
        mkdir -p "$NNUNET_INPUT" "$NNUNET_OUTPUT"

        # nnU-Net expects _0000.nii.gz suffix
        cp "$MRI_PATH" "$NNUNET_INPUT/${SUBJ}_0000.nii"
        gzip "$NNUNET_INPUT/${SUBJ}_0000.nii"

        nnUNet_results="$NNUNET_RESULTS" $NNUNET_ENV/nnUNetv2_predict \
            -i "$NNUNET_INPUT" \
            -o "$NNUNET_OUTPUT" \
            -d 002 -c 3d_fullres -f 0 \
            --disable_tta 2>&1 | tail -5

        # Copy result
        PRED="$NNUNET_OUTPUT/${SUBJ}.nii.gz"
        if [ -f "$PRED" ]; then
            cp "$PRED" "$LABEL_PATH"
            echo "[$SUBJ] Labels saved: $LABEL_PATH"
        else
            echo "[$SUBJ] ERROR: nnU-Net produced no output"
            ls "$NNUNET_OUTPUT/"
            continue
        fi
        rm -rf "$NNUNET_TMP"
    fi

    # Step 2: Optimize placement (CPU, fast)
    echo "[$SUBJ] Optimizing placement..."
    ORIENT=$($PYTHON $REPO/scripts/optimize_placement.py \
        --subject "$SUBJ" --n-search 200 --label-path "$LABEL_PATH" 2>&1 \
        | grep "^BEST:" | head -1)

    if [ -z "$ORIENT" ]; then
        echo "[$SUBJ] ERROR: placement optimizer produced no BEST: line"
        # Fall back to default orientation
        THETA=""
        PHI=""
    else
        THETA=$(echo "$ORIENT" | sed 's/.*theta=\([0-9.]*\).*/\1/')
        PHI=$(echo "$ORIENT" | sed 's/.*phi=\([0-9.]*\).*/\1/')
        echo "[$SUBJ] Best orientation: theta=$THETA phi=$PHI"
    fi

    # Step 3: CW simulation
    echo "[$SUBJ] Running CW simulation..."
    SIM_TMP="$TMPBASE/optimized_${SUBJ}"
    mkdir -p "$SIM_TMP"

    ORIENT_ARGS=""
    if [ -n "$THETA" ] && [ -n "$PHI" ]; then
        ORIENT_ARGS="--orient-theta $THETA --orient-phi $PHI"
    fi

    TMPDIR="$SIM_TMP" \
    DELAY_METHOD=complex_weighted \
    CW_NORM=sum \
    OUTPUT_TAG=optimized_ \
    $PYTHON $REPO/scripts/run_gladys_nnunet_subject.py \
        --subject "$SUBJ" \
        --label-path "$LABEL_PATH" \
        $ORIENT_ARGS \
        --output-tag optimized_ \
        2>&1 | tee "$LOGBASE/optimized_${SUBJ}.log" | tail -20

    echo "[$SUBJ] Done $(date '+%H:%M:%S')"
    echo ""
done

echo "==============================================="
echo "Batch complete $(date '+%H:%M:%S')"
echo "==============================================="
