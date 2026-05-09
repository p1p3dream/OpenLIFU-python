#!/usr/bin/env bash
# Launch 256-element GLADYS simulation on stonkbot (RTX 4090).
#
# Usage:
#   bash scripts/launch_256element_stonkbot.sh                  # run GU008 only
#   bash scripts/launch_256element_stonkbot.sh --subject GU010  # run one subject
#   bash scripts/launch_256element_stonkbot.sh --all            # run all 4 subjects sequentially
#   bash scripts/launch_256element_stonkbot.sh --dry-run        # print commands, don't execute
#   bash scripts/launch_256element_stonkbot.sh --sync-only      # rsync only, don't run sims
set -euo pipefail

# ---------------------------------------------------------------------------
# Stonkbot connection
# ---------------------------------------------------------------------------
STONKBOT_IP="192.168.68.71"
STONKBOT_USER="brandon"
STONKBOT_SSH="${STONKBOT_USER}@${STONKBOT_IP}"

# ---------------------------------------------------------------------------
# Remote paths
# ---------------------------------------------------------------------------
REMOTE_REPO="/home/brandon/OpenLIFU-python"
REMOTE_DATA="/home/brandon/Data/openlifu-validation"
REMOTE_TMPBASE="/mnt/data/tmp/brandon"
GPU_LOCK="/tmp/stonkbot_gladys.lock"

# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------
ALL_SUBJECTS=(GU008 GU002 GU010 NC004)
DEFAULT_SUBJECT="GU008"

# ---------------------------------------------------------------------------
# Local paths (relative to repo root)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SUBJECT=""
RUN_ALL=0
DRY_RUN=0
SYNC_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)
            SUBJECT="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --sync-only)
            SYNC_ONLY=1
            shift
            ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--subject SUBJ] [--all] [--dry-run] [--sync-only]" >&2
            exit 1
            ;;
    esac
done

# Resolve which subjects to run
if [[ ${RUN_ALL} -eq 1 ]]; then
    SUBJECTS=("${ALL_SUBJECTS[@]}")
elif [[ -n "${SUBJECT}" ]]; then
    SUBJECTS=("${SUBJECT}")
else
    SUBJECTS=("${DEFAULT_SUBJECT}")
fi

# ---------------------------------------------------------------------------
# Helper: run or print a command
# ---------------------------------------------------------------------------
run_cmd() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

ssh_cmd() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[dry-run] ssh ${STONKBOT_SSH} $*"
    else
        ssh "${STONKBOT_SSH}" "$@"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: rsync scripts/ and src/ to stonkbot
# ---------------------------------------------------------------------------
echo "=== Syncing scripts/ and src/ to stonkbot ==="
run_cmd rsync -avz --delete \
    "${REPO_ROOT}/scripts/" \
    "${STONKBOT_SSH}:${REMOTE_REPO}/scripts/"

run_cmd rsync -avz --delete \
    "${REPO_ROOT}/src/" \
    "${STONKBOT_SSH}:${REMOTE_REPO}/src/"

echo "=== Sync complete ==="

if [[ ${SYNC_ONLY} -eq 1 ]]; then
    echo "Sync-only mode; exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: For each subject, launch simulation on stonkbot
# ---------------------------------------------------------------------------
for SUBJ in "${SUBJECTS[@]}"; do
    TMPDIR_REMOTE="${REMOTE_TMPBASE}/256elem_${SUBJ}"
    LOG_FILE="${REMOTE_TMPBASE}/256elem_${SUBJ}.log"
    MRI_PATH="${REMOTE_DATA}/datasets/birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI/${SUBJ}_deface.nii"
    LABEL_PATH="${REMOTE_DATA}/results/${SUBJ}_nnunet_labels.nii.gz"

    echo ""
    echo "=== Subject: ${SUBJ} ==="
    echo "  TMPDIR:    ${TMPDIR_REMOTE}"
    echo "  Log:       ${LOG_FILE}"
    echo "  MRI:       ${MRI_PATH}"
    echo "  Labels:    ${LABEL_PATH}"

    # Create TMPDIR on stonkbot
    ssh_cmd "mkdir -p '${TMPDIR_REMOTE}'"

    # Build the remote command. When running all subjects sequentially, wrap
    # the python invocation in a flock so only one GPU sim runs at a time.
    PYTHON_CMD="cd '${REMOTE_REPO}' && \
TMPDIR='${TMPDIR_REMOTE}' \
DELAY_METHOD=complex_weighted \
CW_NORM=sum \
EXPANDED_TARGET_PROBE=1 \
EXPANDED_TARGET_HALF_MM=10.0 \
CUBE_PROBE_STRIDE=2 \
OUTPUT_TAG=256elem_ \
python3 scripts/run_gladys_256element.py \
    --subject '${SUBJ}' \
    --mri-path '${MRI_PATH}' \
    --label-path '${LABEL_PATH}'"

    if [[ ${RUN_ALL} -eq 1 ]]; then
        # Sequential with GPU flock: each subject waits for the lock
        REMOTE_CMD="flock '${GPU_LOCK}' bash -c '${PYTHON_CMD}'"
    else
        REMOTE_CMD="${PYTHON_CMD}"
    fi

    # Launch via nohup in background on stonkbot
    LAUNCH_CMD="nohup bash -c '${REMOTE_CMD}' > '${LOG_FILE}' 2>&1 & echo \$!"

    echo "  Launching..."
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[dry-run] ssh ${STONKBOT_SSH} \"${LAUNCH_CMD}\""
    else
        PID=$(ssh "${STONKBOT_SSH}" "${LAUNCH_CMD}")
        echo "  PID:       ${PID}"
        echo "  Log:       ${LOG_FILE}"
        echo "  Monitor:   ssh ${STONKBOT_SSH} tail -f '${LOG_FILE}'"
    fi
done

echo ""
echo "=== All launches complete ==="
echo "To check running sims:  ssh ${STONKBOT_SSH} 'ps aux | grep run_gladys_256element'"
echo "To watch a log:         ssh ${STONKBOT_SSH} tail -f ${REMOTE_TMPBASE}/256elem_<SUBJECT>.log"
