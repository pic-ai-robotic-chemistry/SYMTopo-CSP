#!/bin/bash

set -eo pipefail

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -p, --target-path PATH       Root directory for this run (default: current directory)."
    echo "  -t, --target-folder FOLDER  CIF directory, relative to PATH or absolute (default: structures)."
    echo "  -g, --gpu-count COUNT       Number of SLURM jobs/GPUs to request (default: 8)."
    echo "  -n, --tasks-per-node COUNT  Number of MACE workers in each job (default: 10)."
    echo "  -h, --help                  Show this help message and exit."
    echo
    echo "Example:"
    echo "  $0 --target-path /fs0/home/user/CSP/run_01 --target-folder structures \\"
    echo "     --gpu-count 8 --tasks-per-node 10"
}

TARGET_PATH="${TARGET_PATH:-.}"
TARGET_FOLDER="${TARGET_FOLDER:-structures}"
GPU_COUNT="${GPU_COUNT:-8}"
TASKS_PER_NODE="${TASKS_PER_NODE:-10}"
JOB_NAME_BASE="${JOB_NAME_BASE:-opt}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--target-path)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
            TARGET_PATH="$2"
            shift 2
            ;;
        --target-path=*)
            TARGET_PATH="${1#*=}"
            shift
            ;;
        -t|--target-folder)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
            TARGET_FOLDER="$2"
            shift 2
            ;;
        --target-folder=*)
            TARGET_FOLDER="${1#*=}"
            shift
            ;;
        -g|--gpu-count)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
            GPU_COUNT="$2"
            shift 2
            ;;
        --gpu-count=*)
            GPU_COUNT="${1#*=}"
            shift
            ;;
        -n|--tasks-per-node)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
            TASKS_PER_NODE="$2"
            shift 2
            ;;
        --tasks-per-node=*)
            TASKS_PER_NODE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            if [[ $# -gt 0 ]]; then
                echo "Error: positional arguments are not supported: $*" >&2
                exit 2
            fi
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm.sh"
PYTHON_SCRIPT="$SCRIPT_DIR/mace_opt_slurm.py"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Error: SLURM script does not exist: $SLURM_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script does not exist: $PYTHON_SCRIPT" >&2
    exit 2
fi

for VALUE_NAME in GPU_COUNT TASKS_PER_NODE; do
    VALUE="${!VALUE_NAME}"
    if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $VALUE_NAME must be a positive integer, got: $VALUE" >&2
        exit 2
    fi
done

if [[ ! -d "$TARGET_PATH" ]]; then
    echo "Error: target path does not exist: $TARGET_PATH" >&2
    exit 2
fi
TARGET_PATH="$(cd -- "$TARGET_PATH" && pwd)"

if [[ "$TARGET_FOLDER" = /* ]]; then
    TARGET_CIF_DIR="$TARGET_FOLDER"
else
    TARGET_CIF_DIR="$TARGET_PATH/$TARGET_FOLDER"
fi

if [[ ! -d "$TARGET_CIF_DIR" ]]; then
    echo "Error: target CIF folder does not exist: $TARGET_CIF_DIR" >&2
    exit 2
fi
TARGET_CIF_DIR="$(cd -- "$TARGET_CIF_DIR" && pwd)"

LOG_DIR="${LOG_DIR:-$TARGET_PATH/logs}"
mkdir -p -- "$LOG_DIR"

echo "Submitting $GPU_COUNT jobs."
echo "Target path       : $TARGET_PATH"
echo "Target CIF folder : $TARGET_CIF_DIR"
echo "Log directory     : $LOG_DIR"
echo "GPU count         : $GPU_COUNT"
echo "Workers per job   : $TASKS_PER_NODE"
echo "Python script     : $PYTHON_SCRIPT"

for ((i = 1; i <= GPU_COUNT; i++)); do
    CURRENT_JOB_NAME="${JOB_NAME_BASE}_${i}"
    echo "Submitting job $i/$GPU_COUNT: name=$CURRENT_JOB_NAME, workers=$TASKS_PER_NODE"

    # Command-line options override the corresponding defaults in slurm.sh.
    # The four trailing values become positional arguments inside slurm.sh.
    sbatch \
        -J "$CURRENT_JOB_NAME" \
        --ntasks-per-node="$TASKS_PER_NODE" \
        -o "$LOG_DIR/${CURRENT_JOB_NAME}.out" \
        -e "$LOG_DIR/${CURRENT_JOB_NAME}.err" \
        "$SLURM_SCRIPT" \
        "$TASKS_PER_NODE" \
        "$TARGET_PATH" \
        "$TARGET_CIF_DIR" \
        "$PYTHON_SCRIPT"
done

echo "All jobs were submitted successfully."
