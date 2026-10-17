#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -e job-%j.err
#SBATCH -p a800
#SBATCH -J mace_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -D ./
#SBATCH --get-user-env

set -eo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 N_JOBS [TARGET_PATH] [TARGET_FOLDER]" >&2
    exit 2
fi

N_JOBS="$1"
TARGET_PATH="${2:-.}"
TARGET_FOLDER="${3:-structures}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

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

echo "========================================================="
echo "SLURM environment:"
echo "Job name                     : ${SLURM_JOB_NAME:-N/A}"
echo "Job ID                       : ${SLURM_JOB_ID:-N/A}"
echo "Job user                     : ${SLURM_JOB_USER:-N/A}"
echo "Job array index              : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Submission directory         : ${SLURM_SUBMIT_DIR:-N/A}"
echo "Temporary directory          : ${TMPDIR:-N/A}"
echo "Submission host              : ${SLURM_SUBMIT_HOST:-N/A}"
echo "Queue/partition              : ${SLURM_JOB_PARTITION:-N/A}"
echo "Node list                    : ${SLURM_JOB_NODELIST:-N/A}"
echo "First-node hostname          : ${HOSTNAME:-N/A}"
echo "Allocated nodes              : ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-N/A}}"
echo "Allocated tasks              : ${SLURM_NTASKS:-N/A}"
echo "Tasks per node               : ${SLURM_TASKS_PER_NODE:-N/A}"
echo "Requested tasks per node     : ${SLURM_NTASKS_PER_NODE:-N/A}"
echo "Requested CPUs per task      : ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Scheduling priority          : ${SLURM_PRIO_PROCESS:-N/A}"
echo "SLURM-assigned GPU IDs       : ${SLURM_JOB_GPUS:-N/A}"
echo "CUDA_VISIBLE_DEVICES         : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo

echo "========================================================="
echo "SLURM job submitted at: $(date)"
START_TIME="$(date +%s)"

echo "========================================================="
echo "Job output begins"
echo "---------------------------------------------------------"
echo "Parallel workers             : $N_JOBS"
echo "Target path                  : $TARGET_PATH"
echo "Target CIF folder            : $TARGET_CIF_DIR"
echo

"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/mace_opt_slurm.py" \
    --path "$TARGET_PATH" \
    --target_folder "$TARGET_CIF_DIR" \
    --n_jobs "$N_JOBS"

echo
echo "---------------------------------------------------------"
echo "Job output ends"
END_TIME="$(date +%s)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "========================================================="
echo "SLURM job finished at: $(date)"
echo "Total run time: ${HOURS} hours ${MINUTES} minutes ${SECONDS} seconds"
echo "========================================================="
