#!/usr/bin/env bash

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
DATASET=$4
WORK_DIR=$5
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py ../configs/$DATASET/$CONFIG.py \
    --work-dir=$WORK_DIR/$DATASET/$CONFIG/ \
    --launcher="slurm" \
    ${PY_ARGS}