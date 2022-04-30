#!/usr/bin/env bash

DATASET=$1
CONFIG=$2
CHECKPOINT=$3
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/ganet/$DATASET/test_dataset.py \
    ../configs/$DATASET/$CONFIG.py \
    $CHECKPOINT \
    --launcher="pytorch" \
    ${PY_ARGS}
