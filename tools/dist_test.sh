#!/usr/bin/env bash

DATASET=$1
CONFIG=$2
CHECKPOINT=$3
GPUS=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/ganet/$DATASET/test_dataset.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:5}

# CUDA_VISIBLE_DEVICES=1 bash tools/dist_test.sh tusimple configs/tusimple/final_exp_res101_s4.py work_dirs/tusimple/large/epoch_250.pth 1
# python tools/ganet/tusimple/evaluate/lane.py work_dirs/tusimple/results/test.json /data1/hrz/datasets/tusimple/test_baseline.json
# output json-like string in terminal

# CUDA_VISIBLE_DEVICES=1 bash tools/dist_test.sh culane configs/culane/final_exp_res101_s4.py work_dirs/culane/large/epoch_60.pth 1
# bash tools/ganet/culane/evaluate/eval_all.sh
# `python tools/ganet/analyse_culane.py` to analyse metric 