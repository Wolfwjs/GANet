#!/usr/bin/env bash
# export NCCL_IB_DISABLE=1
CONFIG=$1 # projects/config/tucurve/final_exp_res18_s8.py
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29510}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PY_ARGS=${@:2}

root=$(dirname $0)/.. # tools

PYTHONPATH=$root:$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $root/tools/train.py \
    $CONFIG \
    --launcher="pytorch" \
    ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/dist_train.sh projects/cfgs/curvelane/final_exp_res18_s8.py
# nccl先单个后并行