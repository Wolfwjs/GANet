#!/usr/bin/env bash

CONFIG=$1 # projects/config/tucurve/final_exp_res18_s8.py
CHECKPOINT=$2 # epoch_19.pth
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-49828}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PY_ARGS=${@:3}

root=$(dirname $0)/.. # tools

PYTHONPATH=$root:$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $root/tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher="pytorch" \
    ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/dist_test.sh projects/cfgs/culane/final_exp_res101_s4.py pr_models/ganet_culane_resnet101.pth --eval (--show-dir ./show)
# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/dist_test.sh projects/cfgs/culane/final_exp_res101_s4.py work_dirs/culane/large/epoch_80.pth --eval (--show-dir ./show)
# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/dist_test.sh /data1/hrz/myGANet/projects/cfgs/curvelane/final_exp_res18_s8.py /data1/hrz/myGANet/work_dirs/curvelane/small/best_F1_epoch_60.pth --eval