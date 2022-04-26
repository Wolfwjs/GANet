#!/usr/bin/env bash

checkpoint=$1
ckpt_path="$(dirname $checkpoint)"

root=../../../
data_dir=$root/datasets/culane/
detect_dir=${ckpt_path}/result/

# evaluate the whole
# These can not be changed
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list=${data_dir}list/test.txt
out=${ckpt_path}/iou_${iou}_all.txt
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out
