#!/bin/bash

checkpoint=$1
ckpt_path="$(dirname $checkpoint)"
epoch_name=$"(basename $checkpoint)"

root=..
data_dir=$root/datasets/culane/
echo $(ls $root)
detect_dir=${ckpt_path}/result/


# evaluate each scenario separately
# These can not be changed
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list0=${data_dir}list/test_split/test0_normal.txt
list1=${data_dir}list/test_split/test1_crowd.txt
list2=${data_dir}list/test_split/test2_hlight.txt
list3=${data_dir}list/test_split/test3_shadow.txt
list4=${data_dir}list/test_split/test4_noline.txt
list5=${data_dir}list/test_split/test5_arrow.txt
list6=${data_dir}list/test_split/test6_curve.txt
list7=${data_dir}list/test_split/test7_cross.txt
list8=${data_dir}list/test_split/test8_night.txt
out0=${ckpt_path}/out0_normal.txt
out1=${ckpt_path}/out1_crowd.txt
out2=${ckpt_path}/out2_hlight.txt
out3=${ckpt_path}/out3_shadow.txt
out4=${ckpt_path}/out4_noline.txt
out5=${ckpt_path}/out5_arrow.txt
out6=${ckpt_path}/out6_curve.txt
out7=${ckpt_path}/out7_cross.txt
out8=${ckpt_path}/out8_night.txt
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list0 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out0
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list1 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out1
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list2 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out2
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list3 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out3
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list4 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out4
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list5 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out5
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list6 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out6
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list7 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out7
./ganet/culane/lane_evaluation/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list8 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out8
cat ${ckpt_path}/out*.txt>${checkpoint}_iou${iou}_split.txt
