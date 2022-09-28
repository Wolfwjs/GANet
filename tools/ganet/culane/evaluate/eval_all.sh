#!/bin/bash

out_path=tools/ganet/culane_out
gt_dir=/data2/hrz/datasets/culane/
pred_dir=/data1/hrz/GANet/work_dirs/culane/results/

rm -rf $out_path
mkdir $out_path
# evaluate each scenario separately
# These can not be changed
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list0=${gt_dir}list/test_split/test0_normal.txt
list1=${gt_dir}list/test_split/test1_crowd.txt
list2=${gt_dir}list/test_split/test2_hlight.txt
list3=${gt_dir}list/test_split/test3_shadow.txt
list4=${gt_dir}list/test_split/test4_noline.txt
list5=${gt_dir}list/test_split/test5_arrow.txt
list6=${gt_dir}list/test_split/test6_curve.txt
list7=${gt_dir}list/test_split/test7_cross.txt
list8=${gt_dir}list/test_split/test8_night.txt
list9=${gt_dir}list/test.txt
out0=${out_path}/out0_normal.txt
out1=${out_path}/out1_crowd.txt
out2=${out_path}/out2_hlight.txt
out3=${out_path}/out3_shadow.txt
out4=${out_path}/out4_noline.txt
out5=${out_path}/out5_arrow.txt
out6=${out_path}/out6_curve.txt
out7=${out_path}/out7_cross.txt
out8=${out_path}/out8_night.txt
out9=${out_path}/out9_total.txt
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list0 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out0
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list1 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out1
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list2 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out2
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list3 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out3
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list4 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out4
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list5 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out5
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list6 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out6
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list7 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out7
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list8 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out8
/data1/hrz/GANet/tools/ganet/culane/evaluate/evaluate -a $gt_dir -d $pred_dir -i $gt_dir -l $list9 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out9

# bash tools/ganet/culane/evaluate/eval_all.sh