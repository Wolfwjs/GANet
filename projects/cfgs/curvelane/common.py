"""
    config file of the small version of GANet for curvelane
"""
dataset_type = 'CurvelaneDataset'
data_root = "/data2/hrz/datasets/curvelane" # 在各个cfg文件里面添加子文件夹/train或者/valid
train_data_list = [data_root + '/train/train.txt']
val_data_list = [data_root + '/valid/valid.txt']
test_data_list = val_data_list

num_lane_classes = 1
max_lane_num = 14 # 这个只用于创建align的车道target(用于deform点的监督),只要超过最大的那个就行,不影响表现
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
# ori_scale = (1640, 590)
# crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)

optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8) 
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) 
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr=1e-5)
workflow = [('train', 1000)] 
runner = dict(type='EpochBasedRunner', max_epochs=250)
dist_params = dict(backend='nccl')

find_unused_parameters=True

checkpoint_config = dict(interval=10)
evaluation = dict(interval=10,save_best="F1",greater_keys=['F1'])
log_level = 'INFO'
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# 数据增强设置
train_compose = dict(bboxes=False, keypoints=True, masks=False)

train_al_pipeline = [
    dict(type='Compose', params=train_compose),
    # dict(
    #     type='Crop',
    #     x_min=crop_bbox[0],
    #     x_max=crop_bbox[2],
    #     y_min=crop_bbox[1],
    #     y_max=crop_bbox[3],
    #     p=1),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict( 
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0),
        ],
        p=0.7),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        type='RandomResizedCrop',
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type='Compose', params=train_compose),
    # dict(
    #     type='Crop',
    #     x_min=crop_bbox[0],
    #     x_max=crop_bbox[2],
    #     y_min=crop_bbox[1],
    #     y_max=crop_bbox[3],
    #     p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]