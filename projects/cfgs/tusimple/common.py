"""
    config file of the small version of GANet for tusimple
"""
dataset_type = 'TuSimpleDataset'
data_root = "/data2/hrz/datasets/tusimple"
train_data_list = [
            data_root + '/label_data_0313.json',
            data_root + '/label_data_0531.json',
            data_root + '/label_data_0601.json'
        ]
val_data_list = [data_root + '/test_label.json'] # test_baseline
test_data_list = [data_root + '/test_label.json']
num_lane_classes = 1
max_lane_num = 6 # 这个只用于创建align的车道target(用于deform点的监督)
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
ori_scale = (1280, 720)
crop_bbox = [0, 160, 1280, 720]
img_scale = (800, 320)

optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8) 
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) 
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10, # LR used at the beginning of warmup equals to warmup_ratio * initial_lr
    min_lr=1e-5)

workflow = [('train', 1000)] 
runner = dict(type='EpochBasedRunner', max_epochs=300)
dist_params = dict(backend='nccl')

find_unused_parameters=True

checkpoint_config = dict(interval=40)
evaluation = dict(interval=5,save_best="F1",greater_keys=['F1'])
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
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
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
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]