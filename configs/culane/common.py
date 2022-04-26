"""
    config file of the small version of GANet for tusimple
"""
total_epochs = 60 # check
optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8) # check
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) # check
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5) # check
checkpoint_config = dict(interval=2) # check
workflow = [('train', 1000)] # check
