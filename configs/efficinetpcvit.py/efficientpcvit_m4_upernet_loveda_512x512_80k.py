_base_ = [
    '../_base_/models/efficientpcvit_m4_upernet.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_efficientvit_loveda.py'
]
# crop_size=(140, 140)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='EfficientPcViT',
        frozen_stages=0, 
        distillation=False, 
        img_size=512,
        patch_size=16,
        embed_dim=[192, 288, 384],
        depth=[1, 2, 3],
        num_heads=[4, 4, 4],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 288, 384],
        in_index=[0, 1, 2],
        pool_scales=(1, 2, 3),
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/lwl_work_dirs/efficientvit_pconv_m4_upernet_scale3_loveda_512x512_80k'