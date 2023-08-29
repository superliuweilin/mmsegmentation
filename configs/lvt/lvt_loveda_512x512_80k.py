_base_ = [
    '../_base_/models/lvt.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# crop_size=(140, 140)

work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/work_dirs/mmsegmentation/lvt_loveda_512x512_80k'
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LVT',
        num_classes=7,
        layers=[2, 2, 2, 2],
        patch_size=4,
        embed_dims=[64, 64, 160, 256],
        num_heads=[2, 2, 5, 8],
        mlp_ratios=[4, 8, 4, 4],
        mlp_depconv=[False, True, True, True],
        sr_ratios=[8, 4, 2, 1],
        sa_layers=['csa', 'rasa', 'rasa', 'rasa']
        ),
    decode_head=dict(
        type='FPNHead',
        in_channels=[64, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

