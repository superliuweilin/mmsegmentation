_base_ = [
    '../_base_/models/cmt.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/work_dirs/mmsegmentation/cmt_ti_loveda_512x512_80k'
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
        type='CMT',
        img_size=1024,
        in_chans=3,
        num_classes=7,
        embed_dims=[46, 92, 184, 368],
        stem_channel=7,
        fc_dim=368,
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[3.6, 3.6, 3.6, 3.6],
        qkv_bias=True, qk_scale=None,
        representation_size=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        depths=[2, 2, 10, 2],
        qk_ratio=1,
        sr_ratios=[8, 4, 2, 1],
        dp=0.1),
    decode_head=dict(
        type='FPNHead',
        in_channels=[46, 92, 184, 368],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
