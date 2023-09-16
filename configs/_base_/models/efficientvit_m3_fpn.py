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
        type='EfficientViT',
        frozen_stages=0, 
        distillation=False, 
        img_size=1024,
        patch_size=16,
        embed_dim=[128, 240, 320],
        depth=[1, 2, 3],
        num_heads=[4, 3, 4],
        window_size=[7, 7, 7],
        kernels=[5, 5, 5, 5]),
    decode_head=dict(
        type='FPNHead',
        in_channels=[128, 240, 320],
        in_index=[0, 1, 2],
        feature_strides=[8, 16, 32],
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
