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
        type='FlattenPvt',
        img_size=1024,
        num_classes=7,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]),
    decode_head=dict(
        type='CSAHead',
        in_channels=(64, 128, 320, 512),
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout=0.1,
        window_size=7,
        num_classes=2,
        resolutions=[(256, 256), (128, 128), (64, 64), (32, 32)],
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
