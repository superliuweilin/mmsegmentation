norm_cfg = dict(type='SyncBN', requires_grad=True)
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
    pretrained=False,
    backbone=dict(
        type='MobileViT',
        image_size=512,
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 128, 160, 640],
        num_classes=16),
    decode_head=dict(
        type='',
        encoder_channels=(64, 96, 128, 640),
        decode_channels=64,
        dropout=0.1,
        window_size=7,
        num_classes=16,
        resolutions=[128, 64, 32, 16],
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
