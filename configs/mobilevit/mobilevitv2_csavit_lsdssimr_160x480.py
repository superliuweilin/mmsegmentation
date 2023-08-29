_base_ = [
    '../_base_/models/mobilevitv2_csanet.py', '../_base_/datasets/lsdssimr.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_csavit_isaid.py'
]
# crop_size=(140, 140)
work_dir = '/home/lyu/lwl_wsp/mmsegmentation/work_dirs/mmsegmentation/csavit_lsdssimr_160x480'
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    # size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MobileViT',
        image_size=(160, 480),
        input_channel=6,
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 128, 160, 640],
        num_classes=2),
    decode_head=dict(
        type='CSAHead',
        in_channels=(64, 96, 128, 640),
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout=0.1,
        window_size=7,
        num_classes=2,
        resolutions=[(40, 120), (40, 120), (20, 60), (10, 30)],
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=20000,
#         by_epoch=False,
#     )
# ]

train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)