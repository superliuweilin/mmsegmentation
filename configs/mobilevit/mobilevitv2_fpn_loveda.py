_base_ = [
    '../_base_/models/mobilevitv2_csanet_isaid.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# crop_size=(140, 140)
work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/lwl_work_dirs/mobilevit_fpn_512x512_loveda_80k'