_base_ = [
    '../_base_/models/efficientvit_m3_fpn.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_efficientvit_loveda.py'
]
# crop_size=(140, 140)

work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/lwl_work_dirs/efficientvit_m3_fpn_loveda_512x512_80k'