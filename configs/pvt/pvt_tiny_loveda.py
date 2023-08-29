_base_ = [
    '../_base_/models/pvt_tiny.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# crop_size=(140, 140)
work_dir = '/home/lyu/lwl_wsp/mmsegmentation/work_dirs/mmsegmentation/pvt_tiny_loveda_512x512_80k'
