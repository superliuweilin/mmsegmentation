_base_ = [
    '../_base_/models/pvtv2_b2.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/lwl_work_dirs/pvtv2_b2_loveda_512x512_80k'
