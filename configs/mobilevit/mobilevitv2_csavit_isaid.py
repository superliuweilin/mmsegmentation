_base_ = [
    '../_base_/models/mobilevitv2_csanet.py', '../_base_/datasets/lsdssimr.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_csavit_isaid.py'
]
work_dir = '/home/lyu/lwl_wsp/mmsegmentation/work_dirs/mmsegmentation/csavit_lsdssimr_160x480'
