custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)

model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet3d', depth=50, pretrained=None),
    cls_head=dict(type='I3DHead',
                  loss_cls=dict(type='FocalLoss'),
                  num_classes=3,
                  in_channels=2048)
)
load_from = "https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth"

clip_len = 32
frame_interval = 1
dataset_type = 'Shotout'
data_root = 'data/dfl-bundesliga-data-shootout'
data_train = 'data/dfl-bundesliga-data-shootout/train'
# data_val = 'data/thumos14/rawframes/test'
ann_file_train = 'data/dfl-bundesliga-data-shootout/train.csv'
# ann_file_val = 'data/thumos14/annotations/apn/apn_test.csv'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='ShotoutSampleFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='pytorchvideo.RandAugment', magnitude=7, num_layers=4, prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
    # dict(type='RandomErasing')
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_train,
        raw_video=True)
)
# evaluation = dict(metrics=['top_k_accuracy'])
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.01,
    warmup_iters=10,
    warmup_by_epoch=True)
total_epochs = 100

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
