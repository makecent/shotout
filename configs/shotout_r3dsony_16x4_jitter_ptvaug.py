custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d_sony',
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_rgb.pth'),
        modality='rgb'
    ),
    cls_head=dict(type='I3DHead',
                  loss_cls=dict(type='CrossEntropyLoss'),
                  num_classes=3,
                  in_channels=1024),
    test_cfg=dict(average_clips='score')
)

clip_len = 16
frame_interval = 4
dataset_type = 'Shotout'

data_root = 'my_data/dfl-bundesliga-data-shootout/train_imgs'
# data_val = 'data/thumos14/rawframes/test'
ann_file_train = 'my_data/dfl-bundesliga-data-shootout/train.csv'
ann_file_val = 'my_data/dfl-bundesliga-data-shootout/val.csv'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='ShotoutSampleFrames', clip_len=clip_len, frame_interval=frame_interval, jitter_magnitude=10),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='pytorchvideo.RandAugment', magnitude=7, num_layers=4, prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
]
val_pipeline = [
    dict(type='ShotoutSampleFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
]
test_pipeline = [
    dict(type='ShotoutSampleFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        # sample_by_class=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root)
)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.01, by_epoch=False)
total_epochs = 50

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=100,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
