_base_ = [
    '../_base_/datasets/nus-3d_radar.py',
    '../_base_/default_runtime.py'
]

point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

voxel_size = [0.1, 0.1, 0.2]

# SSL Specific configuration
ssl_config = dict(
    radar_augmentations=dict(
        noise_std=0.02,
        dropout_prob=0.1,
    ),
    camera_augmentations=dict(
        jitter=0.05,
        brightness=0.1,
        contrast=0.1,
        crop_size=(0.9, 0.9)
    ),
    contrastive_loss=dict(
        type='InfoNCELoss',
        temperature=0.07
    ),
    temporal_ssl=dict(
        sequence_length=3,
        embedding_dim=128,
        transformer_heads=8,
        transformer_layers=4
    ),
    training=dict(
        batch_size=8,
        epochs=30,
        learning_rate=1e-4,
        optimizer='AdamW',
        lr_schedule='cosine'
    )
)

model = dict(
    type='SSLFusionRCBEVDet',
    radar_backbone=dict(
        type='RadarBEVNet',
        radar_encoder=dict(
            dual_stream=True,
            rcs_aware=True
        )
    ),
    camera_backbone=dict(
        type='ResNet',
        depth=50,
        pretrained='torchvision://resnet50',
        out_indices=(2, 3),
        frozen_stages=-1
    ),
    fusion_module=dict(
        type='CAMF',
        deformable_attention=True,
        num_attention_layers=2
    ),
    ssl_module=dict(
        contrastive_projection_dim=128,
        temporal_prediction=True
    ),
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadRadarPoints'),
    dict(type='RadarAugmentation', noise_std=0.02, dropout_prob=0.1),
    dict(type='CameraAugmentation', jitter=0.05, brightness=0.1, contrast=0.1),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'radar']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='SSLNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names
    ),
    val=dict(
        type='SSLNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names
    ),
    test=dict(
        type='SSLNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names
    )
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-5, by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=30)

checkpoint_config = dict(interval=5, max_keep_ckpts=5)
evaluation = dict(interval=5, metric='mAP')