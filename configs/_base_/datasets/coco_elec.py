_base_ = 'coco_instance.py'

dataset_type = 'CocoDataset'
data_root = 'data/electronic/'

metainfo = {
    'classes': ('component',), 
    'palette': [
        (225, 0, 0), 
    ]
}

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train')
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/')
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json'
)
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test.json'
)