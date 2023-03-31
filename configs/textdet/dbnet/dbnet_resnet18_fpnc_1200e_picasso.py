_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/picasso.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
picasso_textdet_train = _base_.picasso_textdet_train
picasso_textdet_train.pipeline = _base_.train_pipeline
picasso_textdet_test = _base_.picasso_textdet_test
picasso_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=picasso_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=picasso_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
