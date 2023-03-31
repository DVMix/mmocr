# toy_det_data_root = 'tests/data/det_toy_dataset'
picasso_textdet_data_root = '/home/home/PycharmProjects/work/ocr_toolbox/tools/torch/mm_ocr/data'

picasso_textdet_train = dict(
    type='OCRDataset',
    data_root=picasso_textdet_data_root,
    ann_file='dataset/textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

picasso_textdet_test = dict(
    type='OCRDataset',
    data_root=picasso_textdet_data_root,
    ann_file='dataset/textdet_test.json',
    test_mode=True,
    pipeline=None)

