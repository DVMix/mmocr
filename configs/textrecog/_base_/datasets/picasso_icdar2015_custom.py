picasso_icdar2015_textrecog_data_root = 'data/picasso'

picasso_icdar2015_textrecog_train = dict(
    type='OCRDataset',
    data_root=picasso_icdar2015_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

picasso_icdar2015_textrecog_test = dict(
    type='OCRDataset',
    data_root=picasso_icdar2015_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
