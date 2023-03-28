from mmocr.apis import MMOCRInferencer
import argparse
import os
from pathlib import Path
import struct
import sys
import cv2
import glob

IEND_CHUNK = bytes.fromhex('0000000049454E44AE426082')
IEND_CHUNK_LENGTH = len(IEND_CHUNK)
ROOT = Path(__file__).parents[2]


def _repair(file_path):
    def _no_iend(f):
        f.seek(IEND_CHUNK_LENGTH, os.SEEK_END)
        data = f.read(IEND_CHUNK_LENGTH)
        return data != IEND_CHUNK

    def _append_iend(f):
        f.seek(0, os.SEEK_END)
        f.write(IEND_CHUNK)

    with open(file_path, 'r+b') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(8, os.SEEK_SET)  # skip PNG magic

        complete_chunks = False
        while True:
            buf = f.read(4)  # read chunk data length
            if len(buf) == 0:
                complete_chunks = True
                break

            length = struct.unpack('>I', buf)[0]
            f.seek(4, os.SEEK_CUR)  # skip chunk name
            data_and_crc = length + 4
            data_left = file_size - f.tell()
            if data_and_crc > data_left:
                break

            f.seek(data_and_crc, os.SEEK_CUR)  # skip chunk data and CRC

        if complete_chunks:
            if _no_iend(f):
                _append_iend(f)
        else:
            print(f'Unable to repair {file_path}')


def _process(arg, recursive):
    path = Path(arg).resolve()
    if path.is_dir():
        for p in sorted(os.listdir(path)):
            sub = path / Path(p)
            if sub.is_dir():
                if recursive:
                    _process(sub, recursive)
            else:
                _repair(sub)
    elif path.is_file():
        _repair(path)


def repair_images(path):
    parser = argparse.ArgumentParser(
        description='Fixes abruptly truncated PNG files by adding IEND chunk at the end if necessary'
    )
    # parser.add_argument('path', help='path to a PNG file or a directory containing PNG files', nargs='+')
    parser.add_argument(
        '-r', '--recursive',
        help='recursively process all the subdirectories of the given directory',
        action='store_true'
    )
    args = parser.parse_args()

    _process(path, args.recursive)


def resize_image(file, size, style='folder'):
    assert style in ['file', 'folder'], 'Fix save style^ should be in [file, folder]'
    image = cv2.imread(file)
    coefficient = round(max(image.shape[:2]) / size, 4)
    dim = (int(image.shape[1] / coefficient), int(image.shape[0] / coefficient))
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    save_path = f'{Path(file).parent}/{Path(file).stem}_resized{max_size}{Path(file).suffix}'
    cv2.imwrite(save_path, resized_image)


def resize_to(path_template, size, style='folder'):
    assert style in ['file', 'folder'], 'Fix save style^ should be in [file, folder]'
    for file in glob.glob(path_template):
        resize_image(file, size, style)


class InferenceBox:
    def __init__(self, det='DBNet', rec='CRNN'):
        self.ocr = MMOCRInferencer(det=det, rec=rec)

    def __call__(self, image_path, show=True, print_result=True):
        self.ocr(inputs=image_path, show=show, print_result=print_result)


if __name__ == '__main__':
    target_folder = ROOT / 'data/original_data'
    # repair_images(target_folder)
    max_size = 900
    path_template = target_folder / '*.png'
    resize_to(path_template=path_template, size=max_size)


