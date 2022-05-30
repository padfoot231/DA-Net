#!/usr/bin/env python3

import sys
import pathlib
import cv2

sys.path.append('..')

from utils import distort_image

IN_PATH = "./tiny-imagenet-200"
OUT_PATH = "./tiny-imagenet-200-fisheye"

F = 120
D = [0.5, 0.5, 0.5, 0.5]


def main():
    convert_dataset(IN_PATH, OUT_PATH)

def convert_dataset(in_dir, out_dir):
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)
    
    out_dir.mkdir(exist_ok=True)

    for fpath in in_dir.rglob('*'):
        if fpath.suffix == '.JPEG':
            # Compute and create paths
            relpath = fpath.relative_to(in_dir)
            outpath = out_dir.joinpath(relpath)
            outpath.parent.mkdir(parents=True, exist_ok=True)

            # print('Creating', outpath)

            # Distort image
            img = cv2.imread(str(fpath))
            distorted = distort_image(img, F, D)
            cv2.imwrite(str(outpath), distorted)


if __name__=='__main__':
    main()