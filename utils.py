import os
import shutil

import torch
import numpy as np
import cv2
from PIL import Image
import h5py


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def load_data(img_path, train=True, mode="once"):
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    if '(1)' in img_name:
        img_name = img_name.replace('(1)', '')
    try:
        index = int(img_name.split('.')[0])
    except ValueError:
        print(img_name)

    prev_index = int(max(1, index - 5))
    post_index = int(min(150, index + 5))

    prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))
    post_img_path = os.path.join(img_folder, '%03d.jpg' % (post_index))

    if mode == "once":
        gt_path = img_path.replace('.jpg', '_resize.h5')
    elif mode == "add":
        gt_path = img_path.replace('.jpg', '_resize_add.h5')
    else:
        raise ValueError
    gt_path = img_path.replace('.jpg', '_resize')

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')

    # resize image to 640*360 as previous work
    prev_img = prev_img.resize((640, 360))
    img = img.resize((640, 360))
    post_img = post_img.resize((640, 360))

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target = cv2.resize(target,
                        (int(target.shape[1] / 8),
                         int(target.shape[0] / 8)),
                        interpolation=cv2.INTER_CUBIC) * 64  # 64 allows max value to be 1.0

    return prev_img, img, post_img, target
