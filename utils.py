import os
import shutil
from collections import OrderedDict

import torch
import numpy as np
import cv2
from PIL import Image
import h5py


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

# def load_data(img_path, train=True, mode="once"):
def load_data(img_path, target_path):
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

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')

    # resize image to 640*360 as previous work
    prev_img = prev_img.resize((640, 360))
    img = img.resize((640, 360))
    post_img = post_img.resize((640, 360))

    target = np.load(target_path)['x']
    target = cv2.resize(target,
                        (int(target.shape[1] / 8),
                         int(target.shape[0] / 8)),
                        interpolation=cv2.INTER_CUBIC) * 64

    return prev_img, img, post_img, target

def load_ucsd_data(img_path, target_path):
    img_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)

    file_name = img_name.split('.')[-2]
    scene_num = file_name.split('_')[2]
    index = int(file_name.split('_')[-1][1:])  # ..._fxxx.png -> int(xxx)

    prev_index = int(max(1, index - 5))
    post_index = int(min(200, index + 5))

    prev_img_path = os.path.join(img_dir, 'vidf1_33_{}_f{:0=3}.png'.format(scene_num, prev_index))
    post_img_path = os.path.join(img_dir, 'vidf1_33_{}_f{:0=3}.png'.format(scene_num, post_index))
    # print(prev_img_path, img_path, post_img_path)

    prev_img = Image.open(prev_img_path).convert('L')
    img = Image.open(img_path).convert('L')
    post_img = Image.open(post_img_path).convert('L')

    target = np.load(target_path)['x']
    target = torch.tensor(target).float()

    return prev_img, img, post_img, target
