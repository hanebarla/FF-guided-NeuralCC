import json
import os
import random
from tkinter.tix import DirList
from matplotlib.pyplot import sci
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from utils import *
import torchvision.transforms.functional as F
from torchvision import transforms
import csv
import cv2
import scipy.io
from scipy.ndimage.filters import gaussian_filter


def dataset_factory(args, train_data, val_data):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    if args.dataset == "FDST":
        train_dataset = FDSTDataset(
            train_data,
            transform=transform
        )
        val_dataset = FDSTDataset(
            val_data,
            transform=transform
        )
    elif args.dataset == "CrowdFlow":
        train_dataset = CrowdFlowDataset(
            train_data,
            transform=transform
        )
        val_dataset = CrowdFlowDataset(
            val_data,
            transform=transform
        )
    elif args.dataset == "venice":
        train_dataset = VeniceDataset(
            train_data,
            transform=transform
        )
        val_dataset = VeniceDataset(
            val_data,
            transform=transform
        )
    elif args.dataset == "CityStreet":
        train_dataset = CityStreetDataset(
            train_data,
            data_type=args.data_mode,
            transform=transform
        )
        val_dataset = CityStreetDataset(
            val_data,
            data_type=args.data_mode,
            transform=transform
        )
    elif args.dataset == "ucsd":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = UCSDDataset(
            train_data,
            transform=transform
        )
        val_dataset = UCSDDataset(
            val_data,
            transform=transform
        )
    else:
        raise NotImplementedError("No such dataset {}".format(args.dataset))
    
    return train_dataset, val_dataset


class CrowdFlowDataset(Dataset):
    def __init__(self, dict, transform):
        self.dict = dict
        self.transform = transform

    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self, index):
        prev_path = self.dict[index]["prev"]
        now_path = self.dict[index]["now"]
        post_path = self.dict[index]["post"]
        target_path = self.dict[index]["target"]

        prev_img = Image.open(prev_path).convert('RGB')
        prev_img = prev_img.resize((640, 360))
        now_img = Image.open(now_path).convert('RGB')
        now_img = now_img.resize((640, 360))
        post_img = Image.open(post_path).convert('RGB')
        post_img = post_img.resize((640, 360))

        prev_img = self.transform(prev_img)
        now_img = self.transform(now_img)
        post_img = self.transform(post_img)

        target = np.load(target_path)["x"]
        target = torch.from_numpy(target.astype(np.float32)).clone()

        return prev_img, now_img, post_img, target


class FDSTDataset(Dataset):
    def __init__(self, data, transform=None):
        self.nSamples = len(data)
        self.lines = data
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]["img"]
        target_path = self.lines[index]["target"]

        prev_img, img, post_img, target = load_data(img_path, target_path)

        if self.transform is not None:
            prev_img = self.transform(prev_img)
            img = self.transform(img)
            post_img = self.transform(post_img)
        return prev_img, img, post_img, target

class CityStreetDataset(Dataset):
    def __init__(self, data_dir, data_type, transform=None, scene=None):
        super().__init__()

        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = None
        self.scene = scene

        self.v1_imgs, self.v1_targets = self._load_h5file("ff_view1.h5")
        self.v2_imgs, self.v2_targets = self._load_h5file("ff_view2.h5")
        self.v3_imgs, self.v3_targets = self._load_h5file("ff_view3.h5")

        # self.imgs = np.concatenate([v1_imgs, v2_imgs, v3_imgs])
        # self.targets = np.concatenate([v1_targets, v2_targets, v3_targets])

    def __len__(self):
        if self.scene is None:
            return (self.v1_imgs.shape[0] - 2) + (self.v2_imgs.shape[0] - 2) + (self.v3_imgs.shape[0] - 2)
        elif self.scene == "view1":
            return self.v1_imgs.shape[0] - 2
        elif self.scene == "view2":
            return self.v2_imgs.shape[0] - 2
        elif self.scene == "view3":
            return self.v3_imgs.shape[0] - 2
        else:
            raise ValueError

    def __getitem__(self, index: int):
        if self.scene is None:
            prev_img, img, post_img, target = self._get_whole_scene_item(index)
        elif self.scene == "view1":
            prev_img = self.v1_imgs[index]
            img = self.v1_imgs[index+1]
            post_img = self.v1_imgs[index+2]
            target = self.v1_targets[index]
        elif self.scene == "view2":
            prev_img = self.v2_imgs[index]
            img = self.v2_imgs[index+1]
            post_img = self.v2_imgs[index+2]
            target = self.v2_targets[index]
        elif self.scene == "view3":
            prev_img = self.v3_imgs[index]
            img = self.v3_imgs[index+1]
            post_img = self.v3_imgs[index+2]
            target = self.v3_targets[index]

        prev_img = torch.from_numpy(prev_img.astype(np.float32)).clone()
        img = torch.from_numpy(img.astype(np.float32)).clone()
        post_img = torch.from_numpy(post_img.astype(np.float32)).clone()
        target = torch.from_numpy(target.astype(np.float32)).clone()

        prev_img = torch.permute(prev_img, (2, 0, 1))
        img = torch.permute(img, (2, 0 ,1))
        post_img = torch.permute(post_img, (2, 0, 1))
        target = torch.permute(target, (2, 0, 1))

        if self.transform is not None:
            prev_img = self.transform(prev_img)
            img = self.transform(img)
            post_img = self.transform(post_img)

        return prev_img, img, post_img, target

    def _get_whole_scene_item(self, index: int):
        if index < (self.v1_imgs.shape[0] - 2):
            offset = 0
            prev_img = self.v1_imgs[index - offset]
            img = self.v1_imgs[index - offset + 1]
            post_img = self.v1_imgs[index - offset + 2]
            target = self.v1_targets[index - offset + 1]
        elif index < (self.v1_imgs.shape[0] - 2) + (self.v2_imgs.shape[0] - 2):
            offset = (self.v1_imgs.shape[0] - 2)
            prev_img = self.v2_imgs[index - offset]
            img = self.v2_imgs[index - offset + 1]
            post_img = self.v2_imgs[index - offset + 2]
            target = self.v2_targets[index - offset + 1]
        else:
            offset = (self.v1_imgs.shape[0] - 2) + (self.v2_imgs.shape[0] - 2)
            prev_img = self.v3_imgs[index - offset]
            img = self.v3_imgs[index - offset + 1]
            post_img = self.v3_imgs[index - offset + 2]
            target = self.v3_targets[index - offset + 1]

        return prev_img, img, post_img, target

    def _load_h5file(self, file_name):
        with h5py.File(os.path.join(self.data_dir, file_name), "r") as f:
            tmp_imgs = np.array(f['imgs'])
            tmp_targets = np.array(f["density_{}".format(self.data_type)])  # data_type: once or add

        return tmp_imgs, tmp_targets

class VeniceDataset(Dataset):
    def __init__(self, pathjson=None, transform=None, width=640, height=360) -> None:
        super().__init__()
        with open(pathjson, "r") as f:
            self.allpath = json.load(f)
        self.transform = transform
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.allpath)

    def __getitem__(self, index: int):
        prev_path = self.allpath[index]["prev"]
        now_path = self.allpath[index]["now"]
        next_path = self.allpath[index]["next"]
        target_path = self.allpath[index]["target"]

        prev_img = cv2.imread(prev_path)
        prev_img = cv2.resize(prev_img, (self.width, self.height))
        prev_img = prev_img / 255.0
        prev_img = self.transform(prev_img)

        now_img = cv2.imread(now_path)
        now_img = cv2.resize(now_img, (self.width, self.height))
        now_img = now_img / 255.0
        now_img = self.transform(now_img)

        next_img = cv2.imread(next_path)
        next_img = cv2.resize(next_img, (self.width, self.height))
        next_img = next_img / 255.0
        next_img = self.transform(next_img)

        target_dict = scipy.io.loadmat(target_path)
        target = np.zeros((720, 1280))

        for p in range(target_dict['annotation'].shape[0]):
            target[int(target_dict['annotation'][p][1]), int(target_dict['annotation'][p][0])] = 1

        target = gaussian_filter(target, 3) * 64
        target = cv2.resize(target, (80, 45))
        target = torch.from_numpy(target.astype(np.float32)).clone()

        return prev_img, now_img, next_img, target
    
class UCSDDataset(Dataset):
    def __init__(self, data, transform=None):
        # with open(json_path, 'r') as outfile:
        #     json_data = json.load(outfile)

        # self.data = []
        # for d in json_data.keys():
        #     self.data.append(json_data[d])

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        paths = self.data[index]

        prev_img, img, post_img, target = load_ucsd_data(paths["img"], paths["gt"])
        prev_img = self.transform(prev_img)
        img = self.transform(img)
        post_img = self.transform(post_img)
        # print(prev_img, img, post_img, target)
        # print(prev_img.size(), img.size(), post_img.size(), target.size())
        # print(prev_img.shape, img.shape, post_img.shape, target.shape)

        return prev_img, img, post_img, target


class Datapath():
    def __init__(self, json_path=None, datakind=None, mode="once") -> None:
        if datakind == "CrowdFlow":
            with open(json_path) as f:
                reader = csv.reader(f)
                self.img_paths = [row for row in reader]
        else:
            with open(json_path, 'r') as outfile:
                self.img_paths = json.load(outfile)
        self.datakind = datakind
        self.mode = mode

    def __getitem__(self, index):
        if self.datakind == "FDST":
            img_path = self.img_paths[i]

            img_folder = os.path.dirname(img_path)
            img_name = os.path.basename(img_path)
            index = int(img_name.split('.')[0])

            prev_index = int(max(1,index-5))
            prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))
            print(prev_img_path)
            prev_img = Image.open(prev_img_path).convert('RGB')
            img = Image.open(img_path).convert('RGB')

            gt_path = img_path.replace('.jpg','_resize.h5')
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density'])

            return prev_img, img, target

        elif self.datakind == "CrowdFlow":
            pathlist = self.img_paths[index]
            t_img_path = pathlist[0]
            t_person_path = pathlist[1]
            t_m_img_path = pathlist[2]
            t_m_person_path = pathlist[3]
            t_m_t_flow_path = pathlist[4]
            t_p_img_path = pathlist[5]
            t_p_person_path = pathlist[6]
            t_t_p_flow_path = pathlist[7]

            prev_img = Image.open(t_m_img_path).convert('RGB')
            img = Image.open(t_img_path).convert('RGB')

            if self.mode == "once":
                target = cv2.imread(t_person_path, 0)
                target = target / np.max(target)
                target = cv2.resize(target, (80, 45), interpolation=cv2.INTER_CUBIC)  # width, height
            else:
                target = np.load(t_person_path)["x"]

            return prev_img, img, target

        elif self.datakind == "venice":
            prev_path = self.img_paths[index]["prev"]
            now_path = self.img_paths[index]["now"]
            next_path = self.img_paths[index]["next"]
            target_path = self.img_paths[index]["target"]

            prev_img = Image.open(prev_path).convert('RGB')
            img = Image.open(now_path).convert('RGB')

            target_dict = scipy.io.loadmat(target_path)
            target = np.zeros((720, 1280))

            for p in range(target_dict['annotation'].shape[0]):
                target[int(target_dict['annotation'][p][1]), int(target_dict['annotation'][p][0])] = 1

            target = gaussian_filter(target, 3) * 64
            target = cv2.resize(target, (80, 45))

            return prev_img, img, target

        elif self.datakind == "animal":
            prev_path = self.img_paths[index]["prev"]
            now_path = self.img_paths[index]["now"]
            next_path = self.img_paths[index]["next"]
            target = None

            prev_img = Image.open(prev_path).convert('RGB')
            img = Image.open(now_path).convert('RGB')

            return prev_img, img, target

