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
from .utils import *
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
            data_type=args.data_mode,
            transform=transform
        )
        val_dataset = CrowdFlowDataset(
            val_data,
            data_type=args.data_mode,
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
    else:
        raise NotImplementedError("No such dataset {}".format(args.dataset))
    
    return train_dataset, val_dataset


class CrowdFlowDataset(Dataset):
    def __init__(self, dict, transform, mode="once"):
        self.dict = dict
        self.transform = transform
        self.mode = mode

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


"""
ras2bits = 0.71
IP = {0: 202.5, 1: 247.5, 2: 292.5, 3: 157.5, 5: 337.5, 6: 22.5, 7: 67.5, 8: 112.5}


class CrowdDatasets(torch.utils.data.Dataset):
    def __init__(self, Trainpath="Data/TrainData_Path.csv", transform=None, width=640, height=360, test_on=False, add=False):
        super().__init__()
        self.transform = transform
        self.width = width
        self.height = height
        self.out_width = int(width / 8)
        self.out_height = int(height / 8)
        self.test_on = test_on
        self.add = add
        with open(Trainpath) as f:
            reader = csv.reader(f)
            self.Pathes = [row for row in reader]

    def __len__(self):
        return len(self.Pathes)

    def __getitem__(self, index):
        pathlist = self.Pathes[index]
        t_img_path = pathlist[0]
        t_person_path = pathlist[1]
        t_m_img_path = pathlist[2]
        t_m_person_path = pathlist[3]
        t_m_t_flow_path = pathlist[4]
        t_p_img_path = pathlist[5]
        t_p_person_path = pathlist[6]
        t_t_p_flow_path = pathlist[7]

        if self.add:
            t_input, t_person = self.gt_npz_density(t_img_path, t_person_path)
            tm_input, tm_person = self.gt_npz_density(t_m_img_path, t_m_person_path)
            tp_input, tp_person = self.gt_npz_density(t_p_img_path, t_p_person_path)
        else:
            t_input, t_person = self.gt_img_density(t_img_path, t_person_path)
            tm_input, tm_person = self.gt_img_density(t_m_img_path, t_m_person_path)
            tp_input, tp_person = self.gt_img_density(t_p_img_path, t_p_person_path)

        # tm2t_flow = self.gt_flow(t_m_t_flow_path)
        # t2tp_flow = self.gt_flow(t_t_p_flow_path)

        return tm_input, t_input, tp_input, t_person

    def IndexProgress(self, i, gt_flow_edge, h, s):
        oheight = self.out_height
        owidth = self.out_width
        if i == 4:
            grid_i = np.zeros((oheight, owidth, 1))
            return grid_i
        elif i == 9:
            gt_flow_edge_ndarr = np.array(gt_flow_edge)
            gtflow_sum = np.sum(gt_flow_edge_ndarr, axis=0)
            grid_i = gtflow_sum
            return grid_i
        else:
            grid_i = np.where((h >= IP[i] * ras2bits) & (h < ((IP[i] + 45) % 360) * ras2bits), 1, 0)
            grid_i = np.array(grid_i, dtype=np.uint8)
            grid_i = s * grid_i
            grid_i = cv2.resize(grid_i, (owidth, oheight))  # width, height
            grid_i_inner = grid_i[1:(oheight-1), 1:(owidth-1)]
            grid_i_edge = grid_i
            grid_i_inner = np.pad(grid_i_inner, 1)
            grid_i_edge[1:(oheight-1), 1:(owidth-1)] = 0
            grid_i_inner = np.reshape(grid_i_inner, (oheight, owidth, 1))  # height, width, channel
            grid_i_edge = np.reshape(grid_i_edge, (oheight, owidth, 1))
            gt_flow_edge.append(grid_i_edge)

            return grid_i_inner

    def gt_flow(self, path):

        if not os.path.isfile(path):
            return print("No such file: {}".format(path))

        gt_flow_list = []
        gt_flow_edge = []
        img = cv2.imread(path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(img_hsv)
        for i in range(10):
            grid_i = self.IndexProgress(i, gt_flow_edge, h, s)
            gt_flow_list.append(grid_i)

        gt_flow_img_data = np.concatenate(gt_flow_list, axis=2)
        gt_flow_img_data /= np.max(gt_flow_img_data)


        # テスト
        # traj = np.sum(gt_flow_img_data, axis=2)
        # print(traj.shape)

        # root = os.getcwd()
        # imgfolder = root + "/images/"
        # heatmap = cv2.resize(traj, (traj.shape[1]*8, traj.shape[0]*8))
        # heatmap = np.array(heatmap*255, dtype=np.uint8)
        # cv2.imwrite(imgfolder+"flow_test.png", heatmap)
        # print(gt_flow_img_data.shape)
        # print(np.max(np.sum(gt_flow_img_data, axis=2)))

        gt_flow_img_data = transforms.ToTensor()(gt_flow_img_data)

        return gt_flow_img_data

    def gt_npz_density(self, img_path, gt_path):
        # print(gt_path)
        if not os.path.isfile(img_path):
            return print("No such file: {}".format(img_path))
        if not os.path.isfile(gt_path):
            return print("No such file: {}".format(gt_path))

        input_img = cv2.imread(img_path)
        input_img = cv2.resize(input_img, (self.width, self.height))  # width, height
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img / 255  # range [0:1]
        input_img = self.transform(input_img)

        gt = np.load(gt_path)["x"]
        # print(gt)
        gt = torch.from_numpy(gt.astype(np.float32)).clone()

        return input_img, gt


    def gt_img_density(self, img_path, mask_path):

        if not os.path.isfile(img_path):
            return print("No such file: {}".format(img_path))
        if not os.path.isfile(mask_path):
            return print("No such file: {}".format(mask_path))

        input_img = cv2.imread(img_path)
        input_img = cv2.resize(input_img, (self.width, self.height))  # width, height
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(mask_path, 0)
        if mask_img is None:
            return print("CRC error: {}".format(mask_path))
        mask_img = np.reshape(mask_img, (mask_img.shape[0], mask_img.shape[1], 1))

        input_img = input_img / 255  # range [0:1]
        mask_img = mask_img / np.max(mask_img)
        mask_img = cv2.resize(mask_img, (self.out_width, self.out_height), interpolation=cv2.INTER_CUBIC)  # width, height

        input_img = self.transform(input_img)
        # mask_img = transforms.ToTensor()(mask_img)
        # print("input max: {}".format(torch.max(input_img)))
        # print("intpu min: {}".format(torch.min(input_img)))

        return input_img, mask_img
"""

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

