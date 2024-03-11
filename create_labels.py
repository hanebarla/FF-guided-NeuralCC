import os
import glob
import json
import csv
import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import trange


def main():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the root folder of the Datasets.
                                                 In default, path is 'E:/Dataset/TUBCrowdFlow/'
                                                 """)
    parser.add_argument('dataset', default='CrowdFlow', help='dataset name')
    parser.add_argument('-p', '--path', default='/groups1/gca50095/aca10350zi/TUBCrowdFlow/')
    parser.add_argument('--mode', default="once", choices=["add", "once"])
    
    args = parser.parse_args()

    if args.dataset == "CrowdFlow":
        CrowdFlowDatasetGenerator(args.path, args.mode)
        create_crowdflow_json(args)
        concat_crowdflow_csv(args)
    elif args.dataset == "FDST":
        FDSTDatasetGenerator(args.path, args.mode)
        create_fdst_json(args)
    elif args.dataset == "CityStreet":
        pass
    elif args.dataset == "venice":
        pass
    else:
        raise NotImplementedError("Dataset {} is not supported.".format(args.dataset))
    

def person_annotate_img_generator(person_pos, frame, mode="add"):
    """
    Gaussian Spot Annotation
    ------
        (txtfile, image) --> (annotated image)
    """
    anno = np.zeros((720, 1280))
    anno_input_same = np.zeros((360, 640))
    anno_input_half = np.zeros((180, 320))
    anno_input_quarter = np.zeros((90, 160))
    anno_input_eighth = np.zeros((45, 80))

    frame_per_pos = np.array(
        person_pos[:, 2 * frame: 2 * (frame + 1)], dtype=np.uint32)
    shape = frame_per_pos.shape

    for i in range(shape[0]):
        pos = frame_per_pos[i, :]
        if pos[0] == 0 and pos[1] == 0:
            continue
        elif pos[0] >= 720 or pos[1] >= 1280:
            continue
        elif pos[0] < 0 or pos[1] < 0:
            continue
        anno[pos[0], pos[1]] = 1.0

        if mode == "add":
            anno_input_same[int(pos[0]/2), int(pos[1]/2)] += 1.0
            anno_input_half[int(pos[0]/4), int(pos[1]/4)] += 1.0
            anno_input_quarter[int(pos[0]/8), int(pos[1]/8)] += 1.0
            anno_input_eighth[int(pos[0]/16), int(pos[1]/16)] += 1.0
        elif mode == "once":
            anno_input_same[int(pos[0]/2), int(pos[1]/2)] = 1.0
            anno_input_half[int(pos[0]/4), int(pos[1]/4)] = 1.0
            anno_input_quarter[int(pos[0]/8), int(pos[1]/8)] = 1.0
            anno_input_eighth[int(pos[0]/16), int(pos[1]/16)] = 1.0
        else:
            raise ValueError

    anno = ndimage.filters.gaussian_filter(anno, 3)
    anno_input_same = ndimage.filters.gaussian_filter(anno_input_same, 3)
    anno_input_half = ndimage.filters.gaussian_filter(anno_input_half, 3)
    anno_input_quarter = ndimage.filters.gaussian_filter(anno_input_quarter, 3)
    anno_input_eighth = ndimage.filters.gaussian_filter(anno_input_eighth, 3)

    return anno, anno_input_same, anno_input_half, anno_input_quarter, anno_input_eighth


def CrowdFlowDatasetGenerator(root, mode="once"):
    frame_num_list = [300, 300, 250, 300, 450]
    FFdataset_path = os.path.join(root, "ff_{}".format(mode))
    if not os.path.isdir(FFdataset_path):
        os.mkdir(FFdataset_path)
    
    for i in range(5):
        scene = "IM0{}".format(i+1)

        staticff_label = np.zeros((720, 1280))
        staticff_label_360x640 = np.zeros((360, 640))
        staticff_label_180x320 = np.zeros((180, 320))
        staticff_label_90x160 = np.zeros((90, 160))
        staticff_label_45x80 = np.zeros((45, 80))

        scene_pos = np.loadtxt(
            os.path.join(
                root,
                "gt_trajectories",
                scene,
                "personTrajectories.csv"),
            delimiter=",")
        
        frame_num = scene_pos.shape[1] // 2

        if not os.path.isdir(FFdataset_path):
            os.mkdir(FFdataset_path)
        if not os.path.isdir(os.path.join(FFdataset_path, scene)):
            os.makedirs(os.path.join(FFdataset_path, scene), exist_ok=True)

        for fr in range(frame_num):
            anno, anno_input_same, anno_input_half, anno_input_quarter, anno_input_eighth = person_annotate_img_generator(scene_pos, fr)
            staticff_label += anno
            staticff_label_360x640 += anno_input_same
            staticff_label_180x320 += anno_input_half
            staticff_label_90x160 += anno_input_quarter
            staticff_label_45x80 += anno_input_eighth
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}.npz".format(fr)), x=anno)
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}_360x640.npz".format(fr)), x=anno_input_same)
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}_180x320.npz".format(fr)), x=anno_input_half)
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}_90x160.npz".format(fr)), x=anno_input_quarter)
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}_45x80.npz".format(fr)), x=anno_input_eighth)

        staticff_label[staticff_label>1] = 1.0
        staticff_label_360x640[staticff_label_360x640>1] = 1.0
        staticff_label_180x320[staticff_label_180x320>1] = 1.0
        staticff_label_90x160[staticff_label_90x160>1] = 1.0
        staticff_label_45x80[staticff_label_45x80>1] = 1.0
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_720x1280.npz"), x=staticff_label)
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_360x640.npz"), x=staticff_label_360x640)
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_180x320.npz"), x=staticff_label_180x320)
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_90x160.npz"), x=staticff_label_90x160)
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_45x80.npz"), x=staticff_label_45x80)

def create_crowdflow_json(args):
    frame_num_list = [300, 300, 250, 300, 450]
    DatasetFolder = args.path
    ImgFolder = DatasetFolder + "images/"
    # GTTrajFolder = DatasetFolder + "gt_trajectories/"
    GTTrajFolder = DatasetFolder + "ff_{}/".format(args.mode)
    GTFlowFolder = DatasetFolder + "gt_flow/"
    GTPersonFolder = "PersonTrajectories/"
    SceneFolderNameLis = [
        "IM01/", "IM01_hDyn/",
        "IM02/", "IM02_hDyn/",
        "IM03/", "IM03_hDyn/",
        "IM04/", "IM04_hDyn/",
        "IM05/", "IM05_hDyn/"
    ]

    for i, scene in enumerate(SceneFolderNameLis):
        frame_num = frame_num_list[int(i / 2)]
        # gtTraj_img_path = GTTrajFolder + scene + GTPersonFolder
        gtTraj_img_path = GTTrajFolder + scene

        tmpPathList = []

        for fr in range(frame_num - 2):
            tm = fr
            t = fr + 1
            tp = fr + 2

            t_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(t)
            tm_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(tm)
            tp_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(tp)

            # t_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(t)
            t_person_img_path = gtTraj_img_path + "{}_45x80.npz".format(t)
            # tm_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(tm)
            tm_person_img_path = gtTraj_img_path + "{}_45x80.npz".format(tm)
            # tp_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(tp)
            tp_person_img_path = gtTraj_img_path + "{}_45x80.npz".format(tp)

            tm2t_flow_path = GTFlowFolder + scene + "frameGT_{:0=4}.png".format(tm)
            t2tp_flow_path = GTFlowFolder + scene + "frameGT_{:0=4}.png".format(t)

            PathList_per_frame = {
                "prev": tm_img_path,
                "now": t_img_path,
                "post": tp_img_path,
                "target": t_person_img_path,
            }

            tmpPathList.append(PathList_per_frame)

        with open("Scene_{}_{}.json".format(scene.replace("/", ""), args.mode), "w") as f:
            json.dump(tmpPathList, f)

def json_file_concat(file_list, file_name):
    file_data_list = []
    for file in file_list:
        with open(file, mode="r") as f:
            file_data_list += json.load(f)
    with open(file_name, 'w') as f:
        json.dump(file_data_list, f)

def cross_dataset(args, train_list, val_list, test_list, concat_file_index):
    train_file_list = []
    for file_name in train_list:
        train_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}_{}.json'.format(file_name, args.mode)))
    json_file_concat(train_file_list, os.path.join(args.savefolder, '{}_train_{}.json'.format(concat_file_index, args.mode)))

    val_file_list = []
    for file_name in val_list:
        val_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}_{}.json'.format(file_name, args.mode)))
    json_file_concat(val_file_list, os.path.join(args.savefolder, '{}_val_{}.json'.format(concat_file_index, args.mode)))

    test_file_list = []
    for file_name in test_list:
        test_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}_{}.json'.format(file_name, args.mode)))
    json_file_concat(test_file_list, os.path.join(args.savefolder, '{}_test_{}.json'.format(concat_file_index, args.mode)))

def concat_crowdflow_csv(args):
    A_train_dataset = [1, 2, 3]
    A_val_dataset = [4]
    A_test_dataset = [5]
    cross_dataset(args, A_train_dataset, A_val_dataset, A_test_dataset, 'A')

    B_train_dataset = [2, 3, 4]
    B_val_dataset = [5]
    B_test_dataset = [1]
    cross_dataset(args, B_train_dataset, B_val_dataset, B_test_dataset, 'B')

    C_train_dataset = [3, 4, 5]
    C_val_dataset = [1]
    C_test_dataset = [2]
    cross_dataset(args, C_train_dataset, C_val_dataset, C_test_dataset, 'C')

    D_train_dataset = [4, 5, 1]
    D_val_dataset = [2]
    D_test_dataset = [3]
    cross_dataset(args, D_train_dataset, D_val_dataset, D_test_dataset, 'D')

    E_train_dataset = [5, 1, 2]
    E_val_dataset = [3]
    E_test_dataset = [4]
    cross_dataset(args, E_train_dataset, E_val_dataset, E_test_dataset, 'E')

def FDSTDatasetGenerator(root, mode="once"):
    train_dir = os.path.join(root, 'train_data')
    test_dir = os.path.join(root, 'test_data')
    
    train_path = []
    for f in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, f)):
            train_path.append(os.path.join(train_dir, f))
    test_path = []
    for f in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, f)):
            test_path.append(os.path.join(test_dir, f))
    path_sets = train_path + test_path
            
    img_paths = []
    for p in path_sets:
        for img_path in glob.glob(os.path.join(p, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        if len(img_path.split('_')) != 2:
            continue
        gt_path = img_path.replace('.jpg', '.json')
        with open(gt_path,'r') as f:
            gt = json.load(f)

        anno_list = list(gt.values())[0]['regions']
        img = plt.imread(img_path)
        k = np.zeros((360, 640))
        rate_h = img.shape[0] / 360
        rate_w = img.shape[1] / 640
        for i in range(len(anno_list)):
            y_anno = min(int(anno_list[i]['shape_attributes']['y']/rate_h), 359)
            x_anno = min(int(anno_list[i]['shape_attributes']['x']/rate_w), 639)

            if mode == "add":
                k[y_anno, x_anno] += 1
            elif mode == "once":
                k[y_anno, x_anno] = 1
            else:
                raise NotImplementedError("mode should be 'add' or 'once'")
            
        k = ndimage.filters.gaussian_filter(k, 3)

        save_path = img_path.replace('.jpg', '_{}.npz'.format(mode))
        np.savez_compressed(save_path, density=k)

def create_fdst_json(args):
    train_dir = os.path.join(args.path, 'train_data')
    test_dir = os.path.join(args.path, 'test_data')
    
    train_scene = []
    test_scene = []
    for i in range(100):
        if (i+1) % 5 == 4 or (i+1) % 5 == 0:
            test_scene.append(i+1)
        else:
            train_scene.append(i+1)

    train_output_path_dict = []
    test_output_path_dict = []
    staticff = None
    for scene in train_scene:
        img_paths = os.listdir(os.path.join(train_dir, scene))
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] == 'jpg']
        img_paths = [os.path.join(train_dir, scene, img_path) for img_path in img_paths]
        img_paths.sort()

        for img_path in img_paths:
            gt_path = img_path.replace('.jpg', '_{}.npz'.format(args.mode))
            train_output_path_dict.append({'img': img_path, 'gt': gt_path})

            target = np.load(gt_path)['density']
            target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            target = ndimage.gaussian_filter(target, 3)

            if staticff is None:
                staticff = target
            else:
                staticff += target
        
        staticff[staticff>1] = 1.0
        staticff_path = os.path.join(train_dir, scene, 'staticff_45x80.npz')
        np.savez_compressed(staticff_path, x=staticff)

    for scene in test_scene:
        img_paths = os.listdir(os.path.join(test_dir, scene))
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] == 'jpg']
        img_paths = [os.path.join(test_dir, scene, img_path) for img_path in img_paths]
        img_paths.sort()

        for img_path in img_paths:
            gt_path = img_path.replace('.jpg', '_{}.npz'.format(args.mode))
            test_output_path_dict.append({'img': img_path, 'gt': gt_path})

            target = np.load(gt_path)['density']
            target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            target = ndimage.gaussian_filter(target, 3)

            if staticff is None:
                staticff = target
            else:
                staticff += target
        
        staticff[staticff>1] = 1.0
        staticff_path = os.path.join(test_dir, scene, 'staticff_45x80.npz')
        np.savez_compressed(staticff_path, x=staticff)
    
    with open(os.path.join(args.path, 'fdst_train.json'), 'w') as f:
        json.dump(train_output_path_dict, f)
    with open(os.path.join(args.path, 'fdst_test.json'), 'w') as f:
        json.dump(test_output_path_dict, f)

if __name__ == "__main__":
    main()
