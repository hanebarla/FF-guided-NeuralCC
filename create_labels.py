import os
import glob
import json
import csv
import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, io


def main():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the root folder of the Datasets.
                                                 In default, path is 'E:/Dataset/TUBCrowdFlow/'
                                                 """)
    parser.add_argument('dataset', default='CrowdFlow', help='dataset name')
    parser.add_argument('-p', '--path', default='/home/data/TUBCrowdFlow/')
    parser.add_argument('--mode', default="once", choices=["add", "once"])
    
    args = parser.parse_args()

    print("Dataset: {}, Mode: {}".format(args.dataset, args.mode))
    if args.dataset == "CrowdFlow":
        print("CrowdFlowDatasetGenerator")
        CrowdFlowDatasetGenerator(args.path, args.mode)
        print("create_crowdflow_json")
        create_crowdflow_json(args)
        print("concat_crowdflow_csv")
        concat_crowdflow_csv(args)
    elif args.dataset == "FDST":
        FDSTDatasetGenerator(args.path, args.mode)
        create_fdst_json(args)
    elif args.dataset == "CityStreet":
        pass
    elif args.dataset == "venice":
        pass
    elif args.dataset == "ucsd":
        UCSDDatasetGenerator(args.path, args.mode)
        create_ucsd_json(args)
        count_ucsd(args)
    else:
        raise NotImplementedError("Dataset {} is not supported.".format(args.dataset))
    

def person_annotate_img_generator(person_pos, frame, mode="add"):
    """
    Gaussian Spot Annotation
    ------
        (txtfile, image) --> (annotated image)
    """
    print("Mode: ", mode)
    anno = np.zeros((720, 1280))
    anno_input_same = np.zeros((360, 640))
    # anno_input_half = np.zeros((180, 320))
    # anno_input_quarter = np.zeros((90, 160))
    # anno_input_eighth = np.zeros((45, 80))
    rate_h = anno.shape[0] / anno_input_same.shape[0]
    rate_w = anno.shape[1] / anno_input_same.shape[1]

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

        # print(pos[0], rate_h, pos[1], rate_w)
        y_anno = min(int(pos[0]/rate_h), 359)
        x_anno = min(int(pos[1]/rate_w), 639)
        if mode == "add":
            anno_input_same[y_anno, x_anno] += 1.0
        elif mode == "once":
            anno_input_same[y_anno, x_anno] = 1.0
        else:
            raise NotImplementedError("mode should be 'add' or 'once'")

    print(anno_input_same.max())
    anno = ndimage.filters.gaussian_filter(anno, 3)
    anno_input_same = ndimage.filters.gaussian_filter(anno_input_same, 3)
    anno_input_eighth = cv2.resize(anno_input_same,
                                   (int(anno_input_same.shape[1]/8), int(anno_input_same.shape[0]/8)),
                                   interpolation=cv2.INTER_CUBIC)*64

    return anno, anno_input_same, anno_input_eighth


def CrowdFlowDatasetGenerator(root, mode="once"):
    frame_num_list = [300, 300, 250, 300, 450]
    FFdataset_path = os.path.join(root, "ff_{}".format(mode))
    if not os.path.isdir(FFdataset_path):
        os.mkdir(FFdataset_path)

    for i in range(5):
        scene = "IM0{}_hDyn".format(i+1)

        staticff_label = np.zeros((720, 1280))
        staticff_label_360x640 = np.zeros((360, 640))
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
            anno, anno_input_same, anno_input_eighth = person_annotate_img_generator(scene_pos, fr, mode=mode)
            staticff_label += anno
            staticff_label_360x640 += anno_input_same
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}.npz".format(fr)), x=anno)
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}_360x640.npz".format(fr)), x=anno_input_same)
            np.savez_compressed(os.path.join(FFdataset_path, scene, "{}_45x80.npz".format(fr)), x=anno_input_eighth)

        staticff_label[staticff_label>1] = 1.0
        staticff_label_360x640[staticff_label_360x640>1] = 1.0
        staticff_label_45x80[staticff_label_45x80>1] = 1.0
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_720x1280.npz"), x=staticff_label)
        np.savez_compressed(os.path.join(FFdataset_path, scene, "staticff_360x640.npz"), x=staticff_label_360x640)
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
            json.dump({scene[3:-1]: tmpPathList}, f)

def json_file_concat(file_list, file_name):
    file_data_dict = {}
    for file in file_list:
        with open(file, mode="r") as f:
            tmp_dict = json.load(f)
            file_data_dict.update(tmp_dict)
    with open(file_name, 'w') as f:
        json.dump(file_data_dict, f)

def cross_dataset(args, train_list, val_list, test_list, concat_file_index):
    train_file_list = []
    for file_name in train_list:
        train_file_list.append(os.path.join('Scene_IM0{}_{}.json'.format(file_name, args.mode)))
    # json_file_concat(train_file_list, os.path.join('crowdflow{}_train_{}.json'.format(concat_file_index, args.mode)))
    json_file_concat(train_file_list, os.path.join('crowdflow_{}_{}_train.json'.format(args.mode, concat_file_index)))

    val_file_list = []
    for file_name in val_list:
        val_file_list.append(os.path.join('Scene_IM0{}_{}.json'.format(file_name, args.mode)))
    # json_file_concat(val_file_list, os.path.join('crowdflow{}_val_{}.json'.format(concat_file_index, args.mode)))
    json_file_concat(val_file_list, os.path.join('crowdflow_{}_{}_val.json'.format(args.mode, concat_file_index)))

    test_file_list = []
    for file_name in test_list:
        test_file_list.append(os.path.join('Scene_IM0{}_{}.json'.format(file_name, args.mode)))
    # json_file_concat(test_file_list, os.path.join('crowdflow{}_test_{}.json'.format(concat_file_index, args.mode)))
    json_file_concat(test_file_list, os.path.join('crowdflow_{}_{}_test.json'.format(args.mode, concat_file_index)))

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
        np.savez_compressed(save_path, x=k)

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
    train_output_path_dict_per_scene = {}
    test_output_path_dict_per_scene = {}

    staticff = None
    for scene in train_scene:
        img_paths = os.listdir(os.path.join(train_dir, str(scene)))
        img_paths = [img_path for img_path in img_paths if "."!=img_path[0]]
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] == 'jpg']
        img_paths = [os.path.join(train_dir, str(scene), img_path) for img_path in img_paths]
        img_paths.sort()

        for img_path in img_paths:
            gt_path = img_path.replace('.jpg', '_{}.npz'.format(args.mode))
            train_output_path_dict.append({'img': img_path, 'gt': gt_path})
            if scene not in train_output_path_dict_per_scene:
                train_output_path_dict_per_scene[scene] = [{'img': img_path, 'gt': gt_path}]
            else:
                train_output_path_dict_per_scene[scene].append({'img': img_path, 'gt': gt_path})
            target = np.load(gt_path)['x']
            target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            target = ndimage.gaussian_filter(target, 3)

            if staticff is None:
                staticff = target
            else:
                staticff += target

        staticff[staticff>1] = 1.0
        staticff_path = os.path.join(train_dir, str(scene), 'staticff_45x80.npz')
        np.savez_compressed(staticff_path, x=staticff)

    for scene in test_scene:
        img_paths = os.listdir(os.path.join(test_dir, str(scene)))
        img_paths = [img_path for img_path in img_paths if "."!=img_path[0]]
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] == 'jpg']
        img_paths = [os.path.join(test_dir, str(scene), img_path) for img_path in img_paths]
        img_paths.sort()

        for img_path in img_paths:
            gt_path = img_path.replace('.jpg', '_{}.npz'.format(args.mode))
            test_output_path_dict.append({'img': img_path, 'gt': gt_path})
            if scene not in test_output_path_dict_per_scene:
                test_output_path_dict_per_scene[scene] = [{'img': img_path, 'gt': gt_path}]
            else:
                test_output_path_dict_per_scene[scene].append({'img': img_path, 'gt': gt_path})

            target = np.load(gt_path)['x']
            target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64

            if staticff is None:
                staticff = target
            else:
                staticff += target
        
        staticff[staticff>1] = 1.0
        staticff_path = os.path.join(test_dir, str(scene), 'staticff_45x80.npz')
        np.savez_compressed(staticff_path, x=staticff)
    
    # with open(os.path.join('fdst_{}_train.json'.format(args.mode)), 'w') as f:
    #     json.dump(train_output_path_dict, f)
    # with open(os.path.join('fdst_{}_test.json'.format(args.mode)), 'w') as f:
    #     json.dump(test_output_path_dict, f)
    with open(os.path.join('fdst_{}_train.json'.format(args.mode)), 'w') as f:
        json.dump(train_output_path_dict_per_scene, f)
    with open(os.path.join('fdst_{}_test.json'.format(args.mode)), 'w') as f:
        json.dump(test_output_path_dict_per_scene, f)
    # with open(os.path.join('fdst_{}_train_per_scene.json'.format(args.mode)), 'w') as f:
    #     json.dump(train_output_path_dict_per_scene, f)
    # with open(os.path.join('fdst_{}_test_per_scene.json'.format(args.mode)), 'w') as f:
    #     json.dump(test_output_path_dict_per_scene, f)

def UCSDDatasetGenerator(root, mode="once"):
    # root: /groups1/gca50095/aca10350zi/ucsdpeds
    # train: 003 ~ 006
    # test: 000 ~ 002, 007 ~ 009
    train_scene_nums = [3, 4, 5, 6]
    test_scene_nums = [0, 1, 2, 7, 8, 9]
    train_dir = os.path.join(root, 'vidf')

    for scene_num in train_scene_nums:
        scene_dir = os.path.join(train_dir, 'vidf1_33_00{}.y'.format(scene_num))
        target_file = os.path.join(root, "vidf-cvpr", 'vidf1_33_00{}_people_full.mat'.format(scene_num))
        mat_data = io.loadmat(target_file, squeeze_me=True)
        staticff = None
        print(type(mat_data["people"][3]))
        # for p in range(mat_data["people"].shape[0]):
        #     tmp_data = mat_data["people"][p].tolist()  # 0: id, 1: scene, 2: loc, 3: num_pts, 4: ldir, 5: tdir
        #     for tmp in tmp_data[2]:
        #         print(tmp)
                # if tmp[0] < 0 or tmp[1] < 0:
                    # print("minus position")
                    # print(tmp)
            # print(tmp_data[2])

        for i in range(200):
            img_path = os.path.join(scene_dir, 'vidf1_33_00{}_f{:0=3}.png'.format(scene_num, i+1))
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # target = np.zeros_like(img, dtype=np.float32)
            large_target = np.zeros_like(cv2.imread(img_path), dtype=np.float32)
            target = np.zeros((int(img.shape[0]/8), int(img.shape[1]/8)))
            # print(target.shape)
            max_w, max_h = img.shape[1], img.shape[0]
            resized_max_w, resized_max_h = int(max_w/8), int(max_h/8)

            num = 0
            for p in range(mat_data["people"].shape[0]):
                # print("person: ", p)
                tmp_data = mat_data["people"][p].tolist()
                for tmp in tmp_data[2]:  # 0: width pos, 1: height pos
                    if tmp[0] < 0 or tmp[1] < 0:
                        # print("minus position")
                        continue
                    elif int(tmp[0]/8) >= resized_max_w or int(tmp[1]/8) >= resized_max_h:
                        # print("over position")
                        continue
                    elif tmp[-1] != i+1:
                        # print("frame mismatch")
                        continue
                    # print(tmp)
                    # print(max_w, tmp[0], resized_max_w, tmp[0]/8, int(tmp[0]/8), max_h, tmp[1], resized_max_h, tmp[1]/8, int(tmp[1]/8))
                    num += 1
                    target[int(tmp[1]/8), int(tmp[0]/8)] += 1.0
                    large_target[int(tmp[1]), int(tmp[0])] += 1.0

            original_target = target
            target = ndimage.gaussian_filter(target, 3)
            gauss_target = target
            # target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            # shrunken_target = target
            # target = ndimage.gaussian_filter(target, 3)
            # second_gauss_target = target
            # print(original_target.sum(), first_gauss_target.sum(), shrunken_target.sum(), second_gauss_target.sum())
            large_target = ndimage.gaussian_filter(large_target, 3)
            large_target = cv2.resize(large_target, (int(large_target.shape[1] / 8), int(large_target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            print(original_target.sum(), gauss_target.sum(), large_target.sum())
            if staticff is None:
                staticff = target
            else:
                staticff += target
            # print(target.max())

            target_file = img_path.replace('.png', '_{}.npz'.format(mode))
            target_img_path = img_path.replace('.png', '_target.png')
            cv2.imwrite(target_img_path, np.array(target/target.max()*255, dtype=np.uint8))
            # np.savez_compressed(target_file, x=target, num=num, original=original_target, first_gauss=first_gauss_target, shrunken=shrunken_target, second_gauss=second_gauss_target)
            np.savez_compressed(target_file, x=target, num=num, original=original_target, gauss=gauss_target, large=large_target)

        staticff[staticff>1] = 1.0
        staticff_path = os.path.join(scene_dir, 'staticff.npz')
        np.savez_compressed(staticff_path, x=staticff)
        cv2.imwrite(os.path.join(scene_dir, 'staticff.png'), np.array(staticff/staticff.max()*255, dtype=np.uint8))

    for scene_num in test_scene_nums:
        scene_dir = os.path.join(train_dir, 'vidf1_33_00{}.y'.format(scene_num))
        target_file = os.path.join(root, "vidf-cvpr", 'vidf1_33_00{}_people_full.mat'.format(scene_num))
        mat_data = io.loadmat(target_file, squeeze_me=True)
        staticff = None

        for i in range(200):
            img_path = os.path.join(scene_dir, 'vidf1_33_00{}_f{:0=3}.png'.format(scene_num, i+1))
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            large_target = np.zeros_like(cv2.imread(img_path), dtype=np.float32)
            # target = np.zeros_like(img, dtype=np.float32)
            target = np.zeros((int(img.shape[0]/8), int(img.shape[1]/8)))
            max_w, max_h = img.shape[1], img.shape[0]
            resized_max_w, resized_max_h = int(max_w/8), int(max_h/8)

            num = 0
            for p in range(mat_data["people"].shape[0]):
                # print("person: ", p)
                tmp_data = mat_data["people"][p].tolist()
                for tmp in tmp_data[2]:
                    if tmp[0] < 0 or tmp[1] < 0:
                        # print("minus position")
                        continue
                    elif int(tmp[0]/8) >= resized_max_w or int(tmp[1]/8) >= resized_max_h:
                        # print("over position")
                        continue
                    elif tmp[-1] != i+1:
                        # print("frame mismatch")
                        continue
                    # print(tmp)
                    num += 1
                    target[int(tmp[1]/8), int(tmp[0]/8)] += 1.0
                    large_target[int(tmp[1]), int(tmp[0])] += 1.0

            original_target = target
            target = ndimage.gaussian_filter(target, 3)
            gauss_target = target
            # target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            # target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8))) * 64
            # shrunken_target = target
            # target = ndimage.gaussian_filter(target, 3)
            # second_gauss_target = target
            large_target = ndimage.gaussian_filter(large_target, 3)
            large_target = cv2.resize(large_target, (int(large_target.shape[1] / 8), int(large_target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
            if staticff is None:
                staticff = target
            else:
                staticff += target
            # print(original_target.sum(), first_gauss_target.sum(), shrunken_target.sum(), second_gauss_target.sum())
            print(original_target.sum(), gauss_target.sum(), large_target.sum())

            target_file = img_path.replace('.png', '_{}.npz'.format(mode))
            target_img_path = img_path.replace('.png', '_target.png')
            cv2.imwrite(target_img_path, np.array(target/target.max()*255, dtype=np.uint8))
            # np.savez_compressed(target_file, x=target, num=num, original=original_target, first_gauss=first_gauss_target, shrunken=shrunken_target, second_gauss=second_gauss_target)
            np.savez_compressed(target_file, x=target, num=num, original=original_target, gauss=gauss_target, large=large_target)

        staticff[staticff>1] = 1.0
        staticff_path = os.path.join(scene_dir, 'staticff.npz')
        np.savez_compressed(staticff_path, x=staticff)
        cv2.imwrite(os.path.join(scene_dir, 'staticff.png'), np.array(staticff/staticff.max()*255, dtype=np.uint8))

def create_ucsd_json(args):
    train_scene_nums = [3, 4, 5, 6]
    test_scene_nums = [0, 1, 2, 7, 8, 9]
    root_dir = os.path.join(args.path, 'vidf')

    train_output_path_dict = []
    test_output_path_dict = []
    train_output_path_dict_per_scene = {}
    test_output_path_dict_per_scene = {}

    for scene in train_scene_nums:
        img_paths = os.listdir(os.path.join(root_dir, "vidf1_33_00{}.y".format(scene)))
        img_paths = [img_path for img_path in img_paths if "."!=img_path[0]]
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] == 'png']
        img_paths = [img_path for img_path in img_paths if img_path.split('_')[-1] != 'target.png']
        img_paths = [img_path for img_path in img_paths if img_path.split('_')[-1] != 'staticff.png']
        img_paths = [os.path.join(root_dir, "vidf1_33_00{}.y".format(scene), img_path) for img_path in img_paths]
        img_paths.sort()
        print(img_paths)

        for img_path in img_paths:
            gt_path = img_path.replace('.png', '_{}.npz'.format(args.mode))
            train_output_path_dict.append({'img': img_path, 'gt': gt_path})
            if scene not in train_output_path_dict_per_scene:
                train_output_path_dict_per_scene[scene] = [{'img': img_path, 'gt': gt_path}]
            else:
                train_output_path_dict_per_scene[scene].append({'img': img_path, 'gt': gt_path})

    for scene in test_scene_nums:
        img_paths = os.listdir(os.path.join(root_dir, "vidf1_33_00{}.y".format(scene)))
        img_paths = [img_path for img_path in img_paths if "."!=img_path[0]]
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] == 'png']
        img_paths = [img_path for img_path in img_paths if img_path.split('_')[-1] != 'target.png']
        img_paths = [img_path for img_path in img_paths if img_path.split('_')[-1] != 'staticff.png']
        img_paths = [os.path.join(root_dir, "vidf1_33_00{}.y".format(scene), img_path) for img_path in img_paths]
        img_paths.sort()
        print(img_paths)

        for img_path in img_paths:
            gt_path = img_path.replace('.png', '_{}.npz'.format(args.mode))
            test_output_path_dict.append({'img': img_path, 'gt': gt_path})
            if scene not in test_output_path_dict_per_scene:
                test_output_path_dict_per_scene[scene] = [{'img': img_path, 'gt': gt_path}]
            else:
                test_output_path_dict_per_scene[scene].append({'img': img_path, 'gt': gt_path})

    # with open(os.path.join('fdst_{}_train.json'.format(args.mode)), 'w') as f:
    #     json.dump(train_output_path_dict, f)
    # with open(os.path.join('fdst_{}_test.json'.format(args.mode)), 'w') as f:
    #     json.dump(test_output_path_dict, f)
    with open(os.path.join('ucsd_{}_train.json'.format(args.mode)), 'w') as f:
        json.dump(train_output_path_dict_per_scene, f)
    with open(os.path.join('ucsd_{}_test.json'.format(args.mode)), 'w') as f:
        json.dump(test_output_path_dict_per_scene, f)
    # with open(os.path.join('fdst_{}_train_per_scene.json'.format(args.mode)), 'w') as f:
    #     json.dump(train_output_path_dict_per_scene, f)
    # with open(os.path.join('fdst_{}_test_per_scene.json'.format(args.mode)), 'w') as f:
    #     json.dump(test_output_path_dict_per_scene, f)

def count_ucsd(args):
    origin_targets = []
    gauss_targets = []
    # shrunken_targets = []
    # second_gauss_targets = []
    targets = []
    large_targets = []
    nums = []
    for i in range(10):
        for t in range(200):
            tmp_Data = np.load(os.path.join(args.path, 'vidf', 'vidf1_33_00{}.y'.format(i), 'vidf1_33_00{}_f{:0=3}_once.npz'.format(i, t+1)))
            tmp_target = tmp_Data['x']
            tmp_num = tmp_Data['num']
            # print(tmp_num.sum())
            origin_targets.append(tmp_Data['original'].sum())
            gauss_targets.append(tmp_Data['gauss'].sum())
            # shrunken_targets.append(tmp_Data['shrunken'].sum())
            # second_gauss_targets.append(tmp_Data['second_gauss'].sum())
            targets.append(tmp_target.sum())
            large_targets.append(tmp_Data['large'].sum())
            nums.append(tmp_num)

    scales = np.array(targets) / np.array(large_targets)
    print("MAE: ", np.abs(np.array(gauss_targets) - np.array(nums)).mean())
    print("Scale: ", scales.mean())

    plt.plot(targets, label="final Label", alpha=0.5)
    plt.plot(nums, label="Ground Truth", alpha=0.5)
    plt.plot(origin_targets, label="Original Label", alpha=0.5)
    plt.plot(gauss_targets, label="Gaussian Label", alpha=0.5)
    plt.plot(large_targets, label="Large Label", alpha=0.5)
    # plt.plot(shrunken_targets, label="Shrunken Label", alpha=0.5)
    # plt.plot(second_gauss_targets, label="Second Gaussian Label")
    plt.legend()
    plt.savefig('ucsd.png', dpi=300)

if __name__ == "__main__":
    main()
