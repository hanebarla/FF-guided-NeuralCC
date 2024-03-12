import os
import json
import csv

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import mean_absolute_error

from utils import *
from dataset import dataset_factory
from argument import create_test_args, load_args
from logger import create_logger

def main():
    test_args = create_test_args()
    args = load_args(test_args.saved_dir)

    with open(args.test_data, 'r') as outfile:
        test_data = json.load(outfile)

    if test_args.dynamicff == 1 and test_args.staticff == 1:
        subdir = 'dynamic_static'
    elif test_args.dynamicff == 1:
        subdir = 'dynamic'
    elif test_args.staticff == 1:
        subdir = 'static'
    else:
        subdir = 'baseline'
    save_dir = os.path.join(test_args.saved_dir, subdir)
    baseline_dir = os.path.join(test_args.saved_dir, 'baseline')

    logger = create_logger(save_dir, 'hyperparam_tune', 'tune.log')
    logger.info("[Test Args]: {}".format(str(test_args)))
    logger.info("[Train Args]: {}".format(str(args)))
    logger.info("[Save Dir]: {}".format(save_dir))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))

    # tune per scene
    for k in test_data.keys():
        save_dir_per_scene = os.path.join(save_dir, str(k))
        if not os.path.exists(save_dir_per_scene):
            os.makedirs(save_dir_per_scene)

        logger.info("Scene Num: {}".format(k))
        _, test_dataset = dataset_factory(args, None, test_data[k])
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.workers,
                                                      prefetch_factor=args.prefetch,
                                                      pin_memory=True)

        # load staticff
        staticff = None
        if test_args.staticff == 1:
            staticff_file = os.path.join(os.path.dirname(test_data[0]["target"]), "staticff_45x80.npz")
            staticff = np.load(staticff_file)["x"]

        # search param
        static_param, temperature_param, beta_param, delta_param = search_params(test_args, test_dataloader, baseline_dir, save_dir_per_scene, staticff, logger)
        logger.info("Scene Num: {}, StaticFF: {}, Temperature: {}, Beta: {}, Delta: {}".format(k, static_param, temperature_param, beta_param, delta_param))


def search_params(args, test_dataloader, baseline_dir, save_dir, staticff, logger):
    scene_num = args.scene_num

    target_nums = []
    static_param = 0
    static_param_pix = 0
    beta_param = 0
    beta_param_pix = 0
    delta_param = 0
    delta_param_pix = 0
    temperature_param = 0
    temperature_param_pix = 0
    mae = 1000
    pix_mae = 1000

    # parameter search area
    static_params = [0.8, 0.9, 1.0, 1.1, 1.2]
    beta_params = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5]
    delta_params = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5]
    temperature_params = [0.1, 0.5, 1., 5., 10., 50., 100.]
    # temperature_params = [0.3, 0.5, 1., 5., 10., 50., 100.]

    for i_s, s in enumerate(static_params):
        if args.StaticFF != 1 and i_s > 0:
            continue
        for i_t, temperature in enumerate(temperature_params):
            if args.DynamicFF != 1 and i_t > 0:
                continue
            for i_b, beta in enumerate(beta_params):
                if args.DynamicFF != 1 and i_b > 0:
                    continue
                for i_d, delta in enumerate(delta_params):
                    if args.DynamicFF != 1 and i_d > 0:
                        continue
                    logger.info("===== {}, {}, {}, {} =====".format(s, temperature, beta, delta))
                    tmp_output_nums = []
                    _pix_mae = []

                    for i, (prev_img, img, target) in enumerate(test_dataloader):
                        if i > scene_num:
                            break
                        target_num = np.array(target)
                        normal_dense = np.load(os.path.join(baseline_dir, "{}.npz".format(i)))["x"]
                        height, width = normal_dense.shape

                        if args.StaticFF == 1:
                            normal_dense *= s*staticff

                        normal_dense_gauss = gaussian_filter(normal_dense, 3)
                        if args.DynamicFF == 1 and past_output is not None:
                            d_t_prev = gaussian_filter(past_output, 3)
                            past_output = beta * normal_dense_gauss + (1 - delta) * d_t_prev
                            past_output = gaussian_filter(past_output, 3)
                            exp_sum = np.sum(np.exp(past_output/temperature)) + 1e-5
                            dynamicff = height * width * np.exp(past_output/temperature) / exp_sum
                            normal_dense *= dynamicff

                        if past_output is None:
                            past_output = beta * normal_dense_gauss

                        _pix_mae.append(np.nanmean(np.abs(np.squeeze(target_num)-normal_dense)))
                        tmp_output_nums.append(normal_dense.sum())

                    tmp_mae = mean_absolute_error(target_nums, tmp_output_nums)
                    tmp_pix_mae = np.mean(np.array(_pix_mae))
                    if tmp_mae < mae:
                        mae = tmp_mae
                        static_param = s
                        beta_param = beta
                        delta_param = delta
                        temperature_param = temperature

                    if tmp_pix_mae < pix_mae:
                        pix_mae = tmp_pix_mae
                        static_param_pix = s
                        beta_param_pix = beta
                        delta_param_pix = delta
                        temperature_param_pix = temperature

                    logger.info("MAE: {}, pix-MAE: {}".format(tmp_mae, tmp_pix_mae))
                    logger.info("Best MAE: {}, Best pix-MAE: {} (s: {}, b: {}, d: {}, t: {})"
                                .format(mae, pix_mae, static_param, beta_param, delta_param, temperature_param))

    with open(os.path.join(save_dir, 'ff_param.csv'), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([static_param, temperature_param, beta_param, delta_param])

    with open(os.path.join(save_dir, 'ff_param_pix.csv'), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([static_param_pix, temperature_param_pix, beta_param_pix, delta_param_pix])

    logger.info("With MAE, best StaticFF param: {}, best temperature param: {}, best beta param: {}, best delta param: {}"
                .format(static_param, temperature_param, beta_param, delta_param))
    logger.info("With pix-MAE, best StaticFF param: {}, best temperature param: {}, best beta param: {}, best delta param: {}"
                .format(static_param_pix, temperature_param_pix, beta_param_pix, delta_param_pix))

    return static_param, temperature_param, beta_param, delta_param

if __name__ == "__main__":
    main()
