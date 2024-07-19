import csv
import os
import json

import torch
from torch import nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter

import dataset
from model import CANNet2s
from utils import save_checkpoint, fix_model_state_dict
from argument import create_test_args, load_args
from logger import create_logger
from dataset import dataset_factory


def main():
    test_args = create_test_args()
    args = load_args(test_args.saved_dir)

    with open(test_args.test_data, 'r') as outfile:
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

    if not os.path.isdir(baseline_dir):
        os.makedirs(baseline_dir)

    logger = create_logger(save_dir, 'test', 'test.log')
    logger.info("[Test Args]: {}".format(str(test_args)))
    logger.info("[Train Args]: {}".format(str(args)))
    logger.info("[Save Dir]: {}".format(save_dir))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))

    # load model
    in_channel = 1 if args.dataset == "ucsd" else 3
    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    elif args.dataset == "ucsd":
        load_weight = True
    else:
        load_weight = False
    model = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate, in_channels=in_channel)
    checkpoint = torch.load(os.path.join(test_args.saved_dir, "model_best.pth"), torch.device(device))
    model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
    try:
        best_prec1 = checkpoint['val']
    except KeyError:
        logger.info("No Key: val")
    # multi gpu
    if torch.cuda.device_count() > 1:
        logger.info("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    # validate per scene
    for k in test_data.keys():
        # dataloader
        _, test_dataset = dataset_factory(args, None, test_data[k])
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.workers,
                                                      prefetch_factor=args.prefetch,
                                                      pin_memory=True,
                                                      shuffle=False)

        # load parameters
        static_param, temperature_param, beta_param, delta_param = get_param(test_args, subdir, k)

        # create result csv file per scene
        save_dir_scene = os.path.join(save_dir, str(k))
        if not os.path.exists(save_dir_scene):
            os.makedirs(save_dir_scene)
        with open(os.path.join(save_dir_scene, "result_per_image.csv"), mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "target_sum", "pred_sum", "mae", "pix_mae", "pix_rmse"])

        # load staticff
        staticff = None
        if test_args.staticff == 1:
            # print(test_data.keys())
            staticff_file = os.path.join(os.path.dirname(test_data[k][0]["gt"]), "staticff.npz")
            staticff = np.load(staticff_file)["x"]

        # validate
        results = validate(test_args, test_dataloader, model, staticff, device,
                           baseline_dir=baseline_dir, subdir=save_dir_scene, tested_num=test_args.scene_num,
                           static_param=static_param, temperature_param=temperature_param,
                           beta_param=beta_param, delta_param=delta_param)
        logger.info("[Scene {}] MAE: {:.3f}, RMSE: {:.3f}, pix MAE: {:.5f}, pix RMSE: {:.5f}"
                    .format(k, results["mae"], results["rmse"], results["pix_mae_val"], results["pix_rmse_val"]))
        # save results
        # print(results)
        np.savez_compressed(os.path.join(save_dir_scene, 'result.npz'), **results)
        # res = np.load(os.path.join(save_dir_scene, 'result.npz'))
        # with open(os.path.join(save_dir_scene, 'result.json'), mode='w') as f:
        #     json.dump(results, f, indent=4)

def get_param(args, subdir, scene):
    if subdir == 'dynamic_static':
        param_path = os.path.join(args.saved_dir, subdir, scene, 'ff_param.csv')
    else:
        param_path = os.path.join(args.saved_dir, subdir, scene, 'ff_param_pix.csv' if args.pix==1 else 'ff_param.csv')
    # print(param_path, os.path.isfile(param_path))
    if os.path.isfile(param_path):
        with open(param_path) as f:
            reader = csv.reader(f)
            params = [row for row in reader]
            static_param, temperature_param, beta_param, delta_param = float(params[0][0]), float(params[0][1]), float(params[0][2]), float(params[0][3])
    else:
        static_param, temperature_param, beta_param, delta_param = None, None, None, None

    return static_param, temperature_param, beta_param, delta_param

def validate(test_args, val_loader, model, staticff, device, baseline_dir=None, subdir=None, tested_num=None, static_param=1.0, temperature_param=1.0, beta_param=0.5, delta_param=0.5, mode=""):
    model.eval()

    var = 0
    mae = 0
    rmse = 0
    pix_mae = []
    pix_rmse = []
    pix_var = []

    pred_scene = []
    gt = []

    past_output = None

    for i, (prev_img, img, post_img, target) in enumerate(val_loader):
        if tested_num is not None:
            if i < tested_num:
                continue

        if os.path.isfile(os.path.join(subdir, "{}.npz".format(i))):
            pred = np.load(os.path.join(subdir, "{}.npz".format(i)))["x"]
        else:
            with torch.no_grad():
                prev_img = prev_img.to(device)
                img = img.to(device)
                prev_flow = model(prev_img, img)
                prev_flow_inverse = model(img, prev_img)

            mask_boundry = torch.zeros(prev_flow.shape[2:])
            mask_boundry[0,:] = 1.0
            mask_boundry[-1,:] = 1.0
            mask_boundry[:,0] = 1.0
            mask_boundry[:,-1] = 1.0

            mask_boundry = mask_boundry.to(device)

            reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
            reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

            overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)

            pred = overall.detach().numpy().copy()
        target = target.detach().numpy().copy()

        # debug_hist_path = os.path.join("/path/to/debug", "{}_hist.png".format(i))
        # debug_hist = {}
        # debug_hist["original"] = pred
        if test_args.staticff == 1:
            # debug_hist["staticff"] = staticff
            # print(static_param, staticff.shape)
            pred *= static_param*staticff
            # debug_hist["pred_with_staticff"] = pred

        pred_g = gaussian_filter(pred, 3)
        if test_args.dynamicff == 1 and past_output is not None:
            # debug_hist["t-1_dynamicff"] = past_output
            d_t_prev = gaussian_filter(past_output, 3)
            # debug_hist["t-1_dynamicff_g"] = d_t_prev
            past_output = beta_param * pred_g + (1 - delta_param) * d_t_prev
            past_output_g = gaussian_filter(past_output, 3)
            height, width = past_output_g.shape
            deno = np.sum(np.exp(past_output_g/temperature_param)) / (height * width)
            nume = np.exp(past_output_g/temperature_param)
            dynamicff = nume / deno
            # debug_hist["t_dynamicff"] = dynamicff
            pred *= dynamicff
            # debug_hist["pred_with_dynamicff"] = pred

        if test_args.dynamicff == 1 and past_output is None:
            past_output = beta_param * pred_g

        pred_sum = np.sum(pred)
        if not os.path.isfile(os.path.join(subdir, "{}.npz".format(i))) and baseline_dir == os.path.dirname(subdir):
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            np.savez_compressed(os.path.join(subdir, "{}.npz".format(i)), x=overall.detach().numpy().copy())
            print(subdir, i, "saved")

        pix_mae.append(np.nanmean(np.abs(target.squeeze()-pred)))
        pix_rmse.append(np.sqrt(np.nanmean(np.square(target.squeeze()-pred))))
        pix_var.append(np.var(pred))

        # print(np.sum(target), pred_sum)
        pred_scene.append(pred_sum)
        gt.append(np.sum(target))

        with open(os.path.join(subdir, "result_per_image.csv"), mode="a") as f:
            writer = csv.writer(f)
            target_sum = np.sum(target)
            mae_per_data = abs(target_sum - pred_sum)
            pix_mae_per_data = np.nanmean(np.abs(target.squeeze()-pred))
            pix_rmse_per_data = np.sqrt(np.nanmean(np.square(target.squeeze()-pred)))
            writer.writerow([i, target_sum, pred_sum, mae_per_data, pix_mae_per_data, pix_rmse_per_data])

        """
        debug_ = False
        if debug_ and i%50 == 1:
            hist_nums = len(debug_hist.keys())
            fig, axes = plt.subplots(2, hist_nums, figsize=(hist_nums*2, 4), tight_layout=True)
            for j, k in enumerate(debug_hist.keys()):
                # print(j%3)
                if hist_nums >= 2:
                    axes[0, j].imshow(debug_hist[k])
                    axes[0, j].set_title(k)
                    axes[1, j].hist(debug_hist[k].ravel())
                    axes[1, j].set_ylim(0, 200)
                else:
                    axes[0].imshow(debug_hist[k])
                    axes[0].set_title(k)
                    axes[1].hist(debug_hist[k].ravel())
                    axes[1].set_ylim(0, 200)

            fig.savefig(debug_hist_path, dpi=300)
        """


    #print("pred: {}".format(np.array(pred)))
    #print("target: {}".format(np.array(gt)))
    abs_diff = np.abs(np.array(pred_scene)-np.array(gt))
    mae = np.nanmean(abs_diff)
    mae_std = np.nanstd(abs_diff)
    var = np.var(np.array(pred_scene))
    squared_diff = np.square(np.array(pred_scene)-np.array(gt))
    rmse = np.sqrt(np.array(np.nanmean(squared_diff)))
    rmse_std = np.sqrt(np.array(np.nanstd(squared_diff)))
    pix_mae_val = np.nanmean(np.array(pix_mae))
    pix_mae_val_std = np.nanstd(np.array(pix_mae))
    pix_rmse_val = np.nanmean(np.array(pix_rmse))
    pix_rmse_val_std = np.nanstd(np.array(pix_rmse))
    pix_var_val = np.nanmean(np.array(pix_var))

    results = {
        "mae": mae,
        "mae_std": mae_std,
        "rmse": rmse,
        "rmse_std": rmse_std,
        "var": var,
        "pix_mae_val": pix_mae_val,
        "pix_mae_val_std": pix_mae_val_std,
        "pix_rmse_val": pix_rmse_val,
        "pix_rmse_val_std": pix_rmse_val_std,
        "pix_var_val": pix_var_val
    }

    return results

if __name__ == "__main__":
    main()
