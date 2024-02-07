import os
import json
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from model import CANNet2s, SimpleCNN
from utils import save_checkpoint, fix_model_state_dict
from dataset import dataset_factory
from argument import create_train_args
from logger import create_logger


def optimizer_factory(model, args):
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay, amsgrad=True)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
    else:
        raise NotImplementedError(args.opt)
    return optimizer


def main():
    args = create_train_args()
    save_dir = os.path.join(args.exp, args.dataset, args.data_mode, args.penalty)

    logger = create_logger(save_dir, 'train', 'train.log')
    logger.info("[Args]: {}".format(str(args)))
    logger.info("[Save Dir]: {}".format(save_dir))

    if args.dataset == "FDST":
        args.print_freq = 400
        with open(args.train_json, 'r') as outfile:
            train_list = json.load(outfile)
        with open(args.val_json, 'r') as outfile:
            val_list = json.load(outfile)
    elif args.dataset == "CrowdFlow":
        args.print_freq = 200
        train_list = args.train_json
        val_list = args.val_json
    elif args.dataset == "venice":
        args.print_freq = 10
        train_list = args.train_json
        val_list = args.val_json
    elif args.dataset == "CityStreet":
        args.print_freq = 50
        train_list = "/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/train/"
        val_list = "/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/"
    else:
        raise ValueError

    if not os.path.exists(args.savefolder):
        os.makedirs(args.savefolder)
    print(args.savefolder)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))

    # Seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    else:
        load_weight = False
    if args.trainmodel == "CAN":
        model = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
    elif args.trainmodel == "SimpleCNN":
        model = SimpleCNN()

    train_time = 0
    best_prec1 = 100

    # Train resume
    if os.path.isfile(os.path.join(save_dir, 'checkpoint.pth.tar')):
        checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        modelbest = torch.load(os.path.join(save_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_prec1 = modelbest['val']
        train_time = modelbest['time']
        logger.info("Train resumed: {} epoch".format(args.start_epoch))
        logger.info("best val: {}".format(best_prec1))

    # multiple GPUs
    if torch.cuda.device_count() > 1:
        logger.info("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optimizer_factory(model, args)

    # Speed up
    torch.backends.cudnn.benchmark = True

    # Train Model
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        start_epoch_time = time.time()
        train_epoch(train_list, model, criterion, optimizer, epoch, device)
        end_epoch_time = time.time()

        epoch_time = end_epoch_time - start_epoch_time
        train_time += epoch_time

        # evaluate on validation set
        prec1 = validate(val_list, model, criterion, device)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        logger.info('best MAE {mae:.3f}, Time {time}'.format(mae=best_prec1, time=train_time))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'val': prec1.item(),
            'epoch': epoch,
            'time': train_time
        }, is_best,
            filename=os.path.join(args.savefolder, 'checkpoint.pth.tar'),
            bestname=os.path.join(args.savefolder, 'model_best.pth.tar'))


def train_epoch(train_list, model, criterion, optimizer, epoch, device):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_dataset = dataset_factory(train_list, args, mode="train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    mae = 0
    model.train()
    end = time.time()

    for i, (prev_img, img, post_img, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        prev_img = prev_img.to(device, dtype=torch.float)
        prev_img = Variable(prev_img)

        img = img.to(device, dtype=torch.float)
        img = Variable(img)

        post_img = post_img.to(device, dtype=torch.float)
        post_img = Variable(post_img)

        prev_flow = model(prev_img, img)
        post_flow = model(img, post_img)

        prev_flow_inverse = model(img, prev_img)
        post_flow_inverse = model(post_img, img)

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        # mask the boundary locations where people can move in/out between regions outside image plane
        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
        reconstruction_from_post = torch.sum(post_flow[0,:9,:,:],dim=0)+post_flow[0,9,:,:]*mask_boundry
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
        reconstruction_from_post_inverse = F.pad(post_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0))+post_flow_inverse[0,4,:,:]+F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+post_flow_inverse[0,9,:,:]*mask_boundry


        # prev_density_reconstruction = torch.sum(prev_flow[0,:9,:,:],dim=0)+prev_flow[0,9,:,:]*mask_boundry
        # prev_density_reconstruction_inverse = F.pad(prev_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow_inverse[0,3,:,1:],(0,1,0,0))+prev_flow_inverse[0,4,:,:]+F.pad(prev_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+prev_flow_inverse[0,9,:,:]*mask_boundry
        # post_density_reconstruction_inverse = torch.sum(post_flow_inverse[0,:9,:,:],dim=0)+post_flow_inverse[0,9,:,:]*mask_boundry
        # post_density_reconstruction = F.pad(post_flow[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow[0,1,1:,:],(0,0,0,1))+F.pad(post_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow[0,3,:,1:],(0,1,0,0))+post_flow[0,4,:,:]+F.pad(post_flow[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow[0,8,:-1,:-1],(1,0,1,0))+post_flow[0,9,:,:]*mask_boundry

        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)

        # cycle consistency
        loss_prev_consistency = criterion(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1])+criterion(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])
        loss_post_consistency = criterion(post_flow[0,0,1:,1:], post_flow_inverse[0,8,:-1,:-1])+criterion(post_flow[0,1,1:,:], post_flow_inverse[0,7,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow_inverse[0,6,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow_inverse[0,5,:,:-1])+criterion(post_flow[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow_inverse[0,3,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow_inverse[0,2,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow_inverse[0,1,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow_inverse[0,0,1:,1:])


        loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+loss_prev_consistency+loss_post_consistency

        # direct loss
        if dloss_on:
            loss_prev_direct = criterion(prev_flow[0,0,1:,1:], prev_flow[0,0,1:,1:])+criterion(prev_flow[0,1,1:,:], prev_flow[0,1,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow[0,2,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow[0,3,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow[0,5,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow[0,6,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow[0,7,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow[0,8,1:,1:])
            loss_post_direct = criterion(post_flow[0,0,1:,1:], post_flow[0,0,1:,1:])+criterion(post_flow[0,1,1:,:], post_flow[0,1,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow[0,2,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow[0,3,:,:-1])+criterion(post_flow[0,4,:,:], post_flow[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow[0,5,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow[0,6,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow[0,7,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow[0,8,1:,1:])
            loss_prev_inv_direct = criterion(prev_flow_inverse[0,0,1:,1:], prev_flow_inverse[0,0,1:,1:])+criterion(prev_flow_inverse[0,1,1:,:], prev_flow_inverse[0,1,:-1,:])+criterion(prev_flow_inverse[0,2,1:,:-1], prev_flow_inverse[0,2,:-1,1:])+criterion(prev_flow_inverse[0,3,:,1:], prev_flow_inverse[0,3,:,:-1])+criterion(prev_flow_inverse[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow_inverse[0,5,:,:-1], prev_flow_inverse[0,5,:,1:])+criterion(prev_flow_inverse[0,6,:-1,1:], prev_flow_inverse[0,6,1:,:-1])+criterion(prev_flow_inverse[0,7,:-1,:], prev_flow_inverse[0,7,1:,:])+criterion(prev_flow_inverse[0,8,:-1,:-1], prev_flow_inverse[0,8,1:,1:])
            loss_post_inv_direct = criterion(post_flow_inverse[0,0,1:,1:], post_flow_inverse[0,0,1:,1:])+criterion(post_flow_inverse[0,1,1:,:], post_flow_inverse[0,1,:-1,:])+criterion(post_flow_inverse[0,2,1:,:-1], post_flow_inverse[0,2,:-1,1:])+criterion(post_flow_inverse[0,3,:,1:], post_flow_inverse[0,3,:,:-1])+criterion(post_flow_inverse[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow_inverse[0,5,:,:-1], post_flow_inverse[0,5,:,1:])+criterion(post_flow_inverse[0,6,:-1,1:], post_flow_inverse[0,6,1:,:-1])+criterion(post_flow_inverse[0,7,:-1,:], post_flow_inverse[0,7,1:,:])+criterion(post_flow_inverse[0,8,:-1,:-1], post_flow_inverse[0,8,1:,1:])

            loss += float(args.myloss) *(loss_prev_direct + loss_post_direct + loss_prev_inv_direct + loss_post_inv_direct)

        # MAE
        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)
        mae += abs(overall.data.sum()-target.sum())

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        del prev_img
        del img
        del post_img
        del target

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
                f.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t\n'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    mae = mae/len(train_loader)
    print(' * Train MAE {mae:.3f} '
              .format(mae=mae))
    print(' * Train Loss {loss:.3f} '
              .format(loss=losses.avg))
    with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
        f.write('Train MAE:{mae:.3f} \nTrain Loss:{loss:.3f} \n\n'
              .format(mae=mae, loss=losses.avg))

def validate(val_list, model, criterion, device):
    print ('begin val')
    with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
        f.write('begin val\n')
    val_dataset = dataset_factory(val_list, args, mode="val")
    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1)

    model.eval()

    losses = AverageMeter()
    mae = 0

    for i,(prev_img, img, post_img, target ) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        with torch.no_grad():
            prev_img = prev_img.to(device, dtype=torch.float)
            prev_img = Variable(prev_img)

            img = img.to(device, dtype=torch.float)
            img = Variable(img)

            post_img = post_img.to(device, dtype=torch.float)
            post_img = Variable(post_img)

            prev_flow = model(prev_img,img)
            prev_flow_inverse = model(img,prev_img)

            post_flow = model(img, post_img)
            post_flow_inverse = model(post_img, img)

            target = target.type(torch.FloatTensor)[0].to(device, dtype=torch.float)
            target = Variable(target)

            mask_boundry = torch.zeros(prev_flow.shape[2:])
            mask_boundry[0,:] = 1.0
            mask_boundry[-1,:] = 1.0
            mask_boundry[:,0] = 1.0
            mask_boundry[:,-1] = 1.0

            mask_boundry = Variable(mask_boundry.cuda())


            reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
            reconstruction_from_post = torch.sum(post_flow[0,:9,:,:],dim=0)+post_flow[0,9,:,:]*mask_boundry
            reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
            reconstruction_from_post_inverse = F.pad(post_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0))+post_flow_inverse[0,4,:,:]+F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+post_flow_inverse[0,9,:,:]*mask_boundry

            overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)

            loss_prev_flow = criterion(reconstruction_from_prev, target)
            loss_post_flow = criterion(reconstruction_from_post, target)
            loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
            loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)

            # cycle consistency
            loss_prev_consistency = criterion(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1])+criterion(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])
            loss_post_consistency = criterion(post_flow[0,0,1:,1:], post_flow_inverse[0,8,:-1,:-1])+criterion(post_flow[0,1,1:,:], post_flow_inverse[0,7,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow_inverse[0,6,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow_inverse[0,5,:,:-1])+criterion(post_flow[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow_inverse[0,3,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow_inverse[0,2,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow_inverse[0,1,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow_inverse[0,0,1:,1:])


            loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+loss_prev_consistency+loss_post_consistency

            if dloss_on:
                loss_prev_direct = criterion(prev_flow[0,0,1:,1:], prev_flow[0,0,1:,1:])+criterion(prev_flow[0,1,1:,:], prev_flow[0,1,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow[0,2,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow[0,3,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow[0,5,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow[0,6,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow[0,7,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow[0,8,1:,1:])
                loss_post_direct = criterion(post_flow[0,0,1:,1:], post_flow[0,0,1:,1:])+criterion(post_flow[0,1,1:,:], post_flow[0,1,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow[0,2,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow[0,3,:,:-1])+criterion(post_flow[0,4,:,:], post_flow[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow[0,5,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow[0,6,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow[0,7,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow[0,8,1:,1:])
                loss_prev_inv_direct = criterion(prev_flow_inverse[0,0,1:,1:], prev_flow_inverse[0,0,1:,1:])+criterion(prev_flow_inverse[0,1,1:,:], prev_flow_inverse[0,1,:-1,:])+criterion(prev_flow_inverse[0,2,1:,:-1], prev_flow_inverse[0,2,:-1,1:])+criterion(prev_flow_inverse[0,3,:,1:], prev_flow_inverse[0,3,:,:-1])+criterion(prev_flow_inverse[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow_inverse[0,5,:,:-1], prev_flow_inverse[0,5,:,1:])+criterion(prev_flow_inverse[0,6,:-1,1:], prev_flow_inverse[0,6,1:,:-1])+criterion(prev_flow_inverse[0,7,:-1,:], prev_flow_inverse[0,7,1:,:])+criterion(prev_flow_inverse[0,8,:-1,:-1], prev_flow_inverse[0,8,1:,1:])
                loss_post_inv_direct = criterion(post_flow_inverse[0,0,1:,1:], post_flow_inverse[0,0,1:,1:])+criterion(post_flow_inverse[0,1,1:,:], post_flow_inverse[0,1,:-1,:])+criterion(post_flow_inverse[0,2,1:,:-1], post_flow_inverse[0,2,:-1,1:])+criterion(post_flow_inverse[0,3,:,1:], post_flow_inverse[0,3,:,:-1])+criterion(post_flow_inverse[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow_inverse[0,5,:,:-1], post_flow_inverse[0,5,:,1:])+criterion(post_flow_inverse[0,6,:-1,1:], post_flow_inverse[0,6,1:,:-1])+criterion(post_flow_inverse[0,7,:-1,:], post_flow_inverse[0,7,1:,:])+criterion(post_flow_inverse[0,8,:-1,:-1], post_flow_inverse[0,8,1:,1:])

                loss += float(args.myloss) *(loss_prev_direct + loss_post_direct + loss_prev_inv_direct + loss_post_inv_direct)

            target = target.type(torch.FloatTensor)

            losses.update(loss.item(), img.size(0))
            mae += abs(overall.data.sum()-target.sum())

        del prev_img
        del img
        del target

    mae = mae/len(val_loader)
    print(' * Val MAE {mae:.3f} '
              .format(mae=mae))
    print(' * Val Loss {loss:.3f} '
              .format(loss=losses.avg))
    with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
        f.write('Val MAE:{mae:.3f} \nVal Loss:{loss:.3f} \n\n'
              .format(mae=mae, loss=losses.avg))

    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
