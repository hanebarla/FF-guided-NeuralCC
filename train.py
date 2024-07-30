import os
import json
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from model import CANNet2s, SimpleCNN
from utils import save_checkpoint, fix_model_state_dict
from dataset import dataset_factory
from argument import create_train_args
from logger import create_logger


def optimizer_factory(model, args):
    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     args.lr, 
                                     eps=args.opt_eps,
                                     betas=args.opt_betas,
                                     weight_decay=args.decay)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                     args.lr, 
                                     weight_decay=args.decay, 
                                     amsgrad=True)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), 
                                    args.lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.decay)
    else:
        raise NotImplementedError(args.opt)

    scheduler = None
    if args.lr_sch == "cosine":
        scheduler = CosineLRScheduler(optimizer, t_initial=args.lr_t_initial,
                                      lr_min=args.lr_min, warmup_t=args.warmup_t, 
                                      warmup_lr_init=args.warmup_lr_init, warmup_prefix=True)
    elif args.lr_sch == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=args.gamma)
    
    return optimizer, scheduler


def main():
    args = create_train_args()

    # get data mode
    conditions = args.train_data.split('_')
    data_mode = conditions[1]
    args.mode = data_mode

    # get save dir
    if args.dataset == "CrowdFlow":
        save_dir = os.path.join(args.exp, "{}".format(conditions[2]), "{}_{}_{}".format(data_mode, args.activate, args.penalty))
    else:
        save_dir = os.path.join(args.exp, args.dataset, "{}_{}_{}".format(data_mode, args.activate, args.penalty))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = create_logger(save_dir, 'train', 'train.log')
    logger.info("[Args]: {}".format(str(args)))
    logger.info("[Save Dir]: {}".format(save_dir))
    with open(os.path.join(save_dir, "command.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    with open(args.train_data, 'r') as outfile:
        train_data = json.load(outfile)
    with open(args.val_data, 'r') as outfile:
        val_data = json.load(outfile)

    new_train_data = []
    new_val_data = []
    for data in train_data.values():
        new_train_data.extend(data)
    for data in val_data.values():
        new_val_data.extend(data)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))

    # Seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    train_dataset, val_dataset = dataset_factory(args, new_train_data, new_val_data)
    # print(len(train_dataset), len(val_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               prefetch_factor=args.prefetch,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             prefetch_factor=args.prefetch,
                                             pin_memory=True)

    # Model
    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    elif args.dataset == "ucsd":
        load_weight = True
    else:
        load_weight = False

    in_channel = 1 if args.dataset == "ucsd" else 3
    if args.model == "CAN":
        model = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate, in_channels=in_channel)
    elif args.model == "SimpleCNN":
        model = SimpleCNN()

    train_time = 0
    best_prec1 = 1000

    # multiple GPUs
    if torch.cuda.device_count() > 1:
        logger.info("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Train resume
    if os.path.isfile(os.path.join(save_dir, 'checkpoint.pth')):
        checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
        modelbest = torch.load(os.path.join(save_dir, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_prec1 = modelbest['val']
        train_time = modelbest['time']
        logger.info("Train resumed: {} epoch".format(args.start_epoch))
        logger.info("best val: {}".format(best_prec1))

    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer, scheduler = optimizer_factory(model, args)
    logger.info("optimizer: {}, scheduler: {}".format(optimizer, scheduler))

    # Speed up
    torch.backends.cudnn.benchmark = True

    # Train Model
    for epoch in range(args.start_epoch, args.epochs):
        logger.info("Epoch: [{}/{}]".format(epoch, args.epochs))
        # train for one epoch
        start_epoch_time = time.time()
        model.train()
        train_epoch(args, train_loader, model, criterion, optimizer, device, logger)
        end_epoch_time = time.time()

        epoch_time = end_epoch_time - start_epoch_time
        train_time += epoch_time

        # evaluate on validation set
        model.eval()
        prec1 = validate(args, val_loader, model, criterion,device, logger)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        logger.info('best MAE {mae:.3f}, Time {time}'.format(mae=best_prec1, time=train_time))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'val': prec1.item(),
            'epoch': epoch,
            'time': train_time
        }, is_best,
            filename=os.path.join(save_dir, 'checkpoint.pth'),
            bestname=os.path.join(save_dir, 'model_best.pth'))

        if args.lr_sch == 'cosine':
            scheduler.step(epoch)
        elif args.lr_sch == 'multistep':
            scheduler.step()


def train_epoch(args, train_loader, model, criterion, optimizer, device, logger):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    on_dloss = 1 if args.penalty != 0 else 0
    mae = 0
    model.train()
    end = time.time()

    for i, (prev_img, img, post_img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        prev_img = prev_img.to(device)
        img = img.to(device)
        post_img = post_img.to(device)
        target = target.to(device)

        prev_flow = model(prev_img, img)
        post_flow = model(img, post_img)

        prev_flow_inverse = model(img, prev_img)
        post_flow_inverse = model(post_img, img)


        # mask the boundary locations where people can move in/out between regions outside image plane
        mask_boundry = torch.zeros(prev_flow.shape[2:]).to(device)
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

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


        loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+ 1*(loss_prev_consistency+loss_post_consistency)

        # penalty term L_follow
        if on_dloss:
            loss_prev_direct = criterion(prev_flow[0,0,1:,1:], prev_flow[0,0,1:,1:])+criterion(prev_flow[0,1,1:,:], prev_flow[0,1,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow[0,2,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow[0,3,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow[0,5,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow[0,6,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow[0,7,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow[0,8,1:,1:])
            loss_post_direct = criterion(post_flow[0,0,1:,1:], post_flow[0,0,1:,1:])+criterion(post_flow[0,1,1:,:], post_flow[0,1,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow[0,2,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow[0,3,:,:-1])+criterion(post_flow[0,4,:,:], post_flow[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow[0,5,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow[0,6,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow[0,7,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow[0,8,1:,1:])
            loss_prev_inv_direct = criterion(prev_flow_inverse[0,0,1:,1:], prev_flow_inverse[0,0,1:,1:])+criterion(prev_flow_inverse[0,1,1:,:], prev_flow_inverse[0,1,:-1,:])+criterion(prev_flow_inverse[0,2,1:,:-1], prev_flow_inverse[0,2,:-1,1:])+criterion(prev_flow_inverse[0,3,:,1:], prev_flow_inverse[0,3,:,:-1])+criterion(prev_flow_inverse[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow_inverse[0,5,:,:-1], prev_flow_inverse[0,5,:,1:])+criterion(prev_flow_inverse[0,6,:-1,1:], prev_flow_inverse[0,6,1:,:-1])+criterion(prev_flow_inverse[0,7,:-1,:], prev_flow_inverse[0,7,1:,:])+criterion(prev_flow_inverse[0,8,:-1,:-1], prev_flow_inverse[0,8,1:,1:])
            loss_post_inv_direct = criterion(post_flow_inverse[0,0,1:,1:], post_flow_inverse[0,0,1:,1:])+criterion(post_flow_inverse[0,1,1:,:], post_flow_inverse[0,1,:-1,:])+criterion(post_flow_inverse[0,2,1:,:-1], post_flow_inverse[0,2,:-1,1:])+criterion(post_flow_inverse[0,3,:,1:], post_flow_inverse[0,3,:,:-1])+criterion(post_flow_inverse[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow_inverse[0,5,:,:-1], post_flow_inverse[0,5,:,1:])+criterion(post_flow_inverse[0,6,:-1,1:], post_flow_inverse[0,6,1:,:-1])+criterion(post_flow_inverse[0,7,:-1,:], post_flow_inverse[0,7,1:,:])+criterion(post_flow_inverse[0,8,:-1,:-1], post_flow_inverse[0,8,1:,1:])

            loss += args.penalty * (loss_prev_direct + loss_post_direct + loss_prev_inv_direct + loss_post_inv_direct)
        
        loss *= 3.0
        # MAE
        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0)
        mae += abs(overall.data.sum()-target.sum())

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Batch: [{}/{}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        .format(i, len(train_loader), batch_time=batch_time,
                         data_time=data_time, loss=losses))

    mae = mae / len(train_loader)
    logger.info('[Train MAE] {mae:.3f}'.format(mae=mae))
    logger.info('[Train Loss] {loss:.3f} '.format(loss=losses.avg))

def validate(args, val_loader, model, criterion, device, logger):
    logger.info('begin val')
    model.eval()
    losses = AverageMeter()
    mae = 0

    on_dloss = 1 if args.penalty != 0 else 0

    for i, (prev_img, img, post_img, target) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        with torch.no_grad():
            prev_img = prev_img.to(device)
            img = img.to(device)
            post_img = post_img.to(device)
            target = target.to(device)

            prev_flow = model(prev_img,img)
            prev_flow_inverse = model(img,prev_img)

            post_flow = model(img, post_img)
            post_flow_inverse = model(post_img, img)

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


            loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+ 1*(loss_prev_consistency+loss_post_consistency)

            if on_dloss:
                loss_prev_direct = criterion(prev_flow[0,0,1:,1:], prev_flow[0,0,1:,1:])+criterion(prev_flow[0,1,1:,:], prev_flow[0,1,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow[0,2,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow[0,3,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow[0,5,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow[0,6,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow[0,7,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow[0,8,1:,1:])
                loss_post_direct = criterion(post_flow[0,0,1:,1:], post_flow[0,0,1:,1:])+criterion(post_flow[0,1,1:,:], post_flow[0,1,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow[0,2,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow[0,3,:,:-1])+criterion(post_flow[0,4,:,:], post_flow[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow[0,5,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow[0,6,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow[0,7,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow[0,8,1:,1:])
                loss_prev_inv_direct = criterion(prev_flow_inverse[0,0,1:,1:], prev_flow_inverse[0,0,1:,1:])+criterion(prev_flow_inverse[0,1,1:,:], prev_flow_inverse[0,1,:-1,:])+criterion(prev_flow_inverse[0,2,1:,:-1], prev_flow_inverse[0,2,:-1,1:])+criterion(prev_flow_inverse[0,3,:,1:], prev_flow_inverse[0,3,:,:-1])+criterion(prev_flow_inverse[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow_inverse[0,5,:,:-1], prev_flow_inverse[0,5,:,1:])+criterion(prev_flow_inverse[0,6,:-1,1:], prev_flow_inverse[0,6,1:,:-1])+criterion(prev_flow_inverse[0,7,:-1,:], prev_flow_inverse[0,7,1:,:])+criterion(prev_flow_inverse[0,8,:-1,:-1], prev_flow_inverse[0,8,1:,1:])
                loss_post_inv_direct = criterion(post_flow_inverse[0,0,1:,1:], post_flow_inverse[0,0,1:,1:])+criterion(post_flow_inverse[0,1,1:,:], post_flow_inverse[0,1,:-1,:])+criterion(post_flow_inverse[0,2,1:,:-1], post_flow_inverse[0,2,:-1,1:])+criterion(post_flow_inverse[0,3,:,1:], post_flow_inverse[0,3,:,:-1])+criterion(post_flow_inverse[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow_inverse[0,5,:,:-1], post_flow_inverse[0,5,:,1:])+criterion(post_flow_inverse[0,6,:-1,1:], post_flow_inverse[0,6,1:,:-1])+criterion(post_flow_inverse[0,7,:-1,:], post_flow_inverse[0,7,1:,:])+criterion(post_flow_inverse[0,8,:-1,:-1], post_flow_inverse[0,8,1:,1:])

                loss += args.penalty * (loss_prev_direct + loss_post_direct + loss_prev_inv_direct + loss_post_inv_direct)
            loss *= 3.0

            losses.update(loss.item(), img.size(0))
            mae += abs(overall.data.sum()-target.sum())

    mae = mae/len(val_loader)
    logger.info('[Val MAE] {mae:.3f} '.format(mae=mae))
    logger.info('[Val Loss] {loss:.3f} '.format(loss=losses.avg))

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
