import os
import json
import argparse

def create_train_args():
    parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

    # data settings
    parser.add_argument('train_data', help='path to train json')
    parser.add_argument('val_data', metavar='VAL', help='path to val json')
    parser.add_argument('--dataset', default="FDST")
    parser.add_argument('--mode', default='once', choices=["add", "once"])  # once or add
    parser.add_argument('--batch_size', default=1, type=int)

    # model settings
    parser.add_argument('--model', default="CAN")
    parser.add_argument('--activate', default="relu")
    parser.add_argument('--bn', default=0, type=int)
    parser.add_argument('--do_rate', default=0.0, type=float)
    parser.add_argument('--pretrained', default=0, type=int)

    # training settings
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--penalty', default=0, type=float)
    parser.add_argument('--opt', default="adamw")
    parser.add_argument('--decay', default=5e-4, type=float)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--lr_sch', default="multistep")
    ## cosine sch
    parser.add_argument('--lr_t_initial', default=100, type=int)
    parser.add_argument('--lr_min', default=1e-5, type=float)
    parser.add_argument('--warmup_t', default=3, type=int)
    parser.add_argument('--warmup_lr_init', default=1e-5, type=float)
    ## multistep sch
    parser.add_argument('--lr_steps', default=[20, 40], nargs='*', type=int)  # 60, 120, 160
    parser.add_argument('--gamma', default=0.2, type=float)

    # misc
    parser.add_argument('--prefetch', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--exp', default='./exp/')

    args = parser.parse_args()

    return args

def create_test_args():
    parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

    # data settings
    parser.add_argument('test_data', help='path to test json')
    parser.add_argument('--dataset', default="FDST")
    parser.add_argument('--mode', default='once', choices=["add", "once"])  # once or add
    parser.add_argument('--batch_size', default=1, type=int)

    # load weights
    parser.add_argument('--saved_dir', default="/path/to/saved_dir")

    # tune setting
    parser.add_argument('--pix', default=0, type=int)
    parser.add_argument('--scene_num', default=50, type=int)

    # floor-field settings
    parser.add_argument('--staticff', default=0, type=int)
    parser.add_argument('--dynamicff', default=0, type=int)

    # misc
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--prefetch', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    return args

def save_args(save_dir, args):
    with open(os.path.join(save_dir, "command.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

def load_args(saved_dir):
    with open(os.path.join(saved_dir, "command.json"), 'r') as f:
        commands = json.load(f)
    return argparse.Namespace(**commands)
