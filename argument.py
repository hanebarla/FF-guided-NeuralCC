import os
import json
import argparse

def create_train_args():
    parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

    # data settings
    parser.add_argument('train_data', help='path to train json')
    parser.add_argument('val_data', metavar='VAL', help='path to val json')
    parser.add_argument('--dataset', default="FDST")
    parser.add_argument('--exp', default='./exp/')
    parser.add_argument('--mode', default='once', choices=["add", "once"])  # once or add
    parser.add_argument('--batch_size', default=1, type=int)

    # model settings
    parser.add_argument('--model', default="CAN")
    parser.add_argument('--activate', default="leaky")
    parser.add_argument('--bn', default=0, type=int)
    parser.add_argument('--do_rate', default=0.0, type=float)
    parser.add_argument('--pretrained', default=0, type=int)

    # training settings
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--penalty', default=0, type=float)
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)

    # misc
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    return args

def create_test_args():
    parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

    # data settings
    parser.add_argument('test_data', help='path to test json')
    parser.add_argument('--dataset', default="FDST")
    parser.add_argument('--exp', default='./exp/')
    parser.add_argument('--mode', default='once', choices=["add", "once"])  # once or add
    parser.add_argument('--batch_size', default=1, type=int)

    # load weights
    parser.add_argument('--saved_dir', default="/path/to/saved_dir")

    # floor-field settings
    parser.add_argument('--staticff', default=0, type=int)
    parser.add_argument('--dynamicff', default=0, type=int)

    # misc
    parser.add_argument('--workers', default=8, type=int)
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
