import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from engine_classification import Engine as EngineClassification
from engine_segmentation import Engine as EngineSegmentation
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Adam-v2', add_help=False)
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--weights', default=None, type=str, help='Path for pre-trained weights')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--task', default="classification", type=str, choices=["classification", "segmentation"], help='Downstream task')
    parser.add_argument('--mode', default="full_transfer", type=str, choices=["full_transfer", "fewshot_5", "fewshot_10"], help='Mode for fine-tuning')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--dataset', default=None, type=str, help='Dataset to finetune on')
    parser.add_argument('--datasets_config', default=None, type=str, help='path to datasets config file')
    parser.add_argument('--pretrain', default="adam", type=str, choices=['adam', 'imagenet'], help='Pre-trained weights to use')
    parser.add_argument('--epochs', default=200, type=int, help='Total epochs to train the model')
    parser.add_argument('--patience', default=20, type=int, help='Patience for early stopping')
    parser.add_argument('--start_trial', default=1, type=int, help='Start Trial')
    parser.add_argument('--trials', default=5, type=int, help='Number of Trials')
    parser.add_argument('--train_toggle', default=True, type=utils.bool_flag, help='Whether to train or skip training')
    parser.add_argument('--resume', default=False, type=utils.bool_flag, help='Resuming Training')
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Input batch size on each device (default: 16)')
    parser.add_argument('--img_size', type=int, default=224, help='img size')
    parser.add_argument("--min_crop_size", type=int, default=156, help="min_crop_size")
    parser.add_argument("--lr", default=2.5e-4, type=float, help="learning rate")
    parser.add_argument("--first_beta", default=0.9, type=float, help="first beta value for optimizer")
    parser.add_argument("--second_beta", default=0.999, type=float, help="second beta value for optimizer")
    parser.add_argument("--weight_decay", default=5e-2, type=float, help="weight decay for optimizer")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument('--workers', default=4, type=int, help='Workers')
    parser.add_argument('--attempt', default="Official", type=str, help='Attempt name')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adam-v2', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args.task)
    if args.task == "classification":
        EngineClassification(args)
    elif args.task == "segmentation":
        EngineSegmentation(args)