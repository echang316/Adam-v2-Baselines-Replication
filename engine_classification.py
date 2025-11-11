import time
import torch
from tqdm import tqdm
import timm

import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from dataset_classification import build_dataset
import shutil

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
import utils
from utils import get_config, metric_AUROC, AverageMeter, ProgressMeter, step_decay
from model_builder import build_classification_model

def setup(args):
    model_path = os.path.join(args.output_dir, args.mode, args.dataset, args.attempt, "Trained_Models")
    output_path = os.path.join(args.output_dir, args.mode, args.dataset, args.attempt, "Outputs")
    log_file = os.path.join(model_path, "Model.log")
    output_file = os.path.join(output_path, "results.txt")
    
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    return model_path, log_file, output_path, output_file 

def Engine(args):
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    train_dataset, val_dataset, test_dataset, num_classes = build_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    model_path, log_file, output_path, output_file = setup(args) 
    
    if args.train_toggle:
        for trial in range(args.start_trial, args.trials+1):
            start_epoch = 0
            best_val_loss = 1000000
            patience_counter = 0
            experiment = "run_" + str(trial)
            save_model_path = os.path.join(model_path, experiment)
            model = build_classification_model(num_classes, args).cuda()

            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            loss = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(args.first_beta, args.second_beta), weight_decay=args.weight_decay)
            fp16_scaler = None
            if args.use_fp16:
                fp16_scaler = torch.GradScaler("cuda")
            
            if args.resume:
                resume = os.path.join(model_path, experiment + '.pth')
                if os.path.isfile(resume):
                    print("=> loading checkpoint '{}'".format(resume))
                    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
                    
                    start_epoch = checkpoint['epoch']
                    best_val_loss = checkpoint['lossMIN']
                    model.load_state_dict(checkpoint['state_dict'])
                    #lr_scheduler.load_state_dict(checkpoint['scheduler'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    dist.barrier()
                    print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                        .format(resume, start_epoch, best_val_loss))
                
                else:
                    print("=> no checkpoint found at '{}'".format(resume))
            
            for epoch in range(start_epoch, start_epoch+args.epochs):
                lr_ = step_decay(epoch, args)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                
                train_one_epoch(train_loader, model, loss, optimizer, fp16_scaler, epoch, args)
                val_loss = evaluate(val_loader, model, loss, epoch, args)
                
                log_stats = {'val_loss': val_loss, 'epoch': epoch}
                
                with (Path(output_path)/"log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                #lr_scheduler.step(avg_val_loss)

                if val_loss < best_val_loss:
                    print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss, save_model_path))
                    save_dict = {
                        'epoch': epoch,
                        'lossMIN': best_val_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        #'scheduler': lr_scheduler.state_dict()
                    }

                    if fp16_scaler is not None:
                        save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                        
                    torch.save(save_dict, Path(save_model_path+".pth"))
            
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss))
                    patience_counter += 1
            
                if patience_counter > args.patience:
                    print("Early Stopping")
                    break
        
            with open(log_file, 'a') as f:
                f.write(experiment + "\n")
                f.close()
    
    mean_loss = []

    with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
        experiment = reader.readline() 

        while experiment:
            experiment = experiment.replace('\n', '')
            saved_model = os.path.join(model_path, experiment + ".pth")
            
            individual, mean = test(saved_model, test_loader, num_classes, args)
            
            print(">>{}: AUC = {}".format(experiment, np.array2string(np.array(individual), precision=4, separator=',')))
            writer.write("{}: AUC = {}\n".format(experiment, np.array2string(np.array(individual))))
            print(">>{}: Mean AUC = {:.4f}".format(experiment, mean))
            writer.write("{}: Mean AUC = {:.4f}\n".format(experiment, mean))
            mean_loss.append(mean) 
            experiment = reader.readline()
        
        mean_loss = np.array(mean_loss)
        
        print(">> All trials: mAUC  = {}".format(np.array2string(mean_loss, precision=4, separator=',')))
        writer.write("All trials: mAUC  = {}\n".format(np.array2string(mean_loss, precision=4, separator='\t')))
        print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_loss)))
        writer.write("Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_loss)))
        print(">> STD over All trials:  = {:.4f}".format(np.std(mean_loss)))
        writer.write("STD over All trials:  = {:.4f}\n".format(np.std(mean_loss)))
    
def train_one_epoch(data_loader_train, model, criterion, optimizer, fp16_scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(data_loader_train),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    end = time.time()
    
    for i, (samples, targets) in enumerate(data_loader_train):
        enabled = fp16_scaler is not None
        with torch.autocast("cuda", enabled=enabled):
            samples, targets = samples.float().cuda(), targets.float().cuda()
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optimizer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        torch.cuda.synchronize()
        losses.update(loss.item(), samples.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % args.print_freq == 0:
          progress.display(i)

def evaluate(data_loader_val, model, criterion, epoch, args):
    model.eval()

    with torch.no_grad():
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
          len(data_loader_val),
          [batch_time, losses], prefix='Val: ')
    
        end = time.time()

        for i, (samples, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().cuda(), targets.float().cuda()
            outputs = model(samples)
            loss = criterion(outputs, targets)
    
            torch.cuda.synchronize()
            losses.update(loss.item(), samples.size(0))
            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg

def test(checkpoint, data_loader_test, classes, args):
    model = build_classification_model(classes, args).cuda()
    modelCheckpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)

    state_dict = modelCheckpoint['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=True)
    
    print("=> loaded fine-tuned model '{}'".format(checkpoint))
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.eval()

    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()

    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

            out = torch.sigmoid(model(varInput))
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    individual_results = metric_AUROC(y_test, p_test, classes)
    mean_over_all_classes = np.array(individual_results).mean()

    return individual_results, mean_over_all_classes 
    