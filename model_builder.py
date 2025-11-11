import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
import timm

def build_classification_model(num_classes, args):
    model = None
    print("Creating model...")
    if args.pretrain == "adam":
        model = models.__dict__['resnet50'](num_classes=num_classes)
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
        if "teacher" in state_dict:
           state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        for k in list(state_dict.keys()):
           if k.startswith('fc'):
              del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded pretrained model '{}'".format(args.weights))
        print("missing keys:", msg.missing_keys)
    elif args.pretrain == "imagenet":
        imagenet_weights = models.__dict__['ResNet50_Weights'].IMAGENET1K_V1
        model = models.__dict__['resnet50'](weights=imagenet_weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print("=> loaded default (imagenet) model")
    
    return model

def build_segmentation_model(num_classes, args):
    model = None
    print("Creating model...")
    if args.pretrain == "adam":
        backbone = 'resnet50'
        model=smp.Unet(encoder_name=backbone, classes=num_classes)
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
        if "teacher" in state_dict:
           state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

        for k in list(state_dict.keys()):
           if k.startswith('fc'):
              del state_dict[k]

        state_dict = {"encoder."+k: v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(args.weights))
        print("missing keys:", msg.missing_keys)
    elif args.pretrain == "imagenet":
        model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet')
        print("=> loaded default (imagenet) model")
    
    return model
        