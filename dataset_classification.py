import os
import torch
import random
import copy
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure, measure
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import json
from glob import glob
from scipy import ndimage
import SimpleITK as sitk
from skimage.transform import resize
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import pandas as pd
import medpy.io
from einops import rearrange
import utils 

class Augmentation():
  def __init__(self, normalize):
    if normalize.lower() == "imagenet":
      self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() =="fundus":
      self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "none":
      self.normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)

  def get_augmentation(self, augment_name, mode, *args):
    try:
      aug = getattr(Augmentation, augment_name)
      return aug(self, mode, *args)
    except Exception as e:
      print (str(e))
      print("Augmentation [{}] does not exist!".format(augment_name))
      exit(-1)

  def basic(self, mode):
    transformList = []
    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def _full(self, transCrop, transResize, mode="train", test_augment=True):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomResizedCrop(transCrop))
      transformList.append(transforms.RandomHorizontalFlip())
      transformList.append(transforms.RandomRotation(7))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "valid":
      transformList.append(transforms.Resize((transResize,transResize)))
      transformList.append(transforms.CenterCrop(transCrop))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "test":
      if test_augment:
        transformList.append(transforms.Resize((transResize,transResize)))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if self.normalize is not None:
          transformList.append(transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
      else:
        transformList.append(transforms.Resize((transResize,transResize)))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
          transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def full_224(self, mode, test_augment=True):
    transCrop = 224
    transResize = 256
    return self._full(transCrop, transResize, mode, test_augment=test_augment)

  def full_448(self, mode, test_augment=True):
    transCrop = 448
    transResize = 512
    return self._full(transCrop, transResize, mode, test_augment=test_augment)
  def full_1024(self, mode, test_augment=True):
    transCrop = 1024
    transResize = 1024
    return self._full(transCrop, transResize, mode, test_augment=test_augment)

  def full_896(self, mode, test_augment=True):
    transCrop = 896
    transResize = 1024
    return self._full(transCrop, transResize, mode, test_augment=test_augment)

def build_dataset(args):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    transform_train = None
    transform_val = None
    transform_test = None

    transform_train = Augmentation(normalize='imagenet').get_augmentation("full_224", "train")
    transform_val = Augmentation(normalize='imagenet').get_augmentation("full_224", "valid")
    transform_test = Augmentation(normalize='imagenet').get_augmentation("full_224", "test")
    
    datasets_config = utils.get_config(args.datasets_config)
    
    if args.dataset == "nih_cxr_14":
        train_dataset = ChestX_ray14(pathImageDirectory=datasets_config['nih_cxr_14']['data_dir'], pathDatasetFile=datasets_config['nih_cxr_14']['train_list'], augment=transform_train)
        val_dataset = ChestX_ray14(pathImageDirectory=datasets_config['nih_cxr_14']['data_dir'], pathDatasetFile=datasets_config['nih_cxr_14']['val_list'], augment=transform_val)
        test_dataset = ChestX_ray14(pathImageDirectory=datasets_config['nih_cxr_14']['data_dir'], pathDatasetFile=datasets_config['nih_cxr_14']['test_list'], augment=transform_test)
        num_classes = int(datasets_config['nih_cxr_14']['classes'])
        
    elif args.dataset == "shenzhen":
        train_dataset = ShenzhenCXR(images_path=datasets_config['shenzhen']['data_dir'], file_path=datasets_config['shenzhen']['train_list'], augment=transform_train)
        val_dataset = ShenzhenCXR(images_path=datasets_config['shenzhen']['data_dir'], file_path=datasets_config['shenzhen']['val_list'], augment=transform_val)
        test_dataset = ShenzhenCXR(images_path=datasets_config['shenzhen']['data_dir'], file_path=datasets_config['shenzhen']['test_list'], augment=transform_test)
        num_classes = int(datasets_config['shenzhen']['classes'])

    elif args.dataset == "vindrcxr":
        train_dataset = VinDrCXR(images_path=datasets_config['vindrcxr']['train_val_dir'], file_path=datasets_config['vindrcxr']['train_list'], augment=transform_train)
        val_dataset = VinDrCXR(images_path=datasets_config['vindrcxr']['train_val_dir'], file_path=datasets_config['vindrcxr']['val_list'], augment=transform_val)
        test_dataset = VinDrCXR(images_path=datasets_config['vindrcxr']['test_dir'], file_path=datasets_config['vindrcxr']['test_list'], augment=transform_test)
        num_classes = int(datasets_config['vindrcxr']['classes'])
        
    return train_dataset, val_dataset, test_dataset, num_classes 

class ChestX_ray14(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=14, anno_percent=100,in_channels=3):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.in_channels =in_channels
    with open(pathDatasetFile, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(pathImageDirectory, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]
      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []
      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    if self.in_channels >1:
      imageData = Image.open(imagePath).convert('RGB')
    else:
      imageData = Image.open(imagePath).convert('L')
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)

class ShenzhenCXR(Dataset):

    def __init__(self, images_path, file_path, augment, num_class=1, in_channels=3):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.in_channels = in_channels
        
        with open(file_path, "r") as fileDescriptor:
            line = True
    
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split(',')
                    
                    imagePath = os.path.join(images_path, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)
    
        indexes = np.arange(len(self.img_list))

    def __getitem__(self, index):
    
        imagePath = self.img_list[index]
        if self.in_channels>1:
            image = Image.open(imagePath).convert('RGB')
        else:
            image = Image.open(imagePath).convert('L')
        imageLabel = torch.FloatTensor(self.img_label[index])
        
        if self.augment != None: image = self.augment(image)
        
        return image, imageLabel

    def __len__(self):

        return len(self.img_list)

class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, in_channels=3):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.in_channels = in_channels
        
        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".png")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.in_channels>1:
            imageData = Image.open(imagePath).convert('RGB')
        else:
            imageData = Image.open(imagePath).convert('L')
        
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    
    def __len__(self):
        return len(self.img_list)
