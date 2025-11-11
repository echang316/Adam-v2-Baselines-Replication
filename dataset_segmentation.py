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
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import pandas as pd
import medpy.io
from einops import rearrange
import utils 
from albumentations.pytorch import ToTensorV2
import albumentations
from albumentations import Compose, HorizontalFlip, Normalize, VerticalFlip, Rotate, Resize, ShiftScaleRotate, OneOf, GridDistortion, OpticalDistortion, \
    ElasticTransform, GaussNoise, MedianBlur,  Blur, CoarseDropout,RandomBrightnessContrast,RandomGamma,RandomSizedCrop, ToFloat

from albumentations.pytorch import ToTensorV2

def build_transform_segmentation(args):
    transformSequence = Compose([
        OneOf([
            RandomBrightnessContrast(),
            RandomGamma(),
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=1, sigma=50),
            GridDistortion(num_steps=5),
            OpticalDistortion(distort_limit=2),
            #This is out of date:
            #ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #GridDistortion(),
            #OpticalDistortion(distort_limit=2, shift_limit=0.5), 
        ], p=0.3),
        RandomSizedCrop(min_max_height=(156, 224), size=(224, 224), p=0.25),
        ToFloat(max_value=1)
    ], p=1)

    return transformSequence

def build_dataset(args):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    transform_train = None
    transform_val = None
    transform_test = None

    transforms = build_transform_segmentation(args)  
    
    datasets_config = utils.get_config(args.datasets_config)

    if args.dataset == "chestxdet":
        if args.mode == "full_transfer":
            train_list = datasets_config['chestxdet']['train_list_full']
        elif args.mode == "fewshot_5":
            train_list = datasets_config['chestxdet']['train_list_few_5']
        elif args.mode == "fewshot_10":
            train_list = datasets_config['chestxdet']['train_list_few_10']
            
        train_dataset = ChestXDet(images_dir=datasets_config['chestxdet']['train_imgs'], masks_dir=datasets_config['chestxdet']['train_masks'], image_list=train_list, augmentation=transforms)
        val_dataset = ChestXDet(images_dir=datasets_config['chestxdet']['train_imgs'], masks_dir=datasets_config['chestxdet']['train_masks'], image_list=datasets_config['chestxdet']['val_list'], augmentation=transforms)
        test_dataset = ChestXDet(images_dir=datasets_config['chestxdet']['test_imgs'], masks_dir=datasets_config['chestxdet']['test_masks'], image_list=datasets_config['chestxdet']['test_list'], augmentation=None)
        num_classes = int(datasets_config['chestxdet']['classes'])

    elif args.dataset == "siim_acr_ptx":
        if args.mode == "full_transfer":
            img_list = datasets_config['siim_acr_ptx']['img_list']
        elif args.mode == "fewshot_5":
            img_list = datasets_config['siim_acr_ptx']['img_list_few_5']
        elif args.mode == "fewshot_10":
            img_list = datasets_config['siim_acr_ptx']['img_list_few_10']
        
        with open(img_list, 'r') as file:
             data = json.load(file)
        
        train_dataset = SIIM_ACR_PTX(data_dir=datasets_config['siim_acr_ptx']['data_dir'], img_list=data['train'], transforms=transforms)
        val_dataset = SIIM_ACR_PTX(data_dir=datasets_config['siim_acr_ptx']['data_dir'], img_list=data['val'], transforms=transforms)
        test_dataset = SIIM_ACR_PTX(data_dir=datasets_config['siim_acr_ptx']['data_dir'], img_list=data['test'], transforms=None)
        num_classes = int(datasets_config['siim_acr_ptx']['classes'])

    elif args.dataset == "drive":
        data=None
        with open(datasets_config['drive']['file'], 'r') as file:
             data = json.load(file)
    
        train_dataset = DRIVE(images_path=datasets_config['drive']['train_val_images'], mask_path=datasets_config['drive']['train_val_masks'], image_list=data['train'], augment=transforms)
        val_dataset = DRIVE(images_path=datasets_config['drive']['train_val_images'], mask_path=datasets_config['drive']['train_val_masks'], image_list=data['val'], augment=transforms)
        test_dataset = DRIVE(images_path=datasets_config['drive']['test_images'], mask_path=datasets_config['drive']['test_masks'], image_list=data['test'], augment=None)
        num_classes = int(datasets_config['drive']['classes'])

    elif args.dataset == "drishti_gs":
        data=None
        with open(datasets_config['drishti_gs']['file'], 'r') as file:
             data = json.load(file)
    
        train_dataset = Drishti_GS(images_path=datasets_config['drishti_gs']['train_val_images'], mask_path=datasets_config['drishti_gs']['train_val_masks'], image_list=data['train'], augment=transforms)
        val_dataset = Drishti_GS(images_path=datasets_config['drishti_gs']['train_val_images'], mask_path=datasets_config['drishti_gs']['train_val_masks'], image_list=data['val'], augment=transforms)
        test_dataset = Drishti_GS(images_path=datasets_config['drishti_gs']['test_images'], mask_path=datasets_config['drishti_gs']['test_masks'], image_list=data['test'], augment=None)
        num_classes = int(datasets_config['drishti_gs']['classes'])
        
    return train_dataset, val_dataset, test_dataset, num_classes 

class ChestXDet(Dataset):
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            image_list,
            augmentation=None, 
            dim=(224, 224, 3),
            normalization='imagenet'
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.dim = dim
        with open(image_list, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [
            os.path.join(masks_dir, f"{os.path.splitext(image_id)[0]}.png")
            for image_id in self.ids
        ]
        
        self.augmentation = augmentation
        self.normalization = normalization
    
    def __getitem__(self, i):    
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image = Image.open(self.images_fps[i])
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.masks_fps[i]))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        #print(f"Image Before: {image.shape}")
        #print(f"Mask Before: {mask.shape}")
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = np.array(image) / 255.
        mask = np.array(mask) / 255.

        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std
        mask = np.array(mask)
        image=image.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        
        #print(f"Image After: {image.shape}")
        #print(f"Mask After: {mask.shape}")
        #print(f"mask values range: min={torch.min(mask)}, max={torch.max(mask)}")
        #print(f"image values range: min={torch.min(image)}, max={torch.max(image)}")
        return (image, mask)
        
    def __len__(self):
        return len(self.ids)

class SIIM_ACR_PTX(Dataset):
    def __init__(self, data_dir, img_list, transforms, dim=(224, 224, 3), normalization="imagenet"):
        self.data_dir = data_dir
        self.transforms = transforms
        self.img_list = img_list
        self.dim = dim 
        self.normalization = normalization
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name = self.img_list[idx]
        image = Image.open(os.path.join(self.data_dir, image_name+".dcm.jpeg"))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = Image.open(os.path.join(self.data_dir, image_name+".dcm_mask.jpeg"))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']


        image = np.array(image) / 255.
        mask = np.array(mask) / 255.

        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std
        mask = np.array(mask)
        image=image.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        
        return (image, mask) 
    
class SCRDataset(Dataset):
    def __init__(self, pathImageDirectory, pathMaskDirectory, pathDatasetFile,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list = []
        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline().strip('\n')
                if line:
                    self.img_list.append(line)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            if num_data ==0:
                num_data =1
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name+".IMG.png"))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name+".gif"))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)

class VinDrRibCXRDataset(Dataset):
    def __init__(self, image_path_file, image_size, mode,annotation=100):
        self.pathImageDirectory, pathDatasetFile = image_path_file
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        self.rib_labels =  ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',
                           'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
        f = open(pathDatasetFile)
        data= json.load(f)
        self.img_list = data['img']
        self.label_list = data
        self.indexes = np.arange(len(self.img_list))
        if annotation < 100:
            random.Random().shuffle(self.indexes)
            num_data = int(self.indexes.shape[0] * annotation / 100.0)

            if num_data ==0:
                num_data =1
            self.indexes = self.indexes[:num_data]
        print("number of images:", len(self.indexes))

    def __getitem__(self, index):
        ind=self.indexes[index]
        imagePath = self.img_list[str(ind)]
        imageData = cv2.imread(os.path.join(self.pathImageDirectory, imagePath), cv2.IMREAD_COLOR)
        label0 = []
        for name in self.rib_labels:
            pts = self.label_list[name][str(ind)]
            label = np.zeros((imageData.shape[:2]), dtype=np.uint8)
            if pts != 'None':
                pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
                label = cv2.fillPoly(label, [pts], 1)
                label = cv2.resize(label, self.image_size,interpolation=cv2.INTER_AREA)
            label0.append(label)
        label0 = np.stack(label0)
        label0 = label0.transpose((1, 2, 0))

        imageData = cv2.resize(imageData,self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode] (image=imageData, mask=label0)
        img = dic['image']
        mask = (dic['mask'].permute(2, 0, 1))
        return img, mask
    def __len__(self):
        return len(self.indexes)

class DRIVE(Dataset):
    def __init__(self, images_path, mask_path, image_list, augment, dim=(512, 512, 3), normalization="imagenet"):
        self.img_list = []
        self.mask_list = []
        self.augment = augment
        self.dim = dim 
        self.normalization = normalization
        
        for file in image_list:
            imagePath = os.path.join(images_path, file+".tif")
            maskPath = os.path.join(mask_path, file+"_manual1.gif")
            self.img_list.append(imagePath)
            self.mask_list.append(maskPath)

    def __getitem__(self, index):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        imagePath = self.img_list[index]
        maskPath = self.mask_list[index]
        image = Image.open(imagePath).convert('RGB')
        image = (np.array(image)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = Image.open(maskPath).convert('L')
        mask = (np.array(mask)).astype('uint8')
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)      
        mask[mask > 0] = 255
        
        if self.augment:
            sample = self.augment(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image=np.array(image) / 255.
        mask=np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std
        #print(f"Image: {image.shape}")
        #print(f"Mask: {mask.shape}")
        mask = np.array(mask)
        image=image.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return image, mask
    
    def __len__(self):
        return len(self.img_list)

class Drishti_GS(Dataset):
    def __init__(self, images_path, mask_path, image_list, augment, dim=(224, 224, 3), normalization='imagenet'):
        self.img_list = []
        self.mask_list = []
        self.augment = augment
        self.dim = dim
        self.normalization = normalization 

        for file in image_list:
            imagePath = os.path.join(images_path, file+".png")
            maskPath = os.path.join(mask_path, file, "SoftMap", file+"_cupsegSoftmap.png")
            self.img_list.append(imagePath)
            self.mask_list.append(maskPath)

    def __getitem__(self, index):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        imagePath = self.img_list[index]
        maskPath = self.mask_list[index]
        image = Image.open(imagePath).convert('RGB')
        image = (np.array(image)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = Image.open(maskPath).convert('L')
        mask = (np.array(mask)).astype('uint8')
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)      
        mask[mask > 0] = 255
        #print(f"Image {image.dtype}")
        #print(f"Mask {mask.dtype}")
        if self.augment:
            sample = self.augment(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image=np.array(image) / 255.
        mask=np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std
        #print(f"Image: {image.shape}")
        #print(f"Mask: {mask.shape}")
        mask = np.array(mask)
        image=image.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return image, mask
    
    def __len__(self):
        return len(self.img_list)