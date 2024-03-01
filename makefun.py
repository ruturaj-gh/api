from app import DataLoader, Dataset,DatasetConfig,dataclass,denormalize,display_image_and_mask
from app import Paths, TrainingConfig, InferenceConfig, id2color, rev_id2color, num_to_rgb, image_overlay
import os
import zipfile
import platform
import warnings
from glob import glob
from dataclasses import dataclass

# To filter UserWarning.
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import requests
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# For data augmentation and preprocessing.
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Imports required SegFormer classes
from transformers import SegformerForSemanticSegmentation

# Importing lighting along with a built-in callback it provides.
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Importing torchmetrics modular and functional implementations.
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score

# To print model summary.
from torchinfo import summary

import joblib

class MedicalSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_classes=2,
        img_size=(640, 640),
        ds_mean=(0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle_validation=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.img_size    = img_size
        self.ds_mean     = ds_mean
        self.ds_std      = ds_std
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory

        self.shuffle_validation = shuffle_validation

    def prepare_data(self):
        # Download dataset.
        dataset_zip_path = f"{DatasetConfig.DATASET_PATH}.zip"

        # Download if dataset does not exists.
        if not os.path.exists(DatasetConfig.DATASET_PATH):

            print("Downloading and extracting assets...", end="")
            file = requests.get(DatasetConfig.URL)
            open(dataset_zip_path, "wb").write(file.content)

            try:
                with zipfile.ZipFile(dataset_zip_path) as z:
                    z.extractall(os.path.split(dataset_zip_path)[0]) # Unzip where downloaded.
                    print("Done")
            except:
                print("Invalid file")

            os.remove(dataset_zip_path) # Remove the ZIP file to free storage space.

    def setup(self, *args, **kwargs):
        # Create training dataset and dataloader.
        train_imgs = sorted(glob(f"{Paths.DATA_TRAIN_IMAGES}"))
        train_msks  = sorted(glob(f"{Paths.DATA_TRAIN_LABELS}"))

        # Create validation dataset and dataloader.
        valid_imgs = sorted(glob(f"{Paths.DATA_VALID_IMAGES}"))
        valid_msks = sorted(glob(f"{Paths.DATA_VALID_LABELS}"))

        self.train_ds = MedicalDataset(image_paths=train_imgs, mask_paths=train_msks, img_size=self.img_size,
                                       is_train=True, ds_mean=self.ds_mean, ds_std=self.ds_std)

        self.valid_ds = MedicalDataset(image_paths=valid_imgs, mask_paths=valid_msks, img_size=self.img_size,
                                       is_train=False, ds_mean=self.ds_mean, ds_std=self.ds_std)

    def train_dataloader(self):
        # Create train dataloader object with drop_last flag set to True.
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        # Create validation dataloader object.
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory,
            num_workers=self.num_workers, shuffle=self.shuffle_validation
        )
    
class MedicalDataset(Dataset):
    def __init__(self, *, image_paths, mask_paths, img_size, ds_mean, ds_std, is_train=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.is_train    = is_train
        self.img_size    = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.transforms  = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)
        print("Length of image_paths:", len(self.image_paths))
        print("Length of mask_paths:", len(self.mask_paths))

    def __len__(self):
        return len(self.image_paths)


    def setup_transforms(self, *, mean, std):
        transforms = []

        # Augmentation to be applied to the training set.
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=self.img_size[1]//20, max_width=self.img_size[0]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
            ])

        # Preprocess transforms - Normalization and converting to PyTorch tensor format (HWC --> CHW).
        transforms.extend([
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(always_apply=True),  # (H, W, C) --> (C, H, W)
        ])
        return A.Compose(transforms)

    def load_file(self, file_path, depth=0):
        file = cv2.imread(file_path, depth)
        if depth == cv2.IMREAD_COLOR:
            file = file[:, :, ::-1]
        return cv2.resize(file, (self.img_size), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, index):

      try:
          # Load image and mask file.
          image = self.load_file(self.image_paths[index], depth=cv2.IMREAD_COLOR)
          mask = self.load_file(self.mask_paths[index], depth=cv2.IMREAD_GRAYSCALE)

          # Apply Preprocessing (+ Augmentations) transformations to image-mask pair
          transformed = self.transforms(image=image, mask=mask)
          image, mask = transformed["image"], transformed["mask"].to(torch.long)  # Ensure mask is of type long

          return image, mask
      except Exception as e:
          print(f"Error loading data at index {index}: {str(e)}")
          # You can return dummy data or handle the error based on your use case.
          # For now, returning zeros as a placeholder.
          return torch.zeros((self.img_size[1], self.img_size[0], 3)), torch.zeros((self.img_size[1], self.img_size[0]))


dm = joblib.load("segformer.joblib")

# Assuming you have a test dataset similar to your training dataset structure
test_imgs = sorted(glob(f"{Paths.DATA_TEST_IMAGES}"))
test_msks = sorted(glob(f"{Paths.DATA_TEST_LABELS}"))

test_ds = MedicalDataset(image_paths=test_imgs, mask_paths=test_msks, img_size=dm.img_size,
                         is_train=False, ds_mean=dm.ds_mean, ds_std=dm.ds_std)

test_dataloader = DataLoader(test_ds, batch_size=dm.batch_size,
                              num_workers=dm.num_workers, shuffle=False)

for index, (batch_images, batch_masks) in enumerate(test_dataloader):
    print(f"Index: {index}, Image Paths Length: {len(dm.valid_ds.image_paths)}, Mask Paths Length: {len(dm.valid_ds.mask_paths)}")
    batch_images = denormalize(batch_images, mean=DatasetConfig.MEAN, std=DatasetConfig.STD).permute(0, 2, 3, 1).numpy()
    batch_masks = batch_masks.numpy()

    print("batch_images shape:", batch_images.shape)
    print("batch_masks shape: ", batch_masks.shape)
    display_image_and_mask(images=batch_images, masks=batch_masks)