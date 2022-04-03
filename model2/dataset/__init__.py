from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import openslide
import hashlib
import random
import torch
import os
import pickle
from torchvision import transforms
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from ..data_augmentation import HEDJitter

import matplotlib.pyplot as plt

"""
    Random Dataset
    Provide random images according to the Gleason Mask
    Pick random negative control images
"""

class randomDataset (Dataset):
    def __init__ (self, train_df, mask_df, normal_label=0, working_folder="/content/mvadlmi/train/train", mask_folder="/content/mvadlmi/train_label_masks/train_label_masks", image_size=512, cache="memory", cache_disk_path="./", augment=True, random_crop=True):
        """
        Parameters:
        -----------
        train_df: original train dataframe
        mask_df: dataframe containing the mask information
        cache: str, if memory the images are cached in memory, if disk they are cached on disk else no cache
        """

        super().__init__()
        self.image_size = image_size
        self.working_folder = working_folder
        self.normal_label = normal_label
        self.cache = cache
        self.cache_dict = {}
        self.cache_disk_path = cache_disk_path
        self.augment = augment
        self.random_crop_allowed = random_crop
        self.jitter = transforms.ColorJitter(brightness=0, contrast=0, saturation=1, hue=0.5),

        # Getting images list
        self.images_list = list(set(train_df["image_id"].values.tolist() + mask_df["image_id"].values.tolist()))
        self.image_openslide_dict = dict([(x, openslide.OpenSlide(f"{working_folder}/{x}.tiff")) for x in self.images_list])

        # Creating positive_df
        #  and gleason_majority == gleason_minority
        self.positives = mask_df \
        .query("(data_provider == 'radboud' and image_label_local >= 3) or (data_provider == 'karolinska' and image_label_local == 2)") \
        .assign(label = lambda x: x.apply(lambda y: y["image_label_local"] if y["data_provider"] == "radboud" else y["gleason_majority"], axis=1))[[
            "image_id", "xmin", "ymin", "xmax", "ymax", "w", "h", "label"
        ]] \
        .assign(
            label=lambda x: x["label"].map({
                3:1,
                4:2,
                5:3
            })
        )

        # Creating negative dataframe
        self.negatives = self._create_negatives_df(train_df, int(2*self.positives.shape[0]))

        # Global dataframe
        self.dataframe = pd.concat([
            self.positives,
            self.negatives
        ])

        # Torchvision utils
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resizer = transforms.Resize(size=self.image_size)
        self.random_crop = transforms.RandomCrop((self.image_size, self.image_size))
        self.center_crop = transforms.CenterCrop((self.image_size, self.image_size))
        self.random_hflip = transforms.RandomHorizontalFlip()
        self.random_vflip = transforms.RandomVerticalFlip()
        self.rotate = transforms.RandomRotation(degrees=90, fill=1)


    def _create_negatives_df (self, train_df, size):
        """
        Creating a dataframe of negative non empty location

        Parameters:
        -----------
        train_df: train dataframe
        size: int, size of the negative dataframe
        """

        # Negatives image list
        negative_images_name = train_df \
            .query("isup_grade == 0")["image_id"].unique().tolist()

        # We loop over images and produce the negative dataframe
        negative_images_dict = []
        for negative_image in negative_images_name:
            negative_image_slide = openslide.OpenSlide(f"/content/mvadlmi/train/train/{negative_image}.tiff")
            negative_image_mask = np.array(negative_image_slide.read_region((0,0), 2, [self.image_size, self.image_size]))[:,:, 0:3].mean(axis=2)
            negative_image_mask = (negative_image_mask <= 220).astype("int")
            negative_image_mask_ratio = negative_image_slide.level_dimensions[0][0]/negative_image_slide.level_dimensions[2][0]
            negative_image_locations = np.where(negative_image_mask == 1)

            negative_image_locations_dict = dict(zip(["ymin", "xmin"], negative_image_locations))
            negative_image_locations_dict["image_id"] = np.array([negative_image for i in range(len(negative_image_locations_dict["xmin"]))])

            negative_images_dict.append(negative_image_locations_dict)

        negatives_df = pd.concat([pd.DataFrame(x) for x in negative_images_dict]) \
        .assign(
            xmin = lambda x: x["xmin"]*negative_image_mask_ratio,
            ymin = lambda x: x["ymin"]*negative_image_mask_ratio,
            xmax = lambda x: x["xmin"]*negative_image_mask_ratio+self.image_size,
            ymax = lambda x: x["ymin"]*negative_image_mask_ratio+self.image_size,
            w = self.image_size,
            h = self.image_size,
            n=size,
            label=self.normal_label
        ).sample(size)[["image_id", "xmin", "ymin", "xmax", "ymax", "w", "h", "label"]]

        return negatives_df

    def _get_image(self, image_id, x, y, w, h):
        
        # Getting image uid
        image_uid = hashlib.md5(
            (f"{image_id}_{x}_{y}_{w}_{h}").encode("utf8")
        ).hexdigest()

        if self.cache in ("disk", "memory") and image_uid in self.cache_dict.keys():
            if self.cache == "disk":
                image_path = self.cache_dict[image_uid]
                image = np.load(image_path)
            else:
                image = torch.clone(self.cache_dict[image_uid]).cpu()
            
            image = torch.tensor(image, dtype=torch.float32)

        else:
            # Loading image
            openslide = self.image_openslide_dict[image_id]
            image = np.array(openslide.read_region([x, y], 0, [w, h]))[:,:,0:3]

            if self.cache in ("disk", "memory") and image_uid not in self.cache_dict.keys():
            if self.cache == "disk":
                image_path = f"{self.cache_disk_path}/{image_uid}.npz"
                np.savez(image_path, image)
            else:
                self.cache_dict[image_uid] = torch.clone(image)
            
            image = torch.tensor(image, dtype=torch.float32)

        return image

    def __len__ (self):
        return self.dataframe.shape[0]

    def __getitem__ (self, idx):
        # Getting image
        image_id = self.dataframe.iloc[idx,]["image_id"]

        label = torch.tensor(self.dataframe.iloc[idx,]["label"]).long()
        x, y, w, h = [int(x) for x in self.dataframe.iloc[idx,][["xmin", "ymin", "w", "h"]].tolist()]
        image = self._get_image(image_id, x, y, max(self.image_size, w), max(self.image_size, h))

        # Image processing

        ## Moving axis
        image = torch.moveaxis(image, 2, 0)

        ## Standardization
        image = image/255.

        ## Resizer
        #image = self.resizer(image)

        ## Random crop
        if self.random_crop:
            image = self.random_crop(image)
        else:
            image = self.resizer(image)
            image = self.center_crop(image)

        ## Normalize
        image = self.normalizer(image)

        ## Augmentation
        if self.augment:
            image = self.random_hflip(self.random_vflip(image))
        if random.random() <= 0.1:
            image = HEDJitter(0.05)(image)
        image = self.rotate(image)

        return image, label