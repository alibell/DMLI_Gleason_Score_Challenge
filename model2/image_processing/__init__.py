from torchvision.models import efficientnet_b0
from torch import nn, optim
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


class imageMapper():
    def __init__ (self, classifier, device="cpu", patch_size=(512, 512), n_labels=6, intensity_theshold=220):
        """
        inference_threshold: Compute inference only if >= inference_threshold of non 255
        """
        self.classifier = classifier.to(device)
        self.device = device
        self.patch_size = patch_size
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.intensity_theshold = intensity_theshold
        self.n_labels = n_labels

    def __call__ (self, image_list):
        """
        Parameters:
        -----------
        List of images paths
        """

        image_maps = []
        zero_mask = torch.zeros(self.n_labels)

        for image in tqdm(image_list):
            # Getting image patch
            slide = openslide.OpenSlide(image)
            image_map = self.create_image_map(image, self.patch_size)
            image_map_output = np.empty(image_map.shape[0:2] + (self.n_labels,))

            images = []
            predictions = []
            # Sending the images by batch in the classifier
            for i in range(image_map.shape[0]):
                images_line = []

                for xmin, ymin, xmax, ymax in image_map[i,:,:]:
                    # Getting image
                    tmp_image = slide.read_region((xmin, ymin), 0, (xmax-xmin, ymax-ymin))
                    images_line.append(np.array(tmp_image)[:,:, 0:3])

                # Creating tensor
                images_line_tensor = torch.tensor(np.stack(images_line), dtype=torch.float32)
                images_line_tensor = torch.moveaxis(images_line_tensor, 3, 1)
                images_line_tensor = images_line_tensor.to(self.device)

                # Mask of values to evaluate
                images_line_tensor_mask = (images_line_tensor.mean(axis = (1,2,3))) <= self.intensity_theshold

                # Noramlization of image
                images_line_tensor = images_line_tensor/255.
                images_line_tensor = self.normalizer(images_line_tensor)

                # Getting the labels
                image_map_output[i, :, :] = zero_mask
                if images_line_tensor_mask.sum() >= 1:
                    y_line_mask_hat = torch.softmax(
                        self.classifier.predict(images_line_tensor[images_line_tensor_mask, :, :, :])
                    , axis=1)
                    if self.n_labels == 2:
                        image_map_output[i, images_line_tensor_mask.cpu().numpy()] = np.concatenate([
                        torch.zeros(y_line_mask_hat.shape[0]).unsqueeze(1),
                        y_line_mask_hat.cpu().numpy()
                        ], axis=1)
                    else:
                        image_map_output[i, images_line_tensor_mask.cpu().numpy()] = y_line_mask_hat.cpu().numpy()
                            
            # If 2 labels : quick fix
            if self.n_labels == 2:
                image_map_output = (1-image_map_output[:,:,0])*image_map_output[:,:,1]

            image_maps.append(image_map_output)

        return image_maps


    def create_image_map (self, image_path, patch_size):
        """
        Given an image path and a patch_size, return a list of xmin, ymin, xmax, ymax of the images patchs
        """

        # Loading image with openslide
        slide = openslide.OpenSlide(image_path)

        # Getting patch size
        patch_x, patch_y = patch_size

        # Get image shape
        images_shape = slide.level_dimensions[0]
        n_cols = images_shape[0] // patch_x
        n_rows = images_shape[1] // patch_y

        border_w = (images_shape[0] % patch_x) // 2
        border_h = (images_shape[1] % patch_y) // 2

        # Creating the image map
        image_map = np.array([[[
        border_w+j*patch_x,
        border_h+i*patch_y,
        border_w+(j+1)*patch_x,
        border_h+(i+1)*patch_y,
        ] for j in range(n_cols)] for i in range(n_rows)])

        return image_map