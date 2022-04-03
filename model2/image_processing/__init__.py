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

def softmax(array, axis):
    array_without_max = array-array.max(axis=axis, keepdims=True)
    array_exp = np.exp(array_without_max) # For numerical stability
    array_exp_sum = array_exp.sum(axis=axis, keepdims=True)

    return (array_exp/array_exp_sum)

def get_image_density_map (image_path, image_map, threshold=200):
    # Loading low res image and creating the filter map on pixel intensity
    slide = openslide.OpenSlide(image_path)
    image_array = np.array(slide.read_region((0, 0), 2, slide.level_dimensions[2]))[:,:, :3]
    image_filter = ((image_array.mean(axis=2) < threshold)).astype("int")

    # Getting the windows parameters and center the filter
    n_cols = image_map.shape[1]
    box_size_x = image_filter.shape[1]//n_cols
    border_w = (image_filter.shape[1]%n_cols) //2
    width = n_cols*box_size_x

    n_rows = image_map.shape[0]
    box_size_y = image_filter.shape[0]//n_rows
    border_h = (image_filter.shape[0]%n_rows)//2
    height = n_rows*box_size_y

    image_filter_centered = image_filter[border_h:border_h+height, border_w:border_w+width]
    # Creating the map
    density_map = np.empty((n_rows, n_cols))
    for col in range(n_cols):
        for row in range(n_rows):
            density_map[row, col] = image_filter_centered[row*box_size_y:(row+1)*box_size_y, col*box_size_x:(col+1)*box_size_x].mean()

    return density_map

def createTiles (n_tiles, n_row, image_path, image_map, map_patch_size, patch_size, density_threshold=200, random_noise=0, prop_patho = 0.7):
    """
      Given an image path and a patch_size, return a list of xmin, ymin, xmax, ymax of the images patchs
    """

    # Loading image with openslide
    slide = openslide.OpenSlide(image_path)

    # Getting patch size
    patch_x, patch_y = patch_size
    map_patch_x, map_patch_y = map_patch_size
    path_x_left, patch_y_left = patch_x, patch_y

    # Getting density map
    density_map = get_image_density_map(image_path, image_map, threshold=density_threshold)
    random_noise = (np.random.rand(*density_map.shape) >= random_noise) # We extinct tiles randomly
    density_map = density_map*random_noise

    # Getting tiles locations
    if len(image_map.shape) > 2:
      image_maps_softmax = image_map[:,:,1:]
      image_maps_softmax = image_maps_softmax*np.expand_dims(density_map, 2) # Excluding empty area
      image_maps_maximum_gleason = image_maps_softmax
    else:
      image_maps_maximum_gleason = image_map*density_map

    # Creating candidates by sampling patchs
    distribution_tiles = np.bincount(np.argmax(image_map, axis=2).flatten())[1:]
    distribution_tiles = ((distribution_tiles*int(prop_patho*n_tiles))/distribution_tiles.sum()).astype("int")
    distribution_tiles = np.concatenate([[n_tiles-distribution_tiles.sum()], distribution_tiles])
    
    # Sampling patchs
    image_maps_maximum_gleason_candidates_x = []
    image_maps_maximum_gleason_candidates_y = []

    for i in range(distribution_tiles.shape[0]):
      n_sample = distribution_tiles[i]
      if n_sample > 0:
        if i == 0:
          image_maps_maximum_gleason_candidate = np.argpartition(density_map.flatten(), -n_sample)[-n_sample:]
        else:
          image_maps_maximum_gleason_candidate = np.argpartition(image_maps_maximum_gleason[:,:,i-1].flatten(), -n_sample)[-n_sample:]

        image_maps_maximum_gleason_candidate_x = image_maps_maximum_gleason_candidate%image_maps_maximum_gleason.shape[1]
        image_maps_maximum_gleason_candidate_y = image_maps_maximum_gleason_candidate//image_maps_maximum_gleason.shape[1]
        image_maps_maximum_gleason_candidates_x.append(image_maps_maximum_gleason_candidate_x)
        image_maps_maximum_gleason_candidates_y.append(image_maps_maximum_gleason_candidate_y)

    image_maps_maximum_gleason_candidates_x = np.concatenate(image_maps_maximum_gleason_candidates_x)
    image_maps_maximum_gleason_candidates_y = np.concatenate(image_maps_maximum_gleason_candidates_y)

    # Get image shape
    images_shape = slide.level_dimensions[0]
    border_w = (images_shape[0] % map_patch_x) // 2
    border_h = (images_shape[1] % map_patch_y) // 2

    # Get the tiles coordonates
    tiles_coordonates = [(
        (x)*map_patch_x+(border_w), 
        (y)*map_patch_y+(border_h), 
        (x+1)*map_patch_x+(border_w), 
        (y+1)*map_patch_y+(border_h)
    ) for x, y in zip(image_maps_maximum_gleason_candidates_x, image_maps_maximum_gleason_candidates_y)]

    # Get the tiles
    tiles_content = []
    for xmin, ymin, xmax, ymax in tiles_coordonates:
      image = slide.read_region((xmin, ymin), 0, (patch_x, patch_y))

      # Resizing :
      image = transforms.Resize(patch_size)(image)
      tiles_content.append(np.array(image)[:,:,:3])

    tiles_content = np.stack(tiles_content)

    # Creating big image
    n_col = int(tiles_content.shape[0]/n_row)

    tiles_contents = []
    for row in range(n_row):
      tiles_contents.append(
          np.concatenate(
            tiles_content[row*n_col:(row+1)*n_col,:,:,:],
            axis=1
          )
      )
    final_image = np.concatenate(tiles_contents, axis=0)

    return final_image