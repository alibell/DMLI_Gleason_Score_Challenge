import openslide
import glob
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw
import copy

def get_contours (image, image_ratio=16, min_size=256):
    coordonates = []
    
    # Drawing inside contours
    contours, hierarchy = cv2.findContours(image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    # Creating mask image
    mask_image = np.zeros(image.shape).astype(np.uint8)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # Filtering contour
        box_content = image[y:y+h, x:x+w]
        box_content_mean = box_content.mean()

        if box_content_mean >= 0.2 and image_ratio*w >= min_size and image_ratio*h >= min_size:
            mask_image[y:y+h, x:x+w] = 1

    # Drawing contours from mask image
    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
    else:
        hierarchy = []
    
    for cnt, hier in zip(contours, hierarchy):
        x,y,w,h = cv2.boundingRect(cnt)
        
        # Filtering contour
        box_content = image[y:y+h, x:x+w]
        box_content_mean = box_content.mean()
        
        if box_content_mean >= 0.2 and image_ratio*w >= min_size and image_ratio*h >= min_size and hier[3] < 0:
            coordonates.append([
                image_ratio*x, image_ratio*y, image_ratio*(x+w), image_ratio*(y+h)
            ])
        
    return coordonates

def draw_contours (image, coordonates):
    image = Image.fromarray(image)
    image_draw = ImageDraw.Draw(image)
    for cnt in coordonates:
        image_draw.rectangle(cnt, width=10)
    return image


def get_images_box(images_df, images_paths):
    output = {}
    for image in tqdm(images_df["image_id"]):
        image_path = f"{images_paths}/{image}.tiff"
        if os.path.exists(image_path):
            os_image = openslide.OpenSlide(image_path)
            whole_image = np.array(os_image.read_region((0, 0), 2, os_image.level_dimensions[2]))[:,:, 0]
            whole_image_ratio = os_image.level_dimensions[0][0]/os_image.level_dimensions[2][0]

            # Getting coordonates
            coordonates = []
            for i in range(1, 6, 1):
                image_mask = (whole_image == i).astype(np.uint8)
                coordonate = get_contours(image_mask, image_ratio=whole_image_ratio)
                coordonates.append(coordonate)
                
            coordonates_dict = dict(zip(range(1, 6, 1), coordonates))

            output[image] = coordonates_dict
        
    images_box_for_df = [[x]+[key]+value for x, y in output.items() for key, values in y.items() for value in values]

    mask_df = pd.DataFrame(images_box_for_df).rename(
            columns={0:"image_id", 1:"image_label_local", 2:"xmin", 3:"ymin", 4:"xmax", 5:"ymax"}
        ) \
        .assign(level=1) \
        .join(
            images_df[["image_id", "gleason_score", "isup_grade", "data_provider"]].set_index("image_id"), on="image_id"
        ) \
        .assign(gleason_score_tmp=lambda x: x["gleason_score"].replace("negative", "0+0")) \
        .assign(
            gleason_majority=lambda x: x["gleason_score_tmp"].str.split("+").apply(lambda y: y[0]).astype("int"),
            gleason_minority=lambda x: x["gleason_score_tmp"].str.split("+").apply(lambda y: y[1]).astype("int")
        ) \
        .assign(
            xmax = lambda x: x["xmax"].astype("int"),
            xmin = lambda x: x["xmin"].astype("int"),
            ymax = lambda x: x["ymax"].astype("int"),
            ymin = lambda x: x["ymin"].astype("int"),
            w = lambda x: x["xmax"]-x["xmin"],
            h = lambda x: x["ymax"]-x["ymin"]
        )


    return mask_df