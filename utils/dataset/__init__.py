from torch.utils.data import Dataset
import tifffile
import shutil
import copy
import os
from tqdm import tqdm
import pickle
from PIL import Image
from scipy.sparse import csr_matrix
from torchvision.io import read_image
from torch import nn
import torch
from torchvision import transforms

class prostateDataset(Dataset):
    """
        The aim of this class is to permit the Image loading and manipulation
        Functionnality :
            - Generate a slicing windows of the images and write them to disk
            - Iterable
            - For each image produce the Gleason and isup_grade
            - Can provide all the images of a file
    """

    def __init__ (self, 
                images_folder, 
                labels, 
                output_folder, 
                slidding_x = 300,
                slidding_y = 300, 
                windows_size_x=500, 
                windows_size_y=500, 
                hflip=False,
                vflip=False,
                reset=False,
                verbose=True):
        """
            Parameters:
            -----------
            images_folder: list, name of the folder containing the images, None if pre-loaded dataset
            labels: list of dictionnary containing the labels with the keys image_id, isup_grade and gleason_score, None if pre-loaded dataset
            output_folder: str, Name of the folder where the images are written
            slidding_x: int, size of the x sliding
            slidding_y: int, size of the y sliding
            windows_size_x: int, width of the windows
            windows_size_y: int, height of the windows
            hflip: boolean, if True a random hflip is performed
            vflip: boolean, if True a random vflip is performed
            reset: Boolean, if true the folder is cleaned
            verbose: Boolean, if true the informations are verbosed
        """

        super().__init__()

        self.images_folder = images_folder
        self.labels = labels
        self.output_folder = output_folder
        self.destination_folder = f"{output_folder}/data"
        self.metadata_path = f"{self.destination_folder}/metadata.pickle"
        self.reset = reset
        self.slidding = (slidding_x, slidding_y)
        self.windows = (windows_size_x, windows_size_y)
        self.verbose = verbose
        self.transform = True
        self.hflip = hflip
        self.vflip = vflip

        # Cleaning output folder
        for folder in [output_folder, self.destination_folder]:
            if os.path.exists(folder) == False:
                os.mkdir(folder)

        if self.reset:
            shutil.rmtree(self.destination_folder, ignore_errors=True)
            os.mkdir(self.destination_folder)

        # Creating images dataset
        if os.path.exists(self.metadata_path):
            self.metadatas = pickle.load(open(self.metadata_path, "rb"))
        else:
            # Writting files
            self.metadatas = []
            for x in labels:
                image_id = x["image_id"]
                metadata = self._create_images(f"{self.images_folder}/{image_id}.tiff")
                for i in range(len(metadata)):
                    if "isup_grade" in x.keys():
                        metadata[i]["isup_grade"] = x["isup_grade"]
                    else:
                        metadata[i]["isup_grade"] = None
                    if "gleason_score" in x.keys():
                        metadata[i]["gleason_score"] = x["gleason_score"]
                    else:
                        metadata[i]["gleason_score"] = None
                
                self.metadatas += metadata

            # Writting metadata
            pickle.dump(self.metadatas, open(self.metadata_path, "wb"))

    def __len__ (self):
        return len(self.metadatas)

    def _load_tiff (self, image_path):
        image = tifffile.imread(image_path)
        
        return image
    
    def _get_windows_location (self, row, col, image_shape, slidding, windows):
        start_row = row*slidding[1]
        end_row = start_row+windows[1]
        start_row = start_row if end_row <= image_shape[0] else image_shape[0]-windows[1]
        end_row = end_row if end_row <= image_shape[0] else image_shape[0]
                                                                        
        start_col = col*slidding[0]
        end_col = start_col+windows[0]
        start_col = start_col if end_col <= image_shape[1] else image_shape[1]-windows[0]
        end_col = end_col if end_col <= image_shape[1] else image_shape[1]

        return start_row, end_row, start_col, end_col

    def _get_windows (self, image_shape):
        """
            Generate a list of potential windows
        """
        
        windows_list = []
        slidding = self.slidding
        windows = self.windows
        
        # Getting the number of line and columns
        n_rows = (image_shape[0]//slidding[1])+(image_shape[0]%slidding[1] != 0)-(windows[1]//slidding[1])+1
        n_cols = (image_shape[1]//slidding[0])+(image_shape[1]%slidding[0] != 0)-(windows[0]//slidding[0])+1        
        
        # For each row and line
        for row in range(n_rows):
            for col in range(n_cols):
                # Getting the current window
                start_row, end_row, start_col, end_col = self._get_windows_location(row, col, image_shape, slidding, windows)
                windows_list.append((start_row, end_row, start_col, end_col))
                
        return windows_list
        
    def _create_images (self, image_path):
        """
            For an image path, write windows on disk and provide a list of windows path with location and original image path
        """
        
        # Loading image
        image = self._load_tiff(image_path)
        image_name = image_path.split("/")[-1]
        
        # Getting the windows list
        windows_list = self._get_windows(image.shape[0:2])
        
        # Creating the images
        output_images = []
        
        ## Creating the folder
        destination_folder = f"{self.destination_folder}/{image_name}"
        if os.path.exists(destination_folder) == False:
            os.mkdir(destination_folder)
        i = 0
        
        if self.verbose:
            print(f"Writting {image_path}")
        for window in tqdm(windows_list):
            # Only for non empty location
            image_mask = ((image[window[0]:window[1], window[2]:window[3]] != 255)*1).sum(axis=2) # List of non white location
            if image_mask.sum() > 0:
                image_output_path = f"{destination_folder}/{i}.jpg"
                Image.fromarray(
                    image[window[0]:window[1], window[2]:window[3], :]
                ).save(image_output_path)
                i += 1
                
                output_images.append({
                    "image_path":image_path,
                    "image_name":image_name,
                    "offset":window,
                    "path":image_output_path,
                    "id":i
                })
                
        return output_images

    def __getitem__(self, index):
        img_metadata = self.metadatas[index]
        img_label = img_metadata["gleason_score"]
        img_label_isup = img_metadata["isup_grade"]
        if "id" not in img_metadata.keys():
            img_id = img_metadata["path"].split("/")[-1].split(".")[0]
        else:
            img_id = str(img_metadata["id"])
        img_name = img_metadata["image_name"]
        img_path = f"{self.destination_folder}/{img_name}/{img_id}.jpg"

        image = read_image(img_path)
        label = [0,0, img_label_isup] if img_label == "negative" else ([int(x) for x in img_label.split("+")] + [img_label_isup])
        label = torch.tensor(label)
        

        # Applying random transformation
        transformations = []
        if self.transform:
            if self.hflip:
                transformations.append(transforms.RandomHorizontalFlip())
            if self.vflip:
                transformations.append(transforms.RandomVerticalFlip())

            if len(transformations) > 0:
                image = nn.Sequential(*transformations)(image)

        return image, label

    def get_image_list (self):
        """
            Get the list of original images

            Output:
            -------
            [str], list of images name
        """

        image_list = list(set([x["image_name"] for x in self.metadatas]))

        return image_list

    def get_subdataset (self, image_name):
        """
            Return subdataset of all images from an original image
        """

        new_dataset = copy.copy(self)
        new_dataset.metadatas = [x for x in new_dataset.metadatas if x["image_name"] == image_name]

        return new_dataset