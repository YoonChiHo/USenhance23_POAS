import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import cv2
import numpy as np
import torch
import json


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase


        if opt.data_type == 'all':
            data_root = f'{opt.dataroot}'
        else:
            data_root = f'{opt.dataroot}_{opt.data_type}'
        self.dir_A = os.path.join(data_root, opt.phase + 'A')  # create a path '/dataroot/trainA'
        self.dir_B = os.path.join(data_root, opt.phase + 'B')  # create a path '/dataroot/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        self.label_path = os.path.join('../datasets', "label_inform.json") #TODO: For MTL Classification Labeling
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        if opt.data_norm == 'ab_seperate':
            self.data_info_path = f"{data_root}/data.json"
            with open(self.data_info_path, "r") as json_file:
                loaded_data = json.load(json_file)
            print("Loaded Data:", loaded_data)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
        if self.opt.resizeBig:
            A_img = cv2.resize(A_img, (512, 512))

        A_img_arr = np.array(A_img).reshape((1,) + A_img.shape)
        
        
        B_img = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE)
        if self.opt.resizeBig:
            B_img = cv2.resize(B_img, (512, 512))
        B_img_arr = np.array(B_img).reshape((1,) + B_img.shape)

        def horizontal_flip(image_array1, image_array2):
            if np.random.rand() < 0.5:  # 0.5의 확률로 뒤집기 수행
                flipped_image1 = np.flip(image_array1, axis=2)
                flipped_image2 = np.flip(image_array2, axis=2)
                return flipped_image1, flipped_image2
            else:
                return image_array1, image_array2
        if self.opt.lr_flip:
            A_img_arr, B_img_arr = horizontal_flip(A_img_arr, B_img_arr)

        if self.opt.data_norm == 'basic':
            A_img_arr_norm = ((A_img_arr / 255.) * 2) - 1
            B_img_arr_norm = ((B_img_arr / 255.) * 2) - 1
        elif self.opt.data_norm == 'ab_seperate':
            with open(self.data_info_path, "r") as json_file:
                loaded_data = json.load(json_file)
            A_img_arr_norm = ((A_img_arr / 255.) - loaded_data['TrainA'][0]) / loaded_data['TrainA'][1]
            B_img_arr_norm = ((B_img_arr / 255.) - loaded_data['TrainB'][0]) / loaded_data['TrainB'][1]
        A = torch.from_numpy(A_img_arr_norm).float()
        B = torch.from_numpy(B_img_arr_norm).float()
        
        if self.phase == "train":
        
            # read json file
            with open(self.label_path, 'r') as f:
                label_inform = json.load(f)
            label = label_inform[os.path.basename(B_path)[:-4]] #TODO: Class나눠서 학습할땐 라벨을 달리해야함 
            if self.opt.data_type == 'typeA' and 'low2high_v3' in self.opt.dataroot:#(TypeA[2,5] -> [1,2],
                if label in [1,3,4]: print(f'ERROR: Not Supported Label {label} for typeA')
                if label == 2: label = 1
                if label == 5: label = 2
            elif self.opt.data_type == 'typeA':     #(TypeA[1,2,5] -> [1,2,3],
                if label in [3,4]: print(f'ERROR: Not Supported Label {label} for typeA')
                if label == 5: label = 3
            elif self.opt.data_type == 'typeB':    # TypeB[3,4] -> [1,2])
                if label in [1,2,5]: print(f'ERROR: Not Supported Label {label} for typeB')
                if label == 3: label = 1
                if label == 4: label = 2
            
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}, label
        
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
