import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import glob
import cv2

class CamVid(Dataset):
    def __init__(self, img_dir="./datasets/CamVid", mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.CAMVID_MEAN = [0.41315641836190786, 0.42286740495593333, 0.43007309808239147]
        self.CAMVID_STD = [0.3012653629575178, 0.3072708881246283, 0.3035014033609992]
        if mode=="train":
            self.train_raw = glob.glob(os.path.join(img_dir, "train", "*.png"))
            self.train_labels = glob.glob(os.path.join(img_dir, "train_labels", "*.png"))
        elif mode=="test":
            self.test_raw = glob.glob(os.path.join(img_dir, "test", "*"))
            self.test_labels = glob.glob(os.path.join(img_dir, "test_labels", "*.png"))
        elif mode=="valid":
            self.valid_raw = glob.glob(os.path.join(img_dir, "val", "*"))
            self.valid_labels = glob.glob(os.path.join(img_dir, "val_labels", "*.png"))
        
        # df = pd.read_csv(os.path.join(img_dir, "class_dict11.csv"))
        # self.label_dict = dict()
        # for x,rows in enumerate(df.iterrows()):
        #     rgb = [rows[1]['r'],rows[1]['g'],rows[1]['b']]
        #     self.label_dict[x] = rgb
        self.label_dict =  {
                            # "Void"
                            0: [[0, 0, 0]],
                            # "Sky"
                            1: [[128, 128, 128]],
                            # "Building"
                            2: [
                                    [0, 128, 64], #"Bridge"
                                    [128, 0, 0],  #... % "Building"
                                    [64, 192, 0],  #... % "Wall"
                                    [64, 0, 64],   #... % "Tunnel"
                                    [192, 0, 128], # ... % "Archway"
                                ],
                            # "Pole"
                            3:[
                                [192, 192, 128], # ... % "Column_Pole"
                                [0, 0, 64] # ... % "TrafficCone"
                            ],

                            # "Road"
                            4:[
                                [128, 64, 128], # ... % "Road"
                                [128, 0, 192], # ... % "LaneMkgsDriv"
                                [192, 0, 64] # ... % "LaneMkgsNonDriv"
                            ],

                            # "Pavement"
                            5:[
                                    [0, 0, 192], # ... % "Sidewalk" 
                                    [64, 192, 128], # ... % "ParkingBlock"
                                    [128, 128, 192], #... % "RoadShoulder"
                            ],
                            # "Tree"
                            6:[
                                [128, 128, 0], # ... % "Tree"
                                [192, 192, 0]     #... % "VegetationMisc"
                            ],

                            # "SignSymbol"
                            7:[
                                                [192, 128, 128], # ... % "SignSymbol"
                                                [128, 128, 64],    # ... % "Misc_Text"
                                                [0, 64, 64] # ... % "TrafficLight"
                            ],
                            # "Fence"
                            8:[
                                [64, 64, 128] #... % "Fence"
                            ],

                            # "Car"
                            9:[
                                [64, 0, 128], # ... % "Car"
                                [64, 128, 192], # ... % "SUVPickupTruck"
                                [192, 128, 192], # ... % "Truck_Bus"
                                [192, 64, 128], # ... % "Train"
                                [128, 64, 64] # ... % "OtherMoving"
                            ],
                            # "Pedestrian"
                            10:[
                                    [64, 64, 0], # ... % "Pedestrian"
                                    [192, 128, 64], # ... % "Child"
                                    [64, 0, 192], # ... % "CartLuggagePram"
                                    [64, 128, 64] # ... % "Animal"
                                ],
                            # "Bicyclist"
                            11:[
                                [0, 128, 192], # ... % "Bicyclist"
                                [192, 0, 192] #... % "MotorcycleScooter"
                            ]
                        }

    def _classEncode(self, target):
        output_shape = (target.shape[0], target.shape[1])
        output = np.zeros(output_shape)
        for label in self.label_dict.keys(): 
            for sub_label in self.label_dict[label]:
                channel = np.array(target == np.array([[sub_label]]), dtype=np.float32)
                channel = np.logical_and(np.logical_and(channel[:, :, 0], channel[:, :, 1]), channel[:, :, 2])*label
                output += channel
        return output
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_raw)
        if self.mode=='valid':
            return len(self.valid_raw)
        return len(self.test_raw)

    def __getitem__(self, idx):

        if self.mode=='train':
            pathX = self.train_raw
            pathY = self.train_labels
        elif self.mode=='test':
            pathX = self.test_raw
            pathY = self.test_labels
        else:
            pathX = self.valid_raw
            pathY = self.valid_labels
        img = cv2.imread(pathX[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(pathY[idx])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            aug = self.transform(image=img, mask=target)
            img = aug['image']
            target = aug['mask']
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.CAMVID_MEAN, self.CAMVID_STD)(img)
        return (img.cuda(), torch.cuda.LongTensor(self._classEncode(target)).squeeze())

