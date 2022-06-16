import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import glob
import cv2

class CamVid(Dataset):
    def __init__(self, img_dir="CamVid", mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        
        if mode=="train":
            self.train_raw = glob.glob(os.path.join(img_dir, "train", "*"))
            self.train_labels = glob.glob(os.path.join(img_dir, "train_labels", "*"))
        elif mode=="test":
            self.test_raw = glob.glob(os.path.join(img_dir, "test", "*"))
            self.test_labels = glob.glob(os.path.join(img_dir, "test_labels", "*"))
        elif mode=="valid":
            self.valid_raw = glob.glob(os.path.join(img_dir, "val", "*"))
            self.valid_labels = glob.glob(os.path.join(img_dir, "val_labels", "*"))
        
        df = pd.read_csv(os.path.join(img_dir, "class_dict11.csv"))
        self.label_dict = dict()
        for x,rows in enumerate(df.iterrows()):
            rgb = [rows[1]['r'],rows[1]['g'],rows[1]['b']]
            self.label_dict[x] = rgb

    def oneHot(self, target):
        output_shape = (len(self.label_dict.keys()), target.shape[0], target.shape[1])
        output = np.zeros(output_shape)
        for i, label in enumerate(self.label_dict.keys()):  
            channel = np.array((target == np.array([[self.label_dict[label]]])), dtype=np.float32)
            channel = np.logical_and(np.logical_and(channel[:, :, 0], channel[:, :, 1]), channel[:, :, 2])
            output[i, :, :] = channel
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
            img, target = self.transform(image=img, mask=target)
        
        return (transforms.ToTensor()(img).cuda(), transforms.ToTensor()(self.oneHot(target)).cuda())
