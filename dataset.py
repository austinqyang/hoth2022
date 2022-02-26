import os
import torch
import pandas as pd
import torchvision
import numpy as np


class StartingDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv("/Users/yang/Documents/hoth/hoth2022/fer2013.csv")
        self.data.rename(columns=self.data.iloc[0]).drop(self.data.index[0])
        self.emotion = self.data.iloc[:, 0]
        self.emotion = np.array(self.emotion)
        self.images = self.data.iloc[:, 1]
        self.images = np.array(list(map(str.split, self.images)), np.float32)
        self.images /= 255
        self.images = self.images.reshape(-1, 1, 48, 48)
        self.datatype = self.data.iloc[:, 2]

    def __getitem__(self, index):
        return self.emotion[index], self.images[index], self.datatype[index]

    def __len__(self):
        return len(self.emotion)
