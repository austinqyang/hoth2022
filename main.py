import os
from dataset import StartingDataset
import torch
from train import train
from cnn import CNNModel
import numpy as np


def trainer(model):
    #print("running")
    data = StartingDataset("../hoth2022/fer2013.csv")
    #print("Done")
    train_dataset = []
    test_dataset = []
    for emotion, image, datatype in data:
        if datatype == "Training":
            train_dataset.append([image, emotion])
        else:
            test_dataset.append([image, emotion])

    #print(np.shape(test_dataset))

    train(train_dataset, test_dataset, model)

def main():

    model = CNNModel()

    trainer(model)
    torch.save(model, "../hoth2022/modelparam.pth")

    

def emotioncheck(PATH):
    model = torch.load("../hoth2022/modelparam.pth")
    model.eval()

    prediction = model(PATH)

    if prediction in (0, 1, 2, 4):
        return "Sad"
    elif prediction in (3, 5):
        return "Happy"
    else:
        return "Neutral"
