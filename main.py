import os
from dataset import StartingDataset
import torch
from train import train
from cnn import CNNModel


def main():

    data = StartingDataset("/Users/yang/Documents/hoth/hoth2022/fer2013.csv")
    #print("Done")
    train_dataset = []
    test_dataset = []
    for emotion, image, datatype in data:
        if datatype == "Training":
            train_dataset.append((image, emotion))
        else:
            test_dataset.append((image, emotion))

    model = CNNModel()

    train(train_dataset, model)

if (__name__ == "__main__"):
    main()

