from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



def train(train_dataset, test_dataset, model):

    model.train()
    #print(np.shape(train_dataset))

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False)
    Optimizer = optim.Adam(model.parameters())
    Loss = nn.CrossEntropyLoss()


    for epoch in range(10):

        model.train()

        for data in iter(train_data):

            batch_inputs, batch_labels = data
            Optimizer.zero_grad()
            predictions = model(batch_inputs)
            current_loss = Loss(predictions, batch_labels)
            current_loss.backward()

            Optimizer.step()

            Optimizer.zero_grad()
        print("Epoch: {} Loss: {}".format(epoch+1, current_loss))
        model.eval()

        for data in iter(test_data):
            inputs, labels = data
            outputs = model(inputs)
            predictions =  torch.argmax(outputs, dim = 1)
            print("Accuracy: ", compute_accuracy(predictions, labels))


def compute_accuracy(predictions, labels):
    correct = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]

        if prediction in (0, 1, 2, 4) and label in (0, 1, 2, 4):
            correct += 1
        elif prediction in (3, 5) and label in (3, 5):
            correct += 1
        elif prediction == label:
            correct += 1
    total = len(predictions)

    return correct/total