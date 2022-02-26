from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim



def train(train_dataset, model):

    model.train()

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
    Optimizer = optim.Adam(model.parameters())
    Loss = nn.CrossEntropyLoss()


    for epoch in range(10):
        for data in iter(train_data):
            batch_inputs, batch_labels = data
            Optimizer.zero_grad()
            predictions = model(batch_inputs)
            current_loss = Loss(predictions, batch_labels)
            current_loss.backward()

            Optimizer.step()

            Optimizer.zero_grad()
        print("Epoch: {} Loss: {}".format(epoch+1, current_loss))

    print("Done")