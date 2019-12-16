#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : baseline.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2019/12/13 11:44
@version: 1.0
@desc  :
This is a LeNet-5 implemented by pytorch on MNIST dataset
The code is heavily borrowed from pytorch tutorials.
[https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/neural_networks_tutorial.py]

A typical training procedure for a neural network is as follows:
- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch import optim
from torchvision.transforms import transforms


# LeNet Architecture
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_dataloader(data_path: str, batch_size: int):
    transform = transforms.ToTensor()

    # define train dataset
    trainset = tv.datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    # define test dataset
    testset = tv.datasets.MNIST(
        root='./data/',
        train=False,
        download=True,
        transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--train_dataset_path", default='../data/', type=str, help="MNIST path")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # set reandom seed
    torch.manual_seed(args.seed)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get train/test dataset
    trainloader, testloader = get_dataloader(args.train_dataset_path, args.batch_size)

    # define network
    net = LeNet()
    net.to(device)

    # define loss function which is MSE
    criterion = nn.CrossEntropyLoss()

    # define optimizer which is SGD
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    # train loop
    for epoch in range(args.train_epochs):
        print("epoch {}:".format(epoch))
        sum_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print loss for each 100 epochs
            sum_loss += loss.item()
            if i % 100 == 99:
                print('iterate %d loss: %.03f' % (i + 1, sum_loss / 100))
                sum_loss = 0.0

        # evaluate on test dataset for each epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # pick the best class
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('** accuracy on testset：%d%% ** \n' % (100 * correct / total))
