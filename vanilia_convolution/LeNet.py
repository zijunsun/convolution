#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : LeNet.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2019/12/13 17:02
@version: 1.0
@desc  : Handwritten digit recognition
"""
import argparse
import random

import numpy as np
import torch
import torchvision as tv
from torchvision.transforms import transforms

from vanilia_convolution.activate_function import ReLU, Softmax
from vanilia_convolution.loss_function import CrossEntropy
from vanilia_convolution.neural_layer import Net, Convolution, MaxPooling, FC
from vanilia_convolution.processor import padding, one_hot_label


# LeNet Architecture
class LeNet(Net):
    def __init__(self):
        self.matrix = None
        self.conv1 = Convolution(in_channel=1, kernal_size=3, out_channel=6)
        self.relu1 = ReLU()
        self.pool1 = MaxPooling(kernal_size=2)

        self.conv2 = Convolution(in_channel=6, kernal_size=3, out_channel=16)
        self.relu2 = ReLU()
        self.pool2 = MaxPooling(kernal_size=2)

        self.flatten = None

        self.fc1 = FC(input_dim=16 * 6 * 6, out_dim=120)
        self.relu3 = ReLU()

        self.fc2 = FC(input_dim=120, out_dim=84)
        self.relu4 = ReLU()

        self.fc3 = FC(input_dim=84, out_dim=10)
        self.softmax = Softmax()

    def forward(self, matrix):
        self.matrix = matrix
        # conv layer 1
        conv1 = self.conv1.forward(matrix)
        relu1 = self.relu1.forward(conv1)  # (m*6*30*30)
        pool1 = self.pool1.forward(relu1)  # (m*6*15*15)

        # conv layer 2
        conv2 = self.conv2.forward(pool1)
        relu2 = self.relu2.forward(conv2)  # (m*16*13*13)
        pool2 = self.pool2.forward(relu2)  # (m*16*6*6)

        self.flatten = pool2.reshape(pool2.shape[0], -1).T  # (567*m)

        # fc layer 1
        fc1_Z = self.fc1.forward(self.flatten)
        relu3 = self.relu3.forward(fc1_Z)  # (120*m1)

        # fc layer 2
        fc2_Z = self.fc2.forward(relu3)
        relu4 = self.relu4.forward(fc2_Z)  # (84*m)

        # fc layer 3
        fc3_Z = self.fc3.forward(relu4)  # (10*m)

        # softmax layer
        y = self.softmax.forward(fc3_Z.T)  # (m*10)
        return y

    def backward(self, y, label, lr):
        # calculate gradients of each layer
        # softmax layer
        delt_fc3_Z = self.softmax.backward(y, label).T  # (10*m)

        # fc layer 3
        delt_fc3_weight, delt_fc3_bias, delt_relu4 = self.fc3.backward(delt_fc3_Z)

        # fc layer 2
        delt_fc2_Z = delt_relu4 * self.relu4.backward(self.relu4.out_matrix)  # (84*m)
        delt_fc2_weight, delt_fc2_bias, delt_relu3 = self.fc2.backward(delt_fc2_Z)

        # fc layer 1
        delt_fc1_Z = delt_relu3 * self.relu3.backward(self.relu3.out_matrix)  # (120*m)
        delt_fc1_weight, delt_fc1_bias, delt_flatten = self.fc1.backward(delt_fc1_Z)

        delt_pool2 = delt_flatten.T.reshape(self.pool2.P.shape)  # (m*16*6*6)

        # conv layer 2
        delt_relu2 = self.pool2.backward(delt_pool2)  # (m*16*13*13)
        delt_conv2 = delt_relu2 * self.relu2.backward(self.relu2.out_matrix)  # (m*16*13*13)
        delt_conv2_weight, delt_conv2_bias, delt_pool1 = self.conv2.backward(delt_conv2)

        # conv layer 1
        delt_relu1 = self.pool1.backward(delt_pool1)  # (m*28*28*6)
        delt_conv1 = delt_relu1 * self.relu1.backward(self.relu1.out_matrix)  # (m*28*28*6)
        delt_conv1_weight, delt_conv1_bias, delt_matrix = self.conv1.backward(delt_conv1)

        # upgrade gradient
        self.fc3.weights -= lr * delt_fc3_weight
        self.fc3.bias -= lr * delt_fc3_bias

        self.fc2.weights -= lr * delt_fc2_weight
        self.fc2.bias -= lr * delt_fc2_bias

        self.fc1.weights -= lr * delt_fc1_weight
        self.fc1.bias -= lr * delt_fc1_bias

        self.conv2.weights -= lr * delt_conv2_weight
        self.conv2.bias -= lr * delt_conv2_bias

        self.conv1.weights -= lr * delt_conv1_weight
        self.conv1.bias -= lr * delt_conv1_bias


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


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--train_dataset_path", default='../data/', type=str, help="MNIST path")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # set reandom seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # get train/test dataset
    trainloader, testloader = get_dataloader(args.train_dataset_path, args.batch_size)

    # define network
    net = LeNet()

    # define loss function which is MSE
    criterion = CrossEntropy()

    # train loop
    for epoch in range(args.train_epochs):
        print("epoch {}:".format(epoch))
        sum_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = padding(inputs.numpy(), 2)  # zero pad (28*28)->(32*32)
            labels = one_hot_label(labels.numpy())

            # forward + backward + optimize
            outputs = net.forward(inputs)
            loss = criterion.loss(outputs, labels)
            net.backward(outputs, labels, args.learning_rate)

            # print loss for each 100 epochs
            sum_loss += loss
            if i % 100 == 99:
                print('iterate %d loss: %.03f' % (i + 1, sum_loss / 100))
                sum_loss = 0.0

        # evaluate on test dataset for each epoch
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images = padding(images.numpy(), 2)  # zero pad (28*28)->(32*32)
            labels = labels.numpy()
            outputs = net.forward(images)
            # pick the best class
            predicts = outputs.argmax(axis=1)
            total += labels.shape[0]
            correct += (predicts == labels).sum()
        print('** accuracy on testset：%d%% ** \n' % (100 * correct / total))
