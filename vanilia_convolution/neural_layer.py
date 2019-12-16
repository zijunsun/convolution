#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : neural_layer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2019/12/13 16:52
@version: 1.0
@desc  : Implementation of  fundamental layer of deep neural network
"""
import math

import numpy as np
import torch
from torch.nn import init

np.random.seed(4)


# to compare with pytorch, we use the same init function as pytoch uses
def reset_paramters(weight_shape, bias_shape):
    weight = torch.rand(weight_shape)
    bias = torch.rand(bias_shape)

    init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(bias, -bound, bound)
    return weight.numpy(), bias.numpy()


class Net(object):
    def forward(self, matrix):
        raise NotImplementedError

    def backward(self, delt_z):
        raise NotImplementedError


class Convolution(Net):
    """
    Basic convolution arithmetic

        A_pre = activate(Z_pre) ---> pre activate layer
        P_pre = pooling(A_pre)  ---> pre pooling layer
        Z = W * P_pre + b       ---> convolution layer (here)
        A = activate(Z)         ---> activate layer
        P = pooling(A)          ---> pooling layer
    """

    def __init__(self, in_channel, kernal_size, out_channel):
        # shape
        self.in_channel = in_channel
        self.kernal_size = kernal_size
        self.out_channel = out_channel
        # init parameters
        self.weights, self.bias = reset_paramters((out_channel, in_channel, kernal_size, kernal_size), (out_channel, 1))
        # cache matrix
        self.Z = None
        self.P_pre = None

    def forward(self, P_pre):
        # get origin shape of image
        m, c, h, w = P_pre.shape
        self.P_pre = P_pre

        # calculate output shape after convolution
        h_steps = h - self.kernal_size + 1
        w_steps = w - self.kernal_size + 1
        self.Z = np.zeros((m, self.out_channel, h_steps, w_steps))
        out_matrix_m, out_matrix_c, out_matrix_h, out_matrix_w = self.Z.shape

        # convolution
        for i in range(out_matrix_m):
            for c in range(out_matrix_c):
                for h in range(out_matrix_h):
                    for w in range(out_matrix_w):
                        X = P_pre[i, :, h:h + self.kernal_size, w:w + self.kernal_size]
                        W = self.weights[c][:][:][:]
                        B = self.bias[c]
                        assert X.shape == W.shape
                        z = float(np.sum(W * X) + B)
                        self.Z[i, c, h, w] = z
        return self.Z

    def backward(self, delt_Z):
        """
            :param delt_Z: gradient of Z, backward from activate layer
            :return: gradients of W, b and A_pre
        """
        # get matrix shape
        out_channel, in_channel, kernal_size, kernal_size = self.weights.shape
        m, n_C, n_H, n_W = delt_Z.shape

        # init zeros matrix to store gradient
        delt_P_pre = np.zeros(self.P_pre.shape)
        delt_weight = np.zeros((out_channel, in_channel, kernal_size, kernal_size))
        delt_bias = np.zeros((out_channel, 1))

        # calculate gradient
        for i in range(m):
            for c in range(n_C):
                for h in range(n_H):
                    for w in range(n_W):
                        a_slice = self.P_pre[i, :, h:h + kernal_size, w:w + kernal_size]
                        delt_P_pre[i, :, h:h + kernal_size, w:w + kernal_size] += self.weights[c, :, :, :] \
                                                                                  * delt_Z[i, c, h, w]
                        delt_weight[c, :, :, :] += delt_Z[i, c, h, w] * a_slice
                        delt_bias[c, :] += delt_Z[i, c, h, w]

        # gradients mean
        delt_weight = delt_weight / m
        delt_bias = delt_bias / m

        return delt_weight, delt_bias, delt_P_pre


class MaxPooling(Net):
    """
    Max Pooling arithmetic

        A = activate(Z)         ---> activate layer
        P = pooling(A)          ---> pooling layer (here)
        Z_next = W * P + b      ---> convolution layer
    """

    def __init__(self, kernal_size):
        # shape
        self.kernal_size = kernal_size
        # cache matrix
        self.mask_A = None
        self.A = None
        self.P = None

    def forward(self, A):
        # get origin shape of matrix
        m, c, h, w = A.shape

        # calculate output shape after pooling
        self.A = A
        self.mask_A = np.zeros(A.shape)  # mask matrix store location
        h_steps = int(h / 2)
        w_steps = int(w / 2)
        self.P = np.zeros((m, c, h_steps, w_steps))
        out_matrix_m, out_matrix_c, out_matrix_h, out_matrix_w = self.P.shape

        # max pooling, find the max value on (kernal_size*kernal_size) window and store location in self.mask_matrix
        for i in range(out_matrix_m):
            for c in range(out_matrix_c):
                for h in range(out_matrix_h):
                    for w in range(out_matrix_w):
                        h_start = h * self.kernal_size
                        w_start = w * self.kernal_size
                        slice_matrix = A[i, c, h_start:h_start + 2, w_start:w_start + 2]
                        self.P[i, c, h, w] = np.max(slice_matrix)
                        location = np.where(slice_matrix == np.max(slice_matrix))
                        self.mask_A[i, c, h_start + location[0][0], w_start + location[1][0]] = 1
        return self.P

    def backward(self, delt_pre):
        # init zeros matrix to store gradient
        self.delt_A = np.zeros(self.A.shape)

        # calculate gradients
        out_matrix_m, out_matrix_c, out_matrix_h, out_matrix_w = self.P.shape
        for i in range(out_matrix_m):
            for c in range(out_matrix_c):
                for h in range(out_matrix_h):
                    for w in range(out_matrix_w):
                        h_start = h * self.kernal_size
                        w_start = w * self.kernal_size
                        self.delt_A[i, c, h_start:h_start + 2, w_start:w_start + 2] = delt_pre[i, c, h, w]

        delt_pool = self.delt_A * self.mask_A
        return delt_pool


class FC(Net):
    """
    Basic fully connect arithmetic

        A_pre = activate(Z_pre) ---> pre activate layer
        Z = W * A_pre + b       ---> fc layer (here)
        A = activate(Z)         ---> activate layer
    """

    def __init__(self, input_dim, out_dim):
        # init parameters
        self.weights, self.bias = reset_paramters((out_dim, input_dim), (out_dim, 1))
        # cache matrix
        self.Z = None
        self.A_pre = None

    def forward(self, A_pre):
        self.Z = np.dot(self.weights, A_pre) + self.bias  # W* A_pre + b = Z
        self.A_pre = A_pre
        return self.Z

    def backward(self, delt_Z):
        """
        :param delt_Z: gradient of Z, backward from activate layer
        :return: gradients of W, b and A_pre
        """
        # calculate gradients
        delt_A_pre = np.dot(self.weights.transpose(), delt_Z)
        delt_weight = np.dot(delt_Z, self.A_pre.transpose())
        delt_bias = delt_Z.sum(axis=1).reshape(-1, 1)

        # gradients mean
        m = delt_Z.shape[1]
        delt_weight = delt_weight / m
        delt_bias = delt_bias / m

        return delt_weight, delt_bias, delt_A_pre
