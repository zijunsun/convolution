#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : activate_function.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2019/12/13 11:37
@version: 1.0
"""
import numpy as np
from numpy import ndarray


class Activator(object):
    def forward(self, x):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError


class ReLU(Activator):
    """
    f(x) = x (x>0)
    f(x) = 0 (x<=0)
    """

    def __init__(self):
        self.out_matrix = None

    def forward(self, x: ndarray):
        self.out_matrix = np.maximum(0, x)
        return self.out_matrix

    def backward(self, x: ndarray):
        return 1 * (x > 0)


class Sigmod(Activator):
    """
    f(x) = 1/ 1+e^(-x)
    """

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))


class Softmax(Activator):

    def forward(self, x):
        tmp = np.max(x, axis=0)
        x -= tmp
        z = np.exp(x)
        tmp = np.sum(z, axis=0)
        z /= tmp
        return z

    def backward(self, y, label):
        return y - label


if __name__ == '__main__':
    relu = ReLU()
    a = np.array([1, 2, 3])
    print(relu.forward(a))
