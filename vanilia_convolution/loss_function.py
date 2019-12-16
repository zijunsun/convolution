#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : loss_function.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2019/12/13 16:58
@version: 1.0
"""
import numpy as np


class Loss(object):
    def loss(self, y, label):
        raise NotImplementedError


class CrossEntropy(Loss):
    def loss(self, y, label):
        loss = -np.sum(label * np.log(y)) / y.shape[0]
        return loss
