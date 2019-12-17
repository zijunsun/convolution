#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : processor.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2019/12/13 17:54
@version: 1.0
"""
import numpy as np


def padding(images, pad):
    """zero padding for images"""
    m = images.shape[0]
    c = images.shape[1]
    h = images.shape[2]
    w = images.shape[3]
    pad_images = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    assert pad_images.shape == (m, c, h + 2 * pad, w + 2 * pad)
    return pad_images


def one_hot_label(label):
    """
    convert the label to one-hot vector
    label : [2,1,0] ->  one-hot vector :[[0,0,1], [0,1,0], [1,0,0]]
    not : this is a specific converter for mnist dataset which only have 10 classes
    """
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab
