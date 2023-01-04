#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: img.py

@Date: 2022/12/28
"""
import cv2
import numpy as np
from utils.model_related import mmcv


class DepthImage:
    def __init__(self, file):
        self.img = cv2.imread(file, -1)

    def height(self):
        return self.img.shape[0]

    def width(self):
        return self.img.shape[1]

    def crop(self, size, dist=-1):
        if dist > 0:
            x_offset = np.random.randint(-1 * dist, dist)
            y_offset = np.random.randint(-1 * dist, dist)
        else:
            x_offset = 0
            y_offset = 0

        crop_x1 = int((self.width() - size) / 2 + x_offset)
        crop_y1 = int((self.height() - size) / 2 + y_offset)
        crop_x2 = crop_x1 + size
        crop_y2 = crop_y1 + size

        self.img = self.img[crop_y1:crop_y2, crop_x1:crop_x2]

        return crop_x1, crop_y1, crop_x2, crop_y2

    def rescale(self, scale, interpolation='bilinear'):
        self.img = mmcv.imrescale(self.img, scale, interpolation=interpolation)

    def rotate(self, rota):
        self.img = mmcv.imrotate(self.img, rota, border_value=float(self.img.max()))

    def flip(self, flip_direction='horizontal'):
        self.img = mmcv.imflip(self.img, direction=flip_direction)

    def normalize(self):
        self.img = np.clip((self.img - self.img.mean()), -1, 1)
