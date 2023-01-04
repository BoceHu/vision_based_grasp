#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: grasp.py

@Date: 2022/12/27
"""
import numpy as np
import math
import scipy.io as scio
from utils.model_related import mmcv

GRASP_WIDTH_MAX = 200.


class GraspMat:
    def __init__(self, file):
        self.grasp = scio.loadmat(file)['A']

    def height(self):
        return self.grasp.shape[1]

    def width(self):
        return self.grasp.shape[2]

    def crop(self, bbox):
        """
        crop the image
        :param bbox: list (x1, y1, x2, y2)
        """
        self.grasp = self.grasp[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def rescale(self, scale, interpolation='nearest'):
        origin_shape = self.grasp.shape[1]
        self.grasp = np.stack([mmcv.imrescale(grasp, scale, interpolation=interpolation)
                               for grasp in self.grasp])
        new_shape = self.grasp.shape[1]
        ratio = new_shape / origin_shape

        self.grasp[2, :, :] = self.grasp[2, :, :] * ratio

    def rotate(self, rota):
        """
        clockwise
        :param rota: degree
        :return:
        """
        self.grasp = np.stack([mmcv.imrotate(grasp, rota) for grasp in self.grasp])

        rota = rota / 180. * np.pi
        self.grasp[1, :, :] -= rota
        self.grasp[1, :, :] = self.grasp[1, :, :] % (np.pi * 2)
        self.grasp[1, :, :] *= self.grasp[0, :, :]

    def _flipAngle(self, angle_mat, confidence_mat):
        angle_out = (angle_mat // math.pi) * 2 * math.pi + math.pi - angle_mat
        angle_out = angle_out * confidence_mat
        angle_out = angle_out % (2 * math.pi)

        return angle_out

    def flip(self):
        """
        horizontal flip
        """
        self.grasp = np.stack([mmcv.imflip(grasp, direction='horizontal') for grasp in self.grasp])
        self.grasp[1, :, :] = self._flipAngle(self.grasp[1, :, :], self.grasp[0, :, :])

    def encode(self):
        self.grasp[1, :, :] = (self.grasp[1, :, :] + 2 * math.pi) % math.pi

        self.grasp_point = self.grasp[0, :, :]
        self.grasp_cos = np.cos(self.grasp[1, :, :] * 2)
        self.grasp_sin = np.sin(self.grasp[1, :, :] * 2)
        self.grasp_width = self.grasp[2, :, :]
