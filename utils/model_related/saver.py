#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: saver.py

@Date: 2022/12/29
"""
import os
import cv2
import sys
import torch
import glob
import numpy as np
import tensorboardX
from pytorch_model_summary import summary


class Saver:
    def __init__(self, path, logdir, modeldir, imgdir, net_description):
        self.path = path  # save path
        self.logdir = logdir  # tensorboard
        self.modeldir = modeldir  # model
        self.imgdir = imgdir  # img
        self.net_description = net_description

    def save_summary(self):
        """
        save tensorboard
        :return:
        """
        save_folder = os.path.join(self.path, self.logdir, self.net_description)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        return tensorboardX.SummaryWriter(save_folder)

    def save_arch(self, net, shape):
        """
        save model architecture to self.path/arch.txt
        :param net:
        :param shape:
        :return:
        """
        with open(os.path.join(self.path, 'arch.txt'), 'w') as f:
            sys.stdout = f
            summary(net, shape)
            sys.stdout = sys.__stdout__

    def save_model(self, net, model_name):
        """
        save model
        :param net:
        :param model_name:
        :return:
        """
        model_path = os.path.join(self.path, self.modeldir, self.net_description)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, model_name))

    def remove_model(self, num):
        model_path = os.path.join(self.path, self.modeldir, self.net_description)
        models = glob.glob(model_path + '/*_.pth')
        models.sort()
        if len(models) > num:
            for file in models[:len(models) - num]:
                os.remove(file)

    def save_img(self, epoch, idx, imgs):
        able_out_1_255 = imgs[1].copy() * (255 / imgs[1].max())
        able_out_1_255 = able_out_1_255.astype(np.uint8)

        able_y, _, _ = imgs[2]
        able_y = able_y.cpu().numpy().squeeze()  # (1, 1, 360, 360) -> (360, 360)

        able_y_255 = able_y.copy() * 255
        able_y_255 = able_y_255.astype(np.uint8)

        save_folder = os.path.join(self.path, self.imgdir, self.net_description)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        able_out_1_filename = os.path.join(save_folder, '{}_{}_{:03d}.jpg'.format('able1', idx, epoch))
        able_y_filename = os.path.join(save_folder, '{}_{}.jpg'.format('abley', idx))

        cv2.imwrite(able_out_1_filename, able_out_1_255)
        cv2.imwrite(able_y_filename, able_y_255)
