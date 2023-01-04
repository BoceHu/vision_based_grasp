#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: common.py

@Date: 2022/12/28
"""
import torch
from skimage.filters import gaussian
from utils.dataset_loader.grasp import GRASP_WIDTH_MAX


def post_process_output(p_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param p_img: position output of ggcnn
    :param cos_img: cos output of ggcnn
    :param sin_img: sin output of ggcnn
    :param width_img: width output of ggcnn
    :return: post-processing p_img, angle, width_img
    """
    p_img = p_img.cpu().numpy().squeeze()
    cos_img = cos_img * 2 - 1
    sin_img = sin_img * 2 - 1
    angle = (torch.atan2(sin_img, cos_img) / 2.).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * GRASP_WIDTH_MAX

    p_img = gaussian(p_img, 2.0, preserve_range=True)
    angle = gaussian(angle, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return p_img, angle, width_img
