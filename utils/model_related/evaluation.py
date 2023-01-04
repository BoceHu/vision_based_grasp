#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: evaluation.py

@Date: 2022/12/27
"""
import math

import numpy as np
from utils.dataset_loader.grasp import GRASP_WIDTH_MAX


def length(pt1, pt2):
    """

    :param pt1:
    :param pt2:
    :return:
    """
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)


def evaluation(pred_pos, pred_agl, pred_wid, target_pos, target_agl, target_wid):
    """

    :param pred_pos:  (h,w)
    :param pred_agl:  (h,w)
    :param pred_wid:  (h,w)
    :param target_pos:  (1,h,w)
    :param target_agl:  (1,h,w)
    :param target_wid:  (1,h,w)
    :return: 0 (wrong) or 1 (correct)
    condition:
    (1) Grab point distance less than 5 pixels
    (2) The angle difference is less than or equal to 30°
    (3) Grab width ratio is in [0.8, 1.2]
    """
    # threshold
    thresh_pos = 0.3
    thresh_pt = 5
    thresh_angle = 30 / 180 * math.pi
    thresh_wid = 0.8

    # label
    target_pos = target_pos.cpu().numpy().squeeze()
    target_agl = target_agl.cpu().numpy().squeeze()
    target_wid = target_wid.cpu().numpy().squeeze() * GRASP_WIDTH_MAX

    if np.max(target_pos) < 1:
        return 1
    if np.max(pred_pos) < thresh_pos:
        return 0

    loc = np.argmax(pred_pos)
    pred_pt_row = loc // pred_pos.shape[1]
    pred_pt_col = loc % pred_pos.shape[1]
    pred_agl = (pred_agl[pred_pt_row, pred_pt_col] + 2 * math.pi) % math.pi
    pred_wid = pred_wid[pred_pt_row, pred_pt_col]

    H, W = pred_pos.shape
    search_l = max(pred_pt_col - thresh_pt, 0)
    search_r = min(pred_pt_col + thresh_pt, W - 1)
    search_t = max(pred_pt_row - thresh_pt, 0)
    search_b = min(pred_pt_row + thresh_pt, H - 1)

    for target_row in range(search_t, search_b + 1):
        for target_col in range(search_l, search_r + 1):
            if target_pos[target_row, target_col] != 1.0:
                continue

            if length([target_row, target_col], [pred_pt_row, pred_pt_col]) > thresh_pt:
                continue

            label_angle = (target_agl[target_row, target_col] + 2 * math.pi) % math.pi

            if abs(label_angle - pred_agl) > thresh_angle and abs(label_angle - pred_agl) < (math.pi - thresh_angle):
                continue

            label_width = target_wid[target_row, target_col]

            if (pred_wid / label_width) >= thresh_wid and (pred_wid / label_width) <= 1. / thresh_wid:
                return 1

    return 0
