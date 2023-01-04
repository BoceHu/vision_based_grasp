#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: ggcnn.py

@Date: 2023/1/2
"""
import torch
import time
import math
from skimage.feature import peak_local_max
import numpy as np
from models.common import post_process_output
from models.loss import get_pred
from models import get_network


def input_img(img, out_size=300):
    """
    crop the image，reserve (out_size, out_size)
    :param img: depth img
    :return: img -> tensor
    """

    assert img.shape[0] >= out_size and img.shape[
        1] >= out_size, 'The size of the depth image should be equal or greater than the output size'

    # crop
    crop_x1 = int((img.shape[1] - out_size) / 2)
    crop_y1 = int((img.shape[0] - out_size) / 2)
    crop_x2 = crop_x1 + out_size
    crop_y2 = crop_y1 + out_size
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # normalize
    img = np.clip(img - img.mean(), -1., 1.).astype(np.float32)

    # numpy to tensor (out_size,out_size) -> (1, 1, out_size,out_size)
    tensor = torch.from_numpy(img[np.newaxis, np.newaxis, :, :])

    return tensor, crop_x1, crop_y1


def arg_thresh(array, thresh):
    """
    :param array: 2D - array
    :param thresh: float thresh
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))  # [[row, col],..., [row, col]]
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i + 1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]  # sort

    return locs


class GGCNN:
    def __init__(self, model, device, network='ggcnn2'):
        self.t = 0
        self.num = 0
        self.device = device
        print('>> loading GGCNN2')
        ggcnn = get_network(network)
        self.net = ggcnn()
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)
        self.net = self.net.to(device)
        print('>> load done')

    def fps(self):
        return 1.0 / (self.t / self.num)

    def predict(self, img, mode, thresh=0.3, peak_dist=1):
        """
        prediction
        :param img: depth np.array (h, w)
        :param thresh: 置信度阈值
        :param peak_dist: 置信度筛选峰值
        :return:
            pred_grasps: list([row, col, angle, width])
            crop_x1
            crop_y1
        """
        # crop images
        input, self.crop_x1, self.crop_y1 = input_img(img)

        t1 = time.time()
        # predict
        self.pos_out, self.cos_out, self.sin_out, self.wid_out = get_pred(self.net, input.to(self.device))
        t2 = time.time() - t1

        # post processing
        pos_pred, ang_pred, wid_pred = post_process_output(self.pos_out, self.cos_out, self.sin_out, self.wid_out)
        if mode == 'peak':
            # peaks
            pred_pts = peak_local_max(pos_pred, min_distance=peak_dist, threshold_abs=thresh)
        elif mode == 'all':
            # thresh
            pred_pts = arg_thresh(pos_pred, thresh=thresh)
        elif mode == 'max':
            # max
            loc = np.argmax(pos_pred)
            row = loc // pos_pred.shape[0]
            col = loc % pos_pred.shape[0]
            pred_pts = np.array([[row, col]])
        else:
            raise ValueError

        pred_grasps = []
        for idx in range(pred_pts.shape[0]):
            row, col = pred_pts[idx]
            angle = (ang_pred[row, col] + 2 * math.pi) % math.pi
            width = wid_pred[row, col]
            row += self.crop_y1
            col += self.crop_x1

            pred_grasps.append([row, col, angle, width])

        self.t += t2
        self.num += 1

        return pred_grasps, self.crop_x1, self.crop_y1
