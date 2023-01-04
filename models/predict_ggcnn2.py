#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: predict_ggcnn2.py

@Date: 2023/1/1
"""
import cv2
import torch
import math
import numpy as np
from models.common import post_process_output
from models.loss import get_pred
from models.ggcnn2 import GGCNN2
from skimage.draw import line


def ptsOnRect(pts):
    """
    get points of five lines on the rectangle
    five lines: 4 edges and 1 diagonal
    pts: np.array, shape=(4, 2) (row, col)
    """
    rows1, cols1 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[1, 0]), int(pts[1, 1]))
    rows2, cols2 = line(int(pts[1, 0]), int(pts[1, 1]), int(pts[2, 0]), int(pts[2, 1]))
    rows3, cols3 = line(int(pts[2, 0]), int(pts[2, 1]), int(pts[3, 0]), int(pts[3, 1]))
    rows4, cols4 = line(int(pts[3, 0]), int(pts[3, 1]), int(pts[0, 0]), int(pts[0, 1]))
    rows5, cols5 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[2, 0]), int(pts[2, 1]))

    rows = np.concatenate((rows1, rows2, rows3, rows4, rows5), axis=0)
    cols = np.concatenate((cols1, cols2, cols3, cols4, cols5), axis=0)
    return rows, cols


def ptsOnRotateRect(pt1, pt2, w):
    """
    draw a rectangle
    pt1: [row, col]
    w: width
    img:
    """
    y1, x1 = pt1
    y2, x2 = pt2

    if x2 == x1:
        if y1 > y2:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        tan = (y1 - y2) / (x2 - x1)
        angle = np.arctan(tan)

    points = []
    points.append([y1 - w / 2 * np.cos(angle), x1 - w / 2 * np.sin(angle)])
    points.append([y2 - w / 2 * np.cos(angle), x2 - w / 2 * np.sin(angle)])
    points.append([y2 + w / 2 * np.cos(angle), x2 + w / 2 * np.sin(angle)])
    points.append([y1 + w / 2 * np.cos(angle), x1 + w / 2 * np.sin(angle)])
    points = np.array(points)

    return ptsOnRect(points)


def calcAngle2(angle):
    """
    Computes the opposite angle from the given angle
    (the inverse angle)
    :param angle: radians
    :return: radians
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi


def drawGrasps(img, grasps, mode='line'):
    """
    draw grasp
    img:    rgb
    grasps: list()	elements: [row, col, angle, width]
    mode:   line or region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
        else:
            img[row, col] = [color_b, color_g, color_r]

    return img


def drawRect(img, rect):
    """
    draw the rectangle
    rect: [x1, y1, x2, y2]
    """
    print(rect)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


def depth2Gray(im_depth):
    """
    depth to gray (1 channel)
    (h, w, 3)
    """
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('error ...')
        raise EOFError

    img = 255 * (im_depth - x_min) / (x_max - x_min)
    return img.astype(np.uint8)


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


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


def collision_detection(pt, dep, angle, depth_map, finger_l1, finger_l2):
    """
    collision detection
    pt: (row, col)
    angle: grasp angle (radians)
    depth_map: depth image
    finger_l1 l2: in pixel

    return:
        True: no collision
        False: have collision
    """
    row, col = pt

    row1 = int(row - finger_l2 * math.sin(angle))
    col1 = int(col + finger_l2 * math.cos(angle))

    rows, cols = ptsOnRotateRect([row, col], [row1, col1], finger_l1)

    if np.min(depth_map[rows, cols]) > dep:
        return True
    return False


def getGraspDepth(camera_depth, grasp_row, grasp_col, grasp_angle, grasp_width, finger_l1, finger_l2):
    """
    Calculate the maximum collision-free grasping depth
    (the depth of descent relative to the surface of the object) based on the depth image,
    grasping angle, and grasping width
    camera_depth:
    grasp_angle：
    grasp_width：
    finger_l1 l2:

    return: depth
    """
    k = math.tan(grasp_angle)

    grasp_width /= 2
    if k == 0:
        dx = grasp_width
        dy = 0
    else:
        dx = k / abs(k) * grasp_width / pow(k ** 2 + 1, 0.5)
        dy = k * dx

    pt1 = (int(grasp_row - dy), int(grasp_col + dx))
    pt2 = (int(grasp_row + dy), int(grasp_col - dx))

    rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])
    min_depth = np.min(camera_depth[rr, cc])

    grasp_depth = min_depth + 0.003
    while grasp_depth < min_depth + 0.05:
        if not collision_detection(pt1, grasp_depth, grasp_angle, camera_depth, finger_l1, finger_l2):
            return grasp_depth - 0.003
        if not collision_detection(pt2, grasp_depth, grasp_angle + math.pi, camera_depth, finger_l1, finger_l2):
            return grasp_depth - 0.003
        grasp_depth += 0.003

    return grasp_depth


class GGCNNNet:
    def __init__(self, model, device):
        self.device = device
        # load model
        print('>> loading GGCNN2')
        self.net = GGCNN2()
        self.net.load_state_dict(torch.load(model, map_location=self.device),
                                 strict=True)
        # self.net = self.net.to(device)
        print('>> load done')

    def predict(self, img, input_size=300):
        """
        prediction
        :param img: depth np.array (h, w)
        :return:
            pred_grasps: list([row, col, angle, width])  width (pixel)
        """
        # crop
        input, self.crop_x1, self.crop_y1 = input_img(img, input_size)

        self.pos_out, self.cos_out, self.sin_out, self.wid_out = get_pred(self.net, input.to(self.device))
        pos_pred, ang_pred, wid_pred = post_process_output(self.pos_out, self.cos_out, self.sin_out, self.wid_out)

        # most graspable position
        loc = np.argmax(pos_pred)
        row = loc // pos_pred.shape[0]
        col = loc % pos_pred.shape[0]
        angle = (ang_pred[row, col] + 2 * math.pi) % math.pi
        width = wid_pred[row, col]
        row += self.crop_y1
        col += self.crop_x1

        return row, col, angle, width
