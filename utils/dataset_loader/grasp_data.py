#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: grasp_data.py

@Date: 2022/12/27
"""
import numpy as np
import torch
import torch.utils.data
import os
import glob

from utils.model_related.img import DepthImage
from utils.dataset_loader.grasp import GraspMat


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, output_size=360,
                 include_depth=True,
                 include_rgb=False, argument=False):
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.argument = argument

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        graspf = glob.glob(os.path.join(file_path, '*grasp.mat'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l * ds_rotate):] + graspf[:int(l * ds_rotate)]

        depthf = [f.replace('grasp.mat', 'd.tiff') for f in graspf]
        # rgbf = [f.replace('grasp.mat', 'd.tiff') for f in graspf]

        self.grasp_files = graspf[int(l * start):int(l * end)]
        self.depth_files = depthf[int(l * start):int(l * end)]
        # self.rgb_files = rgbf[int(l * start):int(l * end)]

    @staticmethod
    def numpy_to_torch(s):
        """
        numpy to tensor
        :param s:
        :return:
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype('float32'))
        else:
            return torch.from_numpy(s.astype('float32'))

    def __getitem__(self, idx):
        image = DepthImage(self.depth_files[idx])
        label = GraspMat(self.grasp_files[idx])

        if self.argument:
            # resize
            scale = np.random.uniform(0.9, 1.1)
            image.rescale(scale)
            label.rescale(scale)
            # rotate
            rota = 30
            rota = np.random.uniform(-1 * rota, rota)
            image.rotate(rota)
            label.rotate(rota)
            # crop
            dist = 30
            crop_bbox = image.crop(self.output_size, dist)
            label.crop(crop_bbox)
            # flip
            flip = True if np.random.rand() < 0.5 else False
            if flip:
                image.flip()
                label.flip()
        else:
            # crop
            crop_bbox = image.crop(self.output_size)
            label.crop(crop_bbox)

        image.normalize()
        label.encode()

        img = self.numpy_to_torch(image.img)
        grasp_point = self.numpy_to_torch(label.grasp_point)
        grasp_cos = self.numpy_to_torch(label.grasp_cos)
        grasp_sin = self.numpy_to_torch(label.grasp_sin)
        grasp_width = self.numpy_to_torch(label.grasp_width)

        return img, (grasp_point, grasp_cos, grasp_sin, grasp_width)

    def __len__(self):
        return len(self.grasp_files)
