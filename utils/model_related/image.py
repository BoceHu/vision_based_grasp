#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: image.py

@Date: 2022/12/27
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize
import imageio


class Image:
    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname):
        return cls(imageio.v3.imread(fname))

    def copy(self):
        return self.__class__(self.img.copy())

    def crop(self, top_left, bottom_right, resize=None):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        :return:
        """
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        i = self.copy()
        i.crop(*args, **kwargs)
        return i

    def normalize(self):
        """
        Normalize the image by
        (1) converting to float [0,1]
        (2) zero-centering
        """
        self.img = self.img.astype(float) / 255.
        self.img -= self.img.mean()

    def resize(self, shape):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)

    def resized(self, *args, **kwargs):
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img, angle / np.pi * 180, center=center, mode='symmetric', preserve_range=True).astype(
            self.img.dtype)

    def rotated(self, *args, **kwargs):
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        origin_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc:self.img.shape[1] - sc]
        self.img = resize(self.img, origin_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs):
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img):
        super().__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename, shape, default_filler=0, index=None):
        """
            Create a depth image from an unstructured PCD file.
            If index isn't specified, use euclidean distance, otherwise choose x/y/z=0/1/2
        """
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.strip().split(' ')

                if len(ls) != 5:
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])
                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                else:
                    img[r, c] = float(ls[index])
        return cls(img / 1000.)

    @classmethod
    def from_tiff(cls, fname):
        return cls(imageio.v3.imread(fname))

    def inpaint(self, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        self.img = cv2.copyMakeBorder(self.img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (self.img == missing_value).astype('uint8')

        scale = np.abs(self.img).max()
        self.img = self.img.astype(np.float32) / scale
        self.img = cv2.inpaint(self.img, mask, 1, cv2.INPAINT_NS)

        self.img = self.img[1:-1, 1:-1]
        self.img = self.img * scale

    def normalize(self):
        """
        Normalize by subtracting the mean and clippint [-1, 1]
        """
        self.img = np.clip((self.img - self.img.mean()), -1, 1)
