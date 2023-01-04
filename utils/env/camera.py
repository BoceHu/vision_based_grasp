#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: camera.py

@Date: 2023/1/1
"""
import math
import numpy as np

# image shape
HEIGHT = 480
WIDTH = 640


def radians_TO_angle(radians):
    """
    radian to angle
    """
    return 180 * radians / math.pi


def angle_TO_radians(angle):
    """
    angle to radian
    """
    return math.pi * angle / 180


def eulerAnglesToRotationMatrix(theta):
    """
    rotation matrix
    refer: https://en.wikipedia.org/wiki/Rotation_matrix
    theta: [r, p, y]
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def getTransfMat(offset, rotate):
    """
    rotation + translation = transformation
    offset: (x, y, z)
    rotate: rotation matrix
    """
    mat = np.array([
        [rotate[0, 0], rotate[0, 1], rotate[0, 2], offset[0]],
        [rotate[1, 0], rotate[1, 1], rotate[1, 2], offset[1]],
        [rotate[2, 0], rotate[2, 1], rotate[2, 2], offset[2]],
        [0, 0, 0, 1.]
    ])
    return mat


class Camera:
    def __init__(self):
        """
        Initialize camera parameters and calculate camera internal parameters
        """
        self.fov = 60  # field of view
        self.length = 0.7  # camera height (meter)
        self.H = self.length * math.tan(angle_TO_radians(
            self.fov / 2))  # The actual distance from the midpoint of the first row of the image to the center of the image m
        self.W = WIDTH * self.H / HEIGHT  # The actual distance from the midpoint of the first column of the image to the center of the image m
        # calculate focal length: f = (Image height) * depth / actual height
        # https://www.cnblogs.com/zipeilu/p/6658177.html
        self.A = (HEIGHT / 2) * self.length / self.H
        # Intrinsic Matrix
        self.InMatrix = np.array([[self.A, 0, WIDTH / 2 - 0.5], [0, self.A, HEIGHT / 2 - 0.5], [0, 0, 1]],
                                 dtype=np.float)
        # World coordinate system -> Camera coordinate system 4*4
        # Angle: (pi, 0, 0)    translation(0, 0, 0.7)
        rotMat = eulerAnglesToRotationMatrix([math.pi, 0, 0])
        self.transMat = getTransfMat([0, 0, 0.7], rotMat)

    def camera_height(self):
        return self.length

    def img2camera(self, pt, dep):
        """
        pixel -> camera
        pt: [x, y]
        dep:
        [(InMatrix)^-1 matmul pixel coordinate] * Z = camera coordinate
        return: [x, y, z]
        """
        pt_in_img = np.array([[pt[0]], [pt[1]], [1]], dtype=np.float)
        ret = np.matmul(np.linalg.inv(self.InMatrix), pt_in_img) * dep
        return list(ret.reshape((3,)))

    def camera2img(self, coord):
        """
        camera -> pixel
        coord: [x, y, z]

        return: [row, col]
        """
        z = coord[2]
        coord = np.array(coord).reshape((3, 1))
        rc = (np.matmul(self.InMatrix, coord) / z).reshape((3,))

        return list(rc)[:-1]

    def length_TO_pixels(self, l, dep):
        """
        actual length: l -> pixel length
        l: meter
        dep: meter (the distance between the line and the camera)
        """
        return l * self.A / dep

    def pixels_TO_length(self, p, dep):
        """
        """
        return p * dep / self.A

    def camera2world(self, coord):
        """
        camera coordinate -> world coordinate
        coord: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(self.transMat, coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2camera(self, coord):
        """
        world coordinate -> camera coordinate
        corrd: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(np.linalg.inv(self.transMat), coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2img(self, coord):
        """
        world coordinate -> pixel coordinate
        corrd: [x, y, z]

        return: [row, col]
        """
        # world -> camera
        coord = self.world2camera(coord)
        # camera -> pixel
        pt = self.camera2img(coord)  # [y, x]
        return [int(pt[1]), int(pt[0])]

    def img2world(self, pt, dep):
        """
        pixel coordinate -> world coordinate
        pt: [x, y]
        dep: m
        return: [x, y, z]
        """
        coordInCamera = self.img2camera(pt, dep)
        return self.camera2world(coordInCamera)


if __name__ == '__main__':
    camera = Camera()
    print(camera.InMatrix)
