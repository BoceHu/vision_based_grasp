#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path

GRASP_MAX_W = 0.08


def calcPlane(pt1, pt2, pt3):
    """
    calculate: ax+by+cz+d=0
    pts: [[x, y, z], [x, y, z], [x, y, z]]
    vect(pt1,pt2) x vect(pt1,pt3) = (a,b,c)
    By out-product, the normal can be found which ax+by+cz+d=0. (x,y,z) is a vector in the plane
    :param pt1:
    :param pt2:
    :param pt3:
    :return: A B C   z=Ax+By+C
    """
    a = (pt2[1] - pt1[1]) * (pt3[2] - pt1[2]) - (pt2[2] - pt1[2]) * (pt3[1] - pt1[1])
    b = (pt2[2] - pt1[2]) * (pt3[0] - pt1[0]) - (pt2[0] - pt1[0]) * (pt3[2] - pt1[2])
    c = (pt2[0] - pt1[0]) * (pt3[1] - pt1[1]) - (pt2[1] - pt1[1]) * (pt3[0] - pt1[0])
    d = 0 - (a * pt1[0] + b * pt1[1] + c * pt1[2])

    return a, b, c, d


def ptsInTriangle(pt1, pt2, pt3):
    """
    Get the coordinate points in the triangle formed by pt1 pt2 pt3
    pt1: float [x, y]
    """
    p = path.Path([pt1, pt2, pt3])

    min_x = int(min(pt1[0], pt2[0], pt3[0]))
    max_x = int(max(pt1[0], pt2[0], pt3[0]))
    min_y = int(min(pt1[1], pt2[1], pt3[1]))
    max_y = int(max(pt1[1], pt2[1], pt3[1]))

    pts = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if p.contains_points([(x, y)])[0]:
                pts.append([x, y])

    return pts


class Mesh():
    """
    read obj files and transform the coordinate to depth image
    """

    def __init__(self, filename, scale=-1):
        """

        :param filename: obj filename
        :param scale: int
            -1 : Automatically set the scale so that the middle side of the bounding rectangle does not exceed 80% of the gripper width (0.07). The maximum scale is 0.001
        """
        assert scale == -1 or scale > 0
        # print(filename)
        if scale > 0:
            self._scale = scale
        else:
            self._scale = 1
            with open(filename) as file:
                self.points = []
                self.faces = []
                while 1:
                    line = file.readline()
                    if not line:
                        break
                    strs = line.split(" ")  # f 1/1/1 5/2/1 7/3/1 3/4/1
                    if strs[0] == "v":
                        self.points.append(
                            (float(strs[1]) * self._scale, float(strs[2]) * self._scale, float(strs[3]) * self._scale))
                    if strs[0] == "f":
                        if strs[1].count('//'):
                            idx1, idx2, idx3 = strs[1].index('//'), strs[2].index('//'), strs[3].index('//')
                            self.faces.append((int(strs[1][:idx1]), int(strs[2][:idx2]), int(strs[3][:idx3])))
                        elif strs[1].count('/') == 0:
                            self.faces.append((int(strs[1]), int(strs[2]), int(strs[3])))

            self.points = np.array(self.points)
            self.faces = np.array(self.faces, dtype=np.int64)
            if scale == -1:
                self._scale = self.get_scale()
                self.points = self.points * self._scale

    def min_z(self):
        """
        return minimum z coordinate
        """
        return np.min(self.points[:, 2])

    def get_scale(self):
        """
        set scale adaptively
        """
        d_x = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        d_y = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        d_z = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        ds = [d_x, d_y, d_z]
        ds.sort()
        scale = (GRASP_MAX_W - 0.01) * 0.8 / ds[1]
        if scale > 0.001:
            scale = 0.001

        return scale

    def scale(self):
        return self._scale

    def calcCenterPt(self):
        """
        calculate the center point in the mesh
        return: [x, y, z]
        """
        return np.mean(self.points, axis=0)

    def transform(self, mat):
        """
        Adjust vertex coordinates according to rotation matrix
        """
        points = self.points.T  # Transpose
        ones = np.ones((1, points.shape[1]))
        points = np.vstack((points, ones))

        new_points = np.matmul(mat, points)[:-1, :]
        self.points = new_points.T  # Transpose  (n, 3)

    def renderTableImg(self, mask_id, size=(0.8, 0.8), unit=0.001):
        """
        Render the depth map and obj mask relative to the horizontal plane, with 0.5mm between each point
        size: (h, w) unit: m

        Algorithm flow:
             Method 1: Calculate the plane equation where each triangular grid is located, calculate the point (x, y) located in the triangular area, bring it into the plane equation, and get the depth z
             Method 2: Calculate the discrete depth in space according to the mesh vertices.
                       For a point within the object area (the outer envelope of the aforementioned discrete point) but with a depth of 0,
                       calculate the depth based on the interpolation of the three nearest points
        """

        # Initialize the depth image
        depth_map = np.zeros((int(size[0] / unit), int(size[1] / unit)), dtype=np.float)
        for face in self.faces:
            pt1 = self.points[face[0] - 1]  # xyz m
            pt2 = self.points[face[1] - 1]
            pt3 = self.points[face[2] - 1]
            # Calculate the plane equation of the triangular mesh xyz -> plane
            plane_a, plane_b, plane_c, plane_d = calcPlane(pt1, pt2, pt3)  # ABC  Ax+By+C=z
            if plane_c == 0:
                continue
            plane = np.array([-1 * plane_a / plane_c, -1 * plane_b / plane_c, -1 * plane_d / plane_c])

            # Convert triangle coordinates to pixel coordinates
            pt1_pixel = [pt1[0] / unit, pt1[1] / unit]  # xy pixel float
            pt2_pixel = [pt2[0] / unit, pt2[1] / unit]
            pt3_pixel = [pt3[0] / unit, pt3[1] / unit]
            # Get the pixel coordinates inside the triangle (x,y)
            pts_pixel = ptsInTriangle(pt1_pixel, pt2_pixel, pt3_pixel)

            if len(pts_pixel) == 0:
                continue

            # Convert pixel coordinates to actual coordinates m
            pts = np.array(pts_pixel, dtype=np.float) * unit  # (n, 2) m
            # Bring in the plane equation to calculate the depth m
            ones = np.ones((pts.shape[0], 1))
            pts = np.hstack((pts, ones))
            depth = np.matmul(pts, plane.reshape((3, 1))).reshape((-1,))
            # Convert the pixel coordinate pts_pixel to the image coordinate system
            pts_pixel = np.array(pts_pixel, dtype=np.int64)
            xs, ys = pts_pixel[:, 0], pts_pixel[:, 1]
            rows = ys * -1 + int(round(depth_map.shape[0] / 2))
            cols = xs + int(round(depth_map.shape[0] / 2))
            # generate depth image pts_pixel depth
            depth_map[rows, cols] = np.maximum(depth_map[rows, cols], depth)

        return depth_map
