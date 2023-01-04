#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: sim_env.py

@Date: 2022/12/30
"""
import pybullet as p
import numpy as np
import skimage.transform as skt
import cv2
import pybullet_data
import os
import math
import shutil
import random
import scipy.stats as ss

IMAGEWIDTH = 640
IMAGEHEIGHT = 480
nearPlane = 0.01
farPlane = 10

fov = 60
aspect = IMAGEWIDTH / IMAGEHEIGHT


def imresize(image, size, interp='nearest'):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic' interpolation are no longer supported.")

    assert interp in skt_interp_map, ('Interpolation "{}" not support'.format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        # proportion
        np_shape = np.asarray(image.shape).astype(float)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        # percentage
        np_shape = np.asarray(image.shape).astype(float)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")


def inpaint(img, missing_value):
    """
    Inpaint missing values in depth image.
    :param img:
    :param missing_value: Value to fill in the depth image.
    :return:
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype('float32') / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


class SimEnv():

    def __init__(self, bullet_client: p, path, gripperId=None):
        """

        :param bullet_client:
        :param path: urdf file path
        :param gripperId:
        """
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, solverResidualThreshold=0, enableFileCaching=0)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                          cameraTargetPosition=[0, 0, 0])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeID = self.p.loadURDF('plane.urdf', [0, 0, 0])
        self.p.setGravity(0, 0, -10)
        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.gripperId = gripperId

        self.movecamera(0, 0)
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        list_file = os.path.join(path, 'list.txt')
        if not os.path.exists(list_file):
            raise shutil.Error
        self.urdfs_list = []
        with open(list_file, 'r') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                self.urdfs_list.append(os.path.join(path, line[:-1] + '.urdf'))

        self.num_urdf = 0
        self.urdfs_id = []
        self.objs_id = []
        self.EulerRPList = [[0, 0], [math.pi / 2, 0], [-1 * math.pi / 2, 0], [math.pi, 0], [0, math.pi / 2],
                            [0, -1 * math.pi / 2]]

    def _urdf_nums(self):
        """
        :return: total number of objs
        """
        return len(self.urdfs_list)

    def movecamera(self, x, y, z=0.7):
        """
        move camera to the specific position
        :param x: the x coordinate in the world coordinate
        :param y: the y coordinate in the world coordinate
        :param z: the z coordinate in the world coordinate
        :return:
        """
        self.viewMatrix = self.p.computeViewMatrix([x, y, z], [x, y, 0], [0, 1, 0])

    def loadObjInURDF(self, urdf_file, idx, render_n=0):
        """
        load single obj in URDF format
        :param urdf_file:
        :param idx: object id if ==-1, use file name
        :param render_n:
        :return:
        """
        if idx >= 0:
            self.urdfs_filename = [self.urdfs_list[idx]]
            self.objs_id = [idx]
        else:
            self.urdfs_filename = [urdf_file]
            self.objs_id = [-1]

        self.num_urdf = 1

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        baseEuler = [random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)]
        baseOrientation = self.p.getQuaternionFromEuler(baseEuler)

        pos = 0.1
        basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1, 0.4)]

        # load obj
        urdf_id = self.p.loadURDF(self.urdfs_filename[0], basePosition, baseOrientation)

        inf = self.p.getVisualShapeData(urdf_id)[0]
        self.urdfs_id.append(urdf_id)
        self.urdfs_xyz.append(inf[5])  # position
        self.urdfs_scale.append(inf[3][0])  # size (scale)

    def loadObjsInURDF(self, idx, num):
        """

        :param idx: start position; if negative, random load num objs
        :param num: the num of objs
        :return:
        """
        assert idx < len(self.urdfs_list)
        self.num_urdf = num

        if idx < 0:
            objs_id = list(range(0, len(self.urdfs_list)))
            self.urdfs_filename, self.objs_id = random.sample(list(zip(self.urdfs_list, objs_id)), self.num_urdf)

        elif (idx + self.num_urdf - 1) > (len(self.urdfs_list) - 1):
            self.urdfs_filename = self.urdfs_list[idx:]
            self.urdfs_filename += self.urdfs_list[:self.num_urdf - len(self.urdfs_list) + idx]
            self.objs_id = list(range(idx, len(self.urdfs_list)))
            self.objs_id += list(range(self.num_urdf - len(self.urdfs_list) + idx))
        else:
            self.urdfs_filename = self.urdfs_list[idx:idx + self.num_urdf]
            self.objs_id = list(range(idx, idx + self.num_urdf))

        print('self.objs_id = \n', self.objs_id)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []

        for i in range(self.num_urdf):
            pos = 0.1
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1, 0.4)]

            baseEuler = [random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)

            urdf_id = self.p.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)
            if self.gripperId is not None:
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 0, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 1, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 2, 1)

            inf = self.p.getVisualShapeData(urdf_id)[0]

            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5])
            self.urdfs_scale.append(inf[3][0])

            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break

    def evalGrasp(self, z_thresh):
        """
        validate the grasp
        :param z_thresh:
        :return:
        """
        for i in range(self.num_urdf):
            offset, _ = self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[2] >= z_thresh:
                return True
        print('!!!!!!!!!!!!!!! Failure !!!!!!!!!!!!!!!')
        return False

    def evalGraspAndRemove(self, x_thresh):
        """
        validate the grasp and delete the grasped object
        :param z_thresh:
        :return:
        """
        for i in range(self.num_urdf):
            offset, _ = self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[0] >= x_thresh:
                self.removeObjInURDF(i)
                return True
        print('!!!!!!!!!!!!!!! Failure !!!!!!!!!!!!!!!')
        return False

    def removeObjInURDF(self, i):
        """
        delete the obj
        :param i:
        :return:
        """
        self.num_urdf -= 1
        self.p.removeBody(self.urdfs_id[i])
        self.urdfs_id.pop(i)
        self.urdfs_xyz.pop(i)
        self.urdfs_scale.pop(i)
        self.urdfs_filename.pop(i)
        self.objs_id.pop(i)

    def removeObjsInURDF(self):
        """
        delete all objs
        """
        for i in range(self.num_urdf):
            self.p.removeBody(self.urdfs_id[i])
        self.num_urdf = 0
        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        self.urdfs_filename = []
        self.objs_id = []

    def resetObjsPoseRandom(self):
        """
        random the position of all objs
        :return:
        """
        for i in range(self.num_urdf):
            pos = 0.1
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.3, 0.6)]
            baseEuler = [random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            self.p.resetBasePositionAndOrientation(self.urdfs_id[i], basePosition, baseOrientation)

            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break

    def renderCameraDepthImage(self):
        """
        distance_to_plane = near * far / (far - depth * (far - near))
        """
        # render image
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]  # width of the image, in pixels
        h = img_camera[1]  # height of the image, in pixels
        dep = img_camera[3]  # depth data

        # get depth image
        depth = np.reshape(dep, (h, w))  # [480, 640
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # unit: meter
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        return im_depthCamera

    def renderCameraMask(self):
        """
        mask
        """
        # render image
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]  # width of the image, in pixels
        h = img_camera[1]  # height of the image, in pixels
        # rgba = img_camera[2]    # color data RGB
        # dep = img_camera[3]    # depth data
        mask = img_camera[4]  # mask data

        im_mask = np.reshape(mask, (h, w)).astype(np.uint8)
        im_mask[im_mask > 0] = 255
        return im_mask

    def gaussian_noise(self, im_depth):
        """
        add gaussian noise on images; refer to the dex-net
        im_depth: depth image (meter)
        """
        gaussian_process_sigma = 0.002
        gaussian_process_scaling_factor = 8.0

        im_height, im_width = im_depth.shape

        # 2
        gp_rescale_factor = gaussian_process_scaling_factor
        gp_sample_height = int(im_height / gp_rescale_factor)  # im_height / 8.0
        gp_sample_width = int(im_width / gp_rescale_factor)  # im_width / 8.0
        gp_num_pix = gp_sample_height * gp_sample_width  # im_height * im_width / 64.0
        gp_sigma = gaussian_process_sigma
        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height,
                                                                        gp_sample_width)

        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")
        im_depth += gp_noise

        return im_depth

    def add_noise(self, img):
        """
        add gaussian noise
        """
        img = self.gaussian_noise(img)
        return img
