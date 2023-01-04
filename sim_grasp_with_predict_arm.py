#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: sim_grasp_with_predict_arm.py

@Date: 2022/12/30
"""
import pybullet as p
import time
import cv2
from utils.env.sim_env import SimEnv
import utils.tool as tool
import utils.env.panda_sim_arm as PandaSim
from utils.env.camera import Camera
from models.predict_ggcnn2 import GGCNNNet, drawGrasps, getGraspDepth

FINGER_L1 = 0.015
FINGER_L2 = 0.005
cv2.namedWindow('im_grasp', cv2.WINDOW_NORMAL)

cv2.resizeWindow("im_grasp", 500, 500)
cv2.moveWindow("im_grasp", 1300, 200)


def run(database_path, start_idx, objs_num):
    cid = p.connect(p.GUI)
    panda = PandaSim.PandaSimAuto(p, [0, -0.6, 0])
    env = SimEnv(p, database_path, panda.pandaId)
    camera = Camera()
    ggcnn = GGCNNNet('./ckpt/g2.pth', device="cpu")
    time.sleep(3)
    success_grasp = 0
    sum_grasp = 0
    tt = 5
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0

    while True:
        # wait for the objects to be stable
        for _ in range(240 * 5):
            p.stepSimulation()
        # render depth
        camera_depth = env.renderCameraDepthImage()
        camera_depth = env.add_noise(camera_depth)

        # prediction
        row, col, grasp_angle, grasp_width_pixels = ggcnn.predict(camera_depth, input_size=300)
        grasp_width = camera.pixels_TO_length(grasp_width_pixels, camera_depth[row, col])

        grasp_x, grasp_y, grasp_z = camera.img2world([col, row], camera_depth[row, col])  # [x, y, z]
        finger_l1_pixels = camera.length_TO_pixels(FINGER_L1, camera_depth[row, col])
        finger_l2_pixels = camera.length_TO_pixels(FINGER_L2, camera_depth[row, col])
        grasp_depth = getGraspDepth(camera_depth, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels,
                                    finger_l2_pixels)
        grasp_z = max(0.7 - grasp_depth, 0)

        print('*' * 100)
        print('grasp pose:')
        print('grasp_x = ', grasp_x)
        print('grasp_y = ', grasp_y)
        print('grasp_z = ', grasp_z)
        print('grasp_depth = ', grasp_depth)
        print('grasp_angle = ', grasp_angle)
        print('grasp_width = ', grasp_width)
        print('*' * 100)

        im_rgb = tool.depth2Gray3(camera_depth)
        im_grasp = drawGrasps(im_rgb, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')
        cv2.imshow('im_grasp', im_grasp)
        cv2.waitKey(30)

        # grasp
        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t % tt == 0:
                time.sleep(1. / 240.)

            if panda.step([grasp_x, grasp_y, grasp_z], grasp_angle, grasp_width / 2):
                t = 0
                break

        sum_grasp += 1

        for ii in range(240):
            p.stepSimulation()
            if ii % 5 == 0:
                time.sleep(1. / 240.)
            panda.setArmPos([0.5 * ii / 240, -0.6 * ii / 240, 0.2])
        if env.evalGraspAndRemove(x_thresh=0.4):
            success_grasp += 1
            continue_fail = 0
            if env.num_urdf == 0:
                p.disconnect()
                return success_grasp, sum_grasp
        else:
            continue_fail += 1
            if continue_fail == 5:
                p.disconnect()
                return success_grasp, sum_grasp


if __name__ == "__main__":
    start_idx = 0
    objs_num = 5
    database_path = 'objects_models/objs'
    success_grasp, all_grasp = run(database_path, start_idx, objs_num)
    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(success_grasp, all_grasp, success_grasp / all_grasp))
    print('\n>>>>>>>>>>>>>>>>>>>> Percent Cleared: {}/{}={}'.format(success_grasp, objs_num, success_grasp / objs_num))
