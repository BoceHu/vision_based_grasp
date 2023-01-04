#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: show_label.py

@Date: 2022/12/27
"""
import cv2
import glob
import os
import math

path = '../cornell'  # dataset
label_files = glob.glob(os.path.join(path, 'pcd*Label.txt'))

for labelfile in label_files:
    pngfile = labelfile.replace('Label.txt', 'r.png')
    if not os.path.exists(pngfile):
        continue

    im = cv2.imread(pngfile)

    # top = 113
    # left = 113

    # cv2.line(im, (top, left), (top + 300, left), (0, 255, 255), 2)
    # cv2.line(im, (top + 300, left), (top + 300, left + 300), (0, 255, 255), 2)
    # cv2.line(im, (top + 300, left + 300), (top, left + 300), (0, 255, 255), 2)
    # cv2.line(im, (top, left + 300), (top, left), (0, 255, 255), 2)

    f = open(labelfile)
    points = f.readlines()
    f.close()

    n = 0
    for point in points:
        point_data = point.split(' ')
        y = int(float(point_data[0]))
        x = int(float(point_data[1]))
        w = float(point_data[-2]) / 2
        n += 1
        if n % 10 == 0:
            if len(point_data) == 4:
                # circle
                cv2.circle(im, (x, y), int(w), (0, 255, 0), 1)

            elif len(point_data) == 5:
                # uni-direction grasp
                angle = float(point_data[2])
                k = math.tan(angle)

                if k == 0:
                    dx = w
                    dy = 0
                else:
                    dx = k / abs(k) * w / pow(k ** 2 + 1, 0.5)
                    dy = k * dx

                if angle < math.pi:
                    cv2.line(im, (x, y), (int(x + dx), int(y - dy)), (0, 255, 0), 1)
                else:
                    cv2.line(im, (x, y), (int(x - dx), int(y + dy)), (0, 255, 0), 1)

            elif len(point_data) == 6:
                # bi-direction grasp
                angle1 = float(point_data[2])
                angle2 = float(point_data[3])
                k = math.tan(angle1)

                if k == 0:
                    dx = w
                    dy = 0
                else:
                    dx = k / abs(k) * w / pow(k ** 2 + 1, 0.5)
                    dy = k * dx
                if angle1 < math.pi:
                    cv2.line(im, (x, y), (int(x + dx), int(y - dy)), (0, 255, 0), 1)
                else:
                    cv2.line(im, (x, y), (int(x - dx), int(y + dy)), (0, 255, 0), 1)

                if angle2 < math.pi:
                    cv2.line(im, (x, y), (int(x + dx), int(y - dy)), (0, 255, 0), 1)
                else:
                    cv2.line(im, (x, y), (int(x - dx), int(y + dy)), (0, 255, 0), 1)

        cv2.circle(im, (x, y), 1, (255, 0, 0), -1)
        # im[y, x] = [0, 255, 0]

    cv2.imshow('im', im)
    key = cv2.waitKeyEx()
    if key == 27:
        cv2.destroyAllWindows()
        break
