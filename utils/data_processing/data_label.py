#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: data_label.py

@Date: 2022/12/26
"""

import os
import glob
import scipy.io as scio
import numpy as np

label_path = '../../cornell'
save_path = '../../cornell_label'


def run():
    label_files = glob.glob(label_path + '/*Label.txt')
    max_w = 0
    for label_file in label_files:
        print('processing: ', label_file)
        label_mat = np.zeros((3, 480, 640), dtype=np.float)

        with open(label_file) as f:
            labels = f.readlines()
            for label in labels:
                label = label.strip().split(' ')
                row = int(float(label[0]))
                col = int(float(label[1]))
                label_mat[0, row, col] = 1.0

                if len(label) == 3:
                    label_mat[1, row, col] = 0.
                else:
                    label_mat[1, row, col] = float(label[2])

                label_mat[2, row, col] = float(label[-1]) / 200.
                if float(label[-1]) > max_w:
                    max_w = float(label[-1])

            save_name = os.path.join(save_path, os.path.basename(label_file).replace('Label.txt', 'grasp.mat'))
            scio.savemat(save_name, {'A': label_mat})

    print("max_w = ", max_w)


if __name__ == '__main__':
    run()
