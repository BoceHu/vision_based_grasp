#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: data_structure.py

@Date: 2022/12/27
"""
import glob
import os
import numpy as np
from imageio import imsave
from utils.model_related.image import DepthImage

if __name__ == '__main__':
    path = "../../cornell_data_raw"
    save_path = "../../cornell_label"
    pcds = glob.glob(os.path.join(path, "pcd*[0-9].txt"))
    pcds.sort()

    for pcd in pcds:
        di = DepthImage.from_pcd(pcd, (480, 640))
        di.inpaint()

        of_name = os.path.basename(pcd).replace('.txt', 'd.tiff')
        save_name = os.path.join(save_path, of_name)
        imsave(save_name, di.img.astype(np.float32))
        print(save_name)
