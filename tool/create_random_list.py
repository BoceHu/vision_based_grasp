#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: create_random_list.py

@Date: 2022/12/30
"""
import os
import glob
import random

path = '../objects_models/objs'
files = glob.glob(os.path.join(path, '*', '*.urdf'))
random.shuffle(files)

txt = open(path + '/list.txt', 'w+')
for file in files:
    fname = os.path.basename(file)
    pre_fname = os.path.basename(os.path.dirname(file))
    txt.write(pre_fname + '/' + fname[:-5] + '\n')

txt.close()
print('Done')
