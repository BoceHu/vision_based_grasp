#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: create_collision_obj.py

@Date: 2022/12/30
"""
import os
import pybullet as p

path = '../objects_models/objs/meshes'

p.connect(p.DIRECT)
files = os.listdir(path)
for file in files:
    print("processing ...",file)
    name_in = os.path.join(path, file)
    name_out = os.path.join(path, file.replace('.obj', '_col.obj'))
    name_log = "log.txt"
    p.vhacd(name_in, name_out, name_log)