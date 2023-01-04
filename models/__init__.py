#!/usr/bin/env python
# encoding: utf-8
"""
@author: Boce Hu

@Project Name: __init__.py.py

@Date: 2022/12/27
"""


def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
