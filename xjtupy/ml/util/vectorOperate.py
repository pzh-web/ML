#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/9 21:51
# @Author: Vincent
# @File  : vectorOperate.py
import numpy as np

"""
向量运算
"""


class VectorOperate(object):

    @staticmethod
    def distance(v, w):
        """两点的距离"""
        return np.math.sqrt(VectorOperate.squared_distance(v, w))

    @staticmethod
    def squared_distance(v, w):
        """
        两点之间的欧式距离
        """
        return np.dot(np.array(v) - np.array(w), np.array(v) - np.array(w))
