#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/27 19:04
# @Author: peng yang
# @File  : normalEquation.py

import numpy as np

'''
正规方程法
'''


class NormalEquation(object):
    def __init__(self, normal=False, normal_param=0.8):
        '''
        :param normal: 是否使用正则化
        :param normal_param: 正则化参数
        '''
        self.normal = normal
        self.normal_param = normal_param

    def normal_equation_method(self, x, y):
        return self.use_normal(x, y) if self.normal else self.not_normal(x, y)

    def not_normal(self, x, y):
        '''
        不适用正则化项
        求 w=(x.T*x).I*x.T*y
        '''
        b = np.ones(len(x))
        x = np.insert(x, len(x[0]), values=b, axis=1)
        x = np.mat(x)
        y = np.mat(y).T
        w = (x.T * x).I * x.T * y
        return x * w

    def use_normal(self, x, y):
        '''
        使用正则化项
        求 w=(x.T*x).I*x.T*y
        '''
        b = np.ones(len(x))
        x = np.insert(x, len(x[0]), values=b, axis=1)
        x = np.mat(x)
        y = np.mat(y).T

        # 生成正则化项
        z = np.mat(np.eye(x.shape[1]))
        z[len(x[0]) - 1][len(x[0]) - 1] = 0

        w = (x.T * x + self.normal_param * z).I * x.T * y
        return x * w
