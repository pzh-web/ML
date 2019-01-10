#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/9 21:30
# @Author: peng yang
# @File  : matrixOperate.py
import numpy as np

"""
矩阵运算工具类
"""


class MatrixOperate(object):

    @staticmethod
    def svd(matrix):
        """
        矩阵奇异值分解
        :return: U,D,V.T
        """
        return np.linalg.svd(matrix)

    @staticmethod
    def mean_row(matrix):
        """
        按行求矩阵的均值
        """
        return np.mean(matrix, axis=0)

    @staticmethod
    def mean_column(matrix):
        """
        按行求矩阵的均值
        """
        return np.mean(matrix, axis=1)
