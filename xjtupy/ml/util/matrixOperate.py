#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/9 21:30
# @Author: Vincent
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
    def mean_column(matrix):
        """
        按列求矩阵的均值
        """
        return np.mean(matrix, axis=0)

    @staticmethod
    def mean_row(matrix):
        """
        按行求矩阵的均值
        """
        return np.mean(matrix, axis=1)

    @staticmethod
    def sum_row(matrix):
        """
        矩阵行相加
        """
        return matrix.sum(axis=0)

    @staticmethod
    def sum_column(matrix):
        """
        矩阵列相加
        """
        return matrix.sum(axis=1)
