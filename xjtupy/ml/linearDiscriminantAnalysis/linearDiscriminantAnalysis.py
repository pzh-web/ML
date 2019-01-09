#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/9 16:34
# @Author: peng yang
# @File  : linearDiscriminantAnalysis.py
import numpy as np

"""
线性判别分析
"""


class LinearDiscriminantAnalysis(object):
    def __init__(self):
        """
        u0、u1：样本均值向量
        w：投影直线的参数
        """
        self.w = None
        self.u0 = None
        self.u1 = None

    def get_cov_matrix(self, class_list, u):
        """
        :param class_list: 样本列表
        :param u: 样本均值向量
        :return:  样本协方差矩阵
        """
        m, n = np.shape(class_list)
        result = np.zeros(shape=(n, n))
        for sample in class_list:
            bias = sample - u
            result += np.dot(bias.reshape(n, 1), bias.reshape(1, n))
        return result / m

    def distance(self, v, w):
        """两点的距离"""
        return np.math.sqrt(self.squared_distance(v, w))

    @staticmethod
    def squared_distance(v, w):
        """
        两点之间的欧式距离
        """
        return np.dot(np.array(v) - np.array(w), np.array(v) - np.array(w))

    def get_weight(self, x, y):
        """
        求解投影直线的参数W
        w = (s0+s1).I*(u0 - u1)
        s0、s1：两类样本的协方差矩阵
        u0、u1：两类样本的均值向量
        """
        # 将样本按类别划分到两个集合
        class0 = [x[i] for i in np.where(y == 0)[0]]
        class1 = [x[i] for i in np.where(y == 1)[0]]
        # 求各类样本均值向量
        self.u0 = np.mean(class0, axis=0)
        self.u1 = np.mean(class1, axis=0)
        # 求各类样本的协方差矩阵
        s0 = self.get_cov_matrix(class0, self.u0)
        s1 = self.get_cov_matrix(class1, self.u1)
        self.w = np.mat(s0 + s1).I * (self.u0 - self.u1).reshape(21, 1)

    def predict(self, test_x, test_y):
        """
        预测样本
        """
        predict_result = []
        m, n = np.shape(test_x)
        for i in range(m):
            # 判断每个样本到两类样本均值向量的距离，划分到最近的类别
            if self.distance(test_x[i], self.u0) > self.distance(test_x[i], self.u1):
                predict_result.append(1)
            else:
                predict_result.append(0)
        success = m - np.sum([1 for i, j in zip(predict_result, test_y) if i != j])
        accuracy = float(success) / m
        print("预测正确：%d条，准确率：%f" % (success, accuracy))
        return predict_result
