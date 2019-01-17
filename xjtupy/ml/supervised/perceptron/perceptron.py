#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/15 10:40
# @Author: Vincent
# @File  : perceptron.py
import numpy as np

"""
感知机算法实现
损失函数对应误分类点到超平面的距离和变小:-np.sum(Yi(wXi+b))
基于随机梯度下降算法
"""


class Perceptron(object):

    def original_form(self, x, y, eta=1):
        """
        原始形式
        :param x:
        :param y:
        :param eta: 学习率
        :return: 超平面参数
        """
        # 在样本值向量前面增加一列(1;x)
        b = np.ones(len(x))
        x = np.insert(x, 2, values=b, axis=1)
        m, n = np.shape(x)
        # 初始化参数(w,b)
        beta = np.zeros(n, dtype=float)
        # 选择一个当前参数下的误分类点
        cur_x, cur_y = self.error_classify_sample(x, y, beta)
        while cur_x is not None:
            # 更新参数
            beta += eta * cur_y * cur_x
            # 迭代，直到收敛
            cur_x, cur_y = self.error_classify_sample(x, y, beta)
        return beta

    def dual_form(self, x, y, eta=1):
        """
        对偶形式
        """
        m, n = np.shape(x)
        # 初始化参数α，b
        alpha = np.zeros(m, dtype=float)
        b = 0
        # 选择一个当前参数下的误分类点
        cur_index, cur_y = self.error_classify_sample2(x, y, alpha, b)
        while cur_y is not None:
            # 更新参数
            alpha[cur_index] += eta
            b += eta * cur_y
            # 迭代，直到收敛
            cur_index, cur_y = self.error_classify_sample2(x, y, alpha, b)
        return alpha, b

    def predict(self, x, y, w):
        b = np.ones(len(x))
        x = np.insert(x, 2, values=b, axis=1)
        m, n = np.shape(x)
        accuracy = np.sum([1 if np.sign(np.dot(cur_x, w)) == cur_y else 0 for cur_x, cur_y in zip(x, y)])
        print('测试精度：%f' % (accuracy / m))

    def error_classify_sample(self, x, y, beta):
        """
        选取误分类点，用于原始形式
        """
        for cur_x, cur_y in zip(x, y):
            if cur_y * np.dot(cur_x, beta) <= 0:
                return cur_x, cur_y
        return None, None

    def error_classify_sample2(self, x, y, alpha, b):
        """
        选取误分类点，用于对偶形式
        """
        for index, (cur_x, cur_y) in enumerate(zip(x, y)):
            if cur_y * (
                    np.sum([alpha[i] * temp_y * np.dot(temp_x, cur_x) for i, (temp_x, temp_y) in
                            enumerate(zip(x, y))]) + b) <= 0:
                print(index, cur_y)
                return index, cur_y
        return None, None
