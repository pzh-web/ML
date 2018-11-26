#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/23 20:39
# @Author: peng yang
# @File  : gradientDescent.py

import math
import numpy as np

'''
梯度下降算法实现
'''


class GradientDescent(object):

    def __init__(self, step_size=0.001, random=True, ridge=False, ridge_param=None):
        self.step_size = step_size
        self.random = random
        self.ridge_param = ridge_param
        self.ridge = ridge

    def sum_of_gradient(self, x, y, thetas, dimension):
        """
        :param x: 给定训练数据
        :param y: 真实输出值
        :param thetas: 训练参数
        :param dimension: 数据维度，其中默认x0=1
        :return:
        """
        m = len(x)
        params = []
        if self.random:
            # 随机梯度下降
            # 获取样本
            sample_index = np.random.randint(0, m, 1, dtype=int)[0]
            for i in np.arange(0, dimension):
                if i == 0:
                    params.append(self.update_thetas(thetas, dimension, x, y, sample_index, -1))
                else:
                    params.append(self.update_thetas(thetas, dimension, x, y, sample_index, i))
        else:
            # 批量梯度下降
            for j in np.arange(0, dimension):
                if j == 0:
                    params.append(1.0 / m * sum([self.update_thetas(thetas, dimension, x, y, i, -1) for i in range(m)]))
                else:
                    param = 1.0 / m * sum([self.update_thetas(thetas, dimension, x, y, i, j) for i in range(m)])
                    # 做岭回归
                    if self.ridge:
                        param = param + self.ridge_param / m * thetas[j]
                    params.append(param)
        return params

    def update_thetas(self, thetas, dimension, x, y, sample_index, param_index):
        sum = 0
        param_value = 1
        for i in range(0, dimension):
            if i == 0:
                sum = sum + thetas[i]
            else:
                sum = sum + thetas[i] * x[sample_index][i - 1]
            if param_index == i:
                param_value = x[sample_index][i - 1]
        sum = sum - y[sample_index]
        sum = sum * param_value
        return sum

    def step(self, thetas, direction, step_size):
        """
        更新参数
        :param thetas:上一步的参数值
        :param direction: 代价函数对各参数求偏导后的值
        :param step_size: 学习率
        :return:
        """
        return [thetas_i + step_size * direction_i for thetas_i, direction_i
                in zip(thetas, direction)]

    def distance(self, v, w):
        """两点的距离"""
        return math.sqrt(self.squared_distance(v, w))

    def squared_distance(self, v, w):
        vector_subtract = [v_i - w_i for v_i, w_i in zip(v, w)]
        return sum(vector_subtract_i * vector_subtract_i for vector_subtract_i, vector_subtract_i
                   in zip(vector_subtract, vector_subtract))

    def gradient_descent(self, x, y, tolerance=0.000000001, max_iter=100000):
        """梯度下降"""
        # 迭代次数
        iter = 0
        # 初始化参数
        thetas = [0 for i in np.arange(0, len(x[0]) + 1)]
        while True:
            # 代价函数对参数求导
            gradient = self.sum_of_gradient(x, y, thetas, len(thetas))
            # 更新参数
            next_thetas = self.step(thetas, gradient, self.step_size)
            if self.distance(next_thetas, thetas) < tolerance:
                break
            thetas = next_thetas
            iter += 1  # update iter
            if iter == max_iter:
                print('Max iteractions exceeded!')
                break
        return thetas


if __name__ == '__main__':
    x = np.array([0.8, 1.1, 1.9, 3.1, 3.3, 3.3, 4.0, 5.1, 4.9, 6.2]).reshape(-1, 1)
    y = [110, 120, 111, 140, 150, 145, 139, 141, 155, 170]
    # 步长
    stepSize = 0.001
    g = GradientDescent(stepSize)
    t0, t1 = g.gradient_descent(x, y)
    print(t0, " ", t1)
