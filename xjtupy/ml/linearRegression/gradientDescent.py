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
    def __init__(self, learn_rate=0.001, random=True, ridge=False, ridge_param=2):
        '''
        :param learn_rate: 梯度下降算法的步长
        :param random: 是否使用随机梯度下降算法
        :param ridge: 是否使用岭回归
        :param ridge_param: 正则化参数
        '''
        self.learn_rate = learn_rate
        self.random = random
        self.ridge_param = ridge_param
        self.ridge = ridge

    def sum_of_gradient(self, sample, true_value, thetas, dimension):
        """
        损失函数梯度和
        :param sample: 给定训练数据
        :param true_value: 真实输出值
        :param thetas: 训练参数,theta0放在最前面
        :param dimension: 数据维度，其中默认x0=1
        :return: 损失函数对参数求偏导
        """
        sample_num = len(sample)
        params = []
        if self.random:
            # 随机梯度下降
            # 获取样本
            sample_index = np.random.randint(0, sample_num, 1, dtype=int)[0]
            for param_index in np.arange(0, dimension):
                param = self.update_thetas(thetas, sample, true_value, sample_index, param_index)
                # 做岭回归
                if self.ridge and param_index != 0:
                    param = param + self.ridge_param * thetas[param_index]
                params.append(param)
        else:
            # 批量梯度下降
            for param_index in np.arange(0, dimension):
                param = 1.0 / sample_num * sum(
                    [self.update_thetas(thetas, sample, true_value, sample_index, param_index)
                     for sample_index in range(sample_num)])
                # 做岭回归
                if self.ridge and param_index != 0:
                    param = param + self.ridge_param / sample_num * thetas[param_index]
                params.append(param)
        return params

    @staticmethod
    def update_thetas(thetas, sample, true_value, sample_index, param_index):
        sample_num = len(sample)
        b = np.ones(sample_num)
        sample = np.insert(sample, 0, values=b, axis=1)
        if param_index == 0:
            param_value = 1
        else:
            param_value = sample[sample_index][param_index]
        return (np.dot(thetas, sample[sample_index]) - true_value[sample_index]) * param_value

    def step(self, thetas, direction):
        """
        更新参数
        :param thetas:上一步的参数值
        :param direction: 代价函数对各参数求偏导后的值
        :return:
        """
        return np.array(thetas) - self.learn_rate * np.array(direction)

    def cost_function(self, sample, true_value, params):
        '''
        代价函数
        :param sample: 样本
        :param true_value: 样本实际输出值
        :param params: 当前参数
        :return: 返回当前代价
        '''
        sample_num = len(sample)
        cost = 0
        b = np.ones(sample_num)
        sample = np.insert(sample, 0, values=b, axis=1)
        for i in range(sample_num):
            # 模型计算出来的值
            fx = np.dot(sample[i], params)
            # 误差
            err = fx - true_value[i]
            cost = cost + np.dot(err.T, err)
        return 1.0 / (2 * sample_num) * cost

    def distance(self, v, w):
        """两点的距离"""
        return math.sqrt(self.squared_distance(v, w))

    @staticmethod
    def squared_distance(v, w):
        return np.dot(np.array(v) - np.array(w), np.array(v) - np.array(w))

    def gradient_descent(self, sample, true_value, tolerance=0.00000001, min_cost=0.25, max_iter=1000000):
        """梯度下降"""
        # 迭代次数
        iter = 0
        # 初始化参数,将b视为theta0
        thetas = [0 for i in np.arange(0, len(sample[0]) + 1)]
        while True:
            # 代价函数对参数求导
            gradient = self.sum_of_gradient(sample, true_value, thetas, len(thetas))
            # 计算代价
            # if self.cost_function(sample, true_value, thetas) < min_cost:
            #     break

            # 更新参数
            next_thetas = self.step(thetas, gradient)
            # 相邻两次参数更新值更接近，直到收敛到某个值
            if self.distance(next_thetas, thetas) < tolerance:
                break
            thetas = next_thetas
            # 更新学习率
            # self.update_learn_rate(iter)
            iter += 1
            if iter == max_iter:
                print('超过最大迭代次数')
                break
        return thetas

    def update_learn_rate(self, iter):
        if iter == 100:
            self.learn_rate = 0.1
        elif iter == 1000:
            self.learn_rate = 0.01
        elif iter == 100000:
            self.learn_rate = 0.001


if __name__ == '__main__':
    x = np.array([0.8, 1.1, 1.9, 3.1, 3.3, 3.3, 4.0, 5.1, 4.9, 6.2]).reshape(-1, 1)
    y = [110, 120, 111, 140, 150, 145, 139, 141, 155, 170]
    # 步长
    g = GradientDescent()
    t0, t1 = g.gradient_descent(x, y)
    print(t0, " ", t1)
