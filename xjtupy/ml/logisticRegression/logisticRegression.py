#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/27 21:19
# @Author: peng yang
# @File  : logisticRegression.py

import numpy as np

"""
logistic回归
"""


class LogisticRegression(object):
    def __init__(self):
        self.beta = None
        self.accuracy = 0.0

    @staticmethod
    def sigmoid(x):
        """
        S函数
        :param x: 样本特征值
        :return: 样本p(y=1 | x;beta）的概率
        """
        return 1.0 / (1 + np.exp(-x))

    def newton_method(self, x, y, iter=50):
        """
        牛顿法
        """
        # 在样本值向量前面增加一列(1;x)
        b = np.ones(len(x))
        x = np.insert(x, 0, values=b, axis=1)
        # 初始化参数
        self.beta = np.zeros(len(x[0]), dtype=float)

        # 行数
        row_num = len(x)
        # 列数
        column_num = len(x[0])

        # 使用牛顿法，直至beta收敛
        for it in range(iter):
            # 一阶导的值
            first_derivative = np.zeros(column_num)
            # 二阶导的值,单特征的时候是一个值，多特征是一个hessian矩阵
            second_derivative = np.zeros(shape=(column_num, column_num))
            for i in np.arange(0, row_num):
                # 为反例的概率
                p0 = self.sigmoid(-np.dot(self.beta, x[i]))
                # 为正例的概率
                p1 = self.sigmoid(np.dot(self.beta, x[i]))
                first_derivative = first_derivative + (p1 - y[i]) * x[i]
                hessian = [b * x[i] * p0 * p1 for b in x[i]]
                second_derivative = np.add(second_derivative, hessian)
            # 更新参数
            next_beta = self.beta - np.dot(np.mat(second_derivative).I, first_derivative)
            # 返回一个扁平（一维）的数组（ndarray）
            next_beta = next_beta.getA1()
            self.beta = next_beta

    def predict(self, x, y):
        """
        预测样本
        """
        predict_result = []
        # 在样本值向量前面增加一列(1;x)
        b = np.ones(len(x))
        x = np.insert(x, 0, values=b, axis=1)
        # 行数
        row_num = len(x)
        success = 0
        for i in range(row_num):
            predict_value = 1 if self.sigmoid(np.dot(self.beta, x[i])) > 0.5 else 0
            predict_result.append(predict_value)
            if predict_value == y[i]:
                success += 1
        accuracy = float(success) / row_num
        print("预测正确：%d条，准确率：%f" % (success, accuracy))
        return predict_result
