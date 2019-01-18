#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/18 10:55
# @Author: Vincent
# @File  : svm.py

import numpy as np

"""
支持向量机实现
公式来源：李航《统计学习方法》
"""


class SVM(object):

    def __init__(self, x, y, c=0.06, tolerance=0.001, kernel_option=('linear', 1.3)):
        """
        :param x: 训练样本
        :param y: 训练样本标签
        :param c: 软间隔SVM参数
        :param tolerance:阈值，用于验证KKT条件
        :param kernel_option:核操作参数（核函数类型，和函数参数）
        """
        self.train_x = x
        self.train_y = y
        self.c = c
        self.tolerance = tolerance
        self.kernel_option = kernel_option
        # 样本数、特征维数
        self.m, self.n = np.shape(x)
        # 初始化参数
        self.b = 0
        self.alphas = np.zeros(self.m)
        # 误差存储向量，即E值,为了节省计算时间
        self.eCache = np.zeros(self.m)
        # 核矩阵
        self.kernel_matrix = self.cal_kernel_matrix()

    def cal_kernel_matrix(self):
        """  计算核矩阵 """
        kernel_matrix = np.mat(np.zeros(shape=(self.m, self.m)))
        for i in range(self.m):
            kernel_matrix[:, i] = self.cal_kernel_value(self.train_x, self.train_x[i, :])
        return kernel_matrix

    def cal_kernel_value(self, train_x, sample):
        """  计算核值 """
        kernel_type = self.kernel_option[0]
        kernel_value = np.mat(np.zeros(shape=(self.m, 1)))

        if kernel_type == 'linear':  # 线性核
            kernel_value = np.mat(np.dot(train_x, sample)).reshape(100, 1)
        elif kernel_type == 'rbf':  # 高斯核
            sigma = self.kernel_option[1]
            if sigma == 0.0:
                sigma = 1.0
            for i in range(self.m):
                diff = train_x[i, :] - sample
                kernel_value[i] = np.exp(-np.dot(diff, diff) / (2 * sigma ** 2))
        return kernel_value

    def smo(self, max_iter=50):
        """
        使用smo算法来优化参数α
        """
        iter_num = 0
        changedAlphasNum = 0
        entireSet = True

        # 外层循环，选择第一个变量alpha_i  -- 128页
        # 循环终止条件
        #   1、达到最大迭代次数
        #   2、所有的alpha都满足KKT条件
        while iter_num < max_iter and (entireSet or changedAlphasNum > 0):
            if entireSet:
                for i in range(self.m):  # 遍历整个数据集，看是否满足KKT条件
                    changedAlphasNum += self.inner_loop(i)
                iter_num += 1
            else:
                # 非边界数据索引
                nonBoundSampleIndexs = [index for index, alpha in enumerate(self.alphas) if
                                        not (alpha > 0 and alpha > self.c)]
                for i in nonBoundSampleIndexs:
                    changedAlphasNum += self.inner_loop(i)
                iter_num += 1

            if entireSet:
                entireSet = False
            elif changedAlphasNum == 0:
                entireSet = True

    def inner_loop(self, index_i):
        """
        内层循环，选择第二个变量alpha_j
        """
        # 用于选择一个使得|error_i - error_j|最大的alpha_j
        error_i = self.cal_error(index_i)
        self.update_error(error_i)
        # 满足KKT的条件
        # 1) yi*f(i) >= 1 and alpha == 0 (边界外)
        # 2) yi*f(i) == 1 and 0<alpha< C (边界上)
        # 3) yi*f(i) <= 1 and alpha == C (边界内)
        # 违背KKT条件
        # 因为y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, 所以
        # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, 违背
        # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, 违背
        # 3) if y[i]*E_i = 0, so yi*f(i) = 1, 在边界上，不需要优化
        #  判断当前alpha_i是否违背KKT条件  --128页：公式(7.111)-(7.113)
        # 违背KKT条件，将其作为当前的第一个变量alpha_i
        if (self.train_y[index_i] * self.eCache[index_i] < -self.tolerance and self.alphas[index_i] < self.c) or (
                self.train_y[index_i] * self.eCache[index_i] > self.tolerance and self.alphas[index_i] > 0):
            # 1、选择alpha_j,并返回error_j
            index_j, error_j = self.select_j(index_i, error_i)
            old_alpha_i = self.alphas[index_i]
            old_alpha_j = self.alphas[index_j]
            # 2、计算alpha_j的取值范围边界[L,H]  --126页
            if self.train_y[index_i] == self.train_y[index_j]:
                L = np.max(0, old_alpha_j + old_alpha_i - self.c)
                H = np.min(self.c, old_alpha_j + old_alpha_i)
            else:
                L = np.max(0, old_alpha_j - old_alpha_i)
                H = np.min(self.c, self.c + old_alpha_j - old_alpha_i)
            if L == H:
                return 0
            # 3、计算eta  --127页：公式(7.107)
            eta = self.kernel_matrix[index_i:index_i] + self.kernel_matrix[index_j:index_j] - 2 * self.kernel_matrix[
                                                                                                  index_i:index_j]
            if eta >= 0:
                return 0
            # 4、计算未经剪辑的alpha_j，即可能得出的alpha_j没有在约束条件[L,H]  --127页：公式(7.106)
            unc_alpha_j = old_alpha_j + self.train_y[index_j] * (self.eCache[index_i] - self.eCache[index_j]) / eta
            # 5、计算优化后alpha_j  --127页：公式(7.108)
            self.alphas[index_j] = self.clip_alpha_j(unc_alpha_j, L, H)
            # alpha_j的更新变化小于阈值
            if np.abs(old_alpha_j - self.alphas[index_j]) < self.tolerance:
                self.update_error(index_j)
                return 0
            # 6、计算优化后alpha_i  --127页：公式(7.109)
            self.alphas[index_i] = old_alpha_i + self.train_y[index_i] * self.train_y[index_j] * (
                    old_alpha_j - self.alphas[index_j])
            # 7、更新b  --130页：公式(7.115)、(7.116)
            b1 = self.b - self.eCache[index_i] - self.train_y[index_i] * self.kernel_matrix[index_i:index_i] * (
                    self.alphas[index_i] - old_alpha_i) - self.train_y[index_j] * self.kernel_matrix[
                                                                                  index_j:index_i] * (
                         self.alphas[index_j] - old_alpha_j)
            b2 = self.b - self.eCache[index_j] - self.train_y[index_i] * self.kernel_matrix[index_i:index_j] * (
                    self.alphas[index_i] - old_alpha_i) - self.train_y[index_j] * self.kernel_matrix[
                                                                                  index_j:index_j] * (
                         self.alphas[index_j] - old_alpha_j)
            if (self.alphas[index_i] > 0) and (self.alphas[index_i] < self.c):
                self.b = b1
            elif (self.alphas[index_j] > 0) and (self.alphas[index_j] < self.c):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2
            # 8、更新eCache中的error_i,error_j  --130页：公式(7.117)
            self.update_error(index_i)
            self.update_error(index_j)
            # 优化一次
            return 1
        else:  # 不违背KKT条件
            return 0

    def cal_error(self, index):
        """
        计算当前选择的样本的误差   --127页：7.105
        :param index: 样本索引
        :return:
        """
        print(self.kernel_matrix[0][index])
        return np.sum(
            [self.alphas[i] * self.train_y[i] * self.kernel_matrix[i:index]
             for i in range(self.m)]) + self.b - self.train_y[index]

    def update_error(self, index):
        error = self.cal_error(index)
        self.eCache[index] = error

    def select_j(self, index_i, error_i):
        """
        选择alpha_j，使得|error_i - error_j|
        -- 129页：第2个变量的选择
        :param index_i: 第一个变量的索引
        :param error_i:  第一个变量的误差
        :return: 第一个变量的索引、误差
        """
        index_j = 0
        error_j = 0
        max_error = 0
        # 获取误差矩阵中非零元素的索引，用于备选的alpha_j，从而优化它
        need_optimize_alpha = [i for i in self.eCache if self.eCache[i] is not 0]
        if len(need_optimize_alpha) > 1:
            for index in need_optimize_alpha:
                if index == index_i:
                    continue
                else:
                    # 计算当前索引的误差
                    cur_error = self.cal_error(index)
                    abs_error = np.abs(cur_error, error_i)
                    if abs_error > max_error:
                        max_error = abs_error
                        index_j = index
                        error_j = cur_error
        else:
            index_j = self.random_select_j(index_i)
            error_j = self.cal_error(index_j)
        return index_j, error_j

    def random_select_j(self, index_i):
        """
        随机选择一个alpha_j
        """
        index_j = index_i
        while index_j == index_i:
            index_j = int(np.random.uniform(0, self.m))
        return index_j

    def clip_alpha_j(self, unc_alpha_j, L, H):
        """
        剪辑alpha_j,将alpha_j的取值限定到[L,H]
        :param unc_alpha_j: 未经剪辑的
        :param L: 优化后alpha_j的下界
        :param H: 优化后alpha_j的上界
        :return: 优化后alpha_j
        """
        if unc_alpha_j > H:
            return H
        elif unc_alpha_j < L:
            return L
        else:
            return unc_alpha_j
