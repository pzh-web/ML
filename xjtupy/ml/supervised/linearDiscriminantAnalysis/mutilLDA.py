#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/10 10:18
# @Author: peng yang
# @File  : mutilLDA.py
from ML.xjtupy.ml.supervised.linearDiscriminantAnalysis.linearDiscriminantAnalysis import LinearDiscriminantAnalysis
import numpy as np

from ML.xjtupy.ml.util.matrixOperate import MatrixOperate
from ML.xjtupy.ml.util.vectorOperate import VectorOperate


class MutilLDA(LinearDiscriminantAnalysis):

    def __init__(self):
        self.category = []  # 类别
        self.cate_u = []  # 类别均值向量
        self.w = None  # K维空间的投影矩阵

    def get_weight(self, x, y, k):
        """
        :param k: 降到K维
        :return: K维空间中投影矩阵
        """
        m, n = np.shape(x)
        # 类内散度矩阵
        Sw = np.zeros(shape=(n, n))
        for c in self.category:
            # 当前类的数据集
            data_c = [x[i] for i in np.where(y == c)[0]]
            datai = data_c - MatrixOperate.mean_row(data_c)
            Swi = np.mat(datai).T * np.mat(datai)
            Sw += Swi
        # 类间散度矩阵
        Sb = np.zeros(shape=(n, n))
        # 样本数据集的均值
        u = MatrixOperate.mean_row(x)
        for c in self.category:
            data_c = [x[i] for i in np.where(y == c)[0]]
            data_u = MatrixOperate.mean_row(data_c)
            self.cate_u.append(data_u)
            # 当前类别样本数
            num = np.shape(data_c)[0]
            datai = data_u - u
            Sbi = num * np.mat(datai).T * np.mat(datai)
            Sb += Sbi
        s = np.linalg.inv(Sw) * Sb
        # 求S的特征值和特征向量
        eig_values, eig_vectors = np.linalg.eig(s)
        # 返回数组值从小到大的索引
        eig_val_ind = np.argsort(eig_values)
        # 取前K维特征值较小的索引
        eig_val_ind = eig_val_ind[:(-k - 1):-1]
        # 取相应的特征向量构成投影矩阵
        self.w = eig_vectors[:, eig_val_ind]
        return self.w

    def get_category(self, train_y):
        self.category = np.unique(train_y)

    def predict_mutil(self, test_x, test_y):
        predict_result = []
        m, n = np.shape(test_x)
        for i in range(m):
            # 投影到k维空间
            low_dimension = np.dot(self.w.T, test_x[i])
            lowest_distance = 100000
            cur_type = None
            # 计算与那个类别中心点投影距离最近
            for i in range(len(self.category)):
                distance = VectorOperate.distance(low_dimension, np.dot(self.w.T, self.cate_u[i]))
                if distance < lowest_distance:
                    lowest_distance = distance
                    cur_type = self.category[i]
            predict_result.append(cur_type)
        success = m - np.sum([1 for i, j in zip(predict_result, test_y) if i != j])
        accuracy = float(success) / m
        print("预测正确：%d条，准确率：%f" % (success, accuracy))
        return predict_result
