#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/9 10:39
# @Author: Vincent
# @File  : mutilLR.py
import numpy as np

from ML.xjtupy.ml.supervised.logisticRegression.logisticRegression import LogisticRegression

"""
用于多分类的逻辑回归
"""


class MutilLR(LogisticRegression):
    def __init__(self):
        self.category = []  # 类别
        self.classifiers = []  # 学得的分类器
        self.zoom_rate = []  # 缩放比例，为防止类别不平衡导致的训练误差

    def one_vs_rest(self, x, y):
        """
        使用一对多策略：将类型class1看作正样本，其他类型全部看作负样本，从而形成m个分类器
        将这m个分类器应用到测试样本上，概率最大的分类类别
        """
        # 将每个类别分别作为正例，其余作为反例
        for i in self.category:
            classify_y = [1 if i == j else 0 for j in y]
            self.zoom_rate.append(classify_y.count(0) / classify_y.count(1))
            # 训练当前分类器，即获得参数
            self.classifiers.append(self.newton_method(x, classify_y))

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
            # 用多个分类器分别取预测，选择概率最大的那个
            # 使用在缩放策略，防止类别不平衡导致的误差
            predict_value = [
                1 if (self.sigmoid(np.dot(beta, x[i])) / (1 - self.sigmoid(np.dot(beta, x[i])))) * rate > 1 else 0 for
                beta, rate in zip(self.classifiers, self.zoom_rate)]
            # 处理预测数据
            if np.max(predict_value) == 0:  # 未预测出类别
                predict_result.append(4)
            else:
                # 预测类别
                predict_classify = predict_value.index(1) + 1
                predict_result.append(predict_classify)
                if predict_classify == y[i]:
                    success += 1
        accuracy = float(success) / row_num
        print("预测正确：%d条，准确率：%f" % (success, accuracy))
        return predict_result

    def get_category(self, y):
        """
        获取分类类别
        """
        self.category = np.unique(y)
