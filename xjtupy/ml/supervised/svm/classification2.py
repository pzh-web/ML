#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/18 10:34
# @Author: Vincent
# @File  : classification.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ML.xjtupy.ml.supervised.svm.svm import SVM
from ML.xjtupy.ml.util.matrixOperate import MatrixOperate

"""
使用和感知机算法一样的数据验证SVM
线性数据
"""
if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False
    train_path = 'data.txt'
    data = np.loadtxt(train_path, dtype=float, delimiter=',')
    # train_x = data[:, 0:2]
    # train_y = data[:, 2]
    train_x, train_y = data[:80, :2], data[:80, 2]
    test_x, test_y = data[80:, :2], data[80:, 2]
    svm = SVM(train_x, train_y, kernel_option=('linear', 0))
    svm.smo()

    predict_result = svm.predict_linear(test_x, test_y)

    fig = plt.figure('训练数据')
    plt.title('支持向量机')
    ax = fig.add_subplot(111)
    X = train_x[:, 0]
    Y = train_x[:, 1]

    type1_x, type1_y, type2_x, type2_y, type3_x, type3_y = [], [], [], [], [], []
    for i in range(len(train_y)):
        if svm.alphas[i] > 0:
            type1_x.append(X[i].tolist())
            type1_y.append(Y[i].tolist())
        else:
            if train_y[i] > 0:
                type2_x.append(X[i].tolist())
                type2_y.append(Y[i].tolist())
            else:
                type3_x.append(X[i].tolist())
                type3_y.append(Y[i].tolist())
    type1 = ax.scatter(type1_x, type1_y, color='blue', marker='x')
    type2 = ax.scatter(type2_x, type2_y, color='red', marker='+')
    type3 = ax.scatter(type3_x, type3_y, color='green', marker='_')
    # 求w
    matrix = np.array([svm.alphas[i] * train_y[i] * train_x[i] for i in range(svm.m)])
    w = MatrixOperate.sum_row(matrix)
    o1 = w[0]
    o2 = w[1]
    x = np.linspace(3, 6, 50)
    y = (-o1 * x - svm.b) / o2
    plt.plot(x, y)
    plt.legend((type1, type2, type3), ('支持向量', '正样本', '负样本'))
    plt.show()
