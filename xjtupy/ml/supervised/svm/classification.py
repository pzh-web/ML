#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/18 10:34
# @Author: Vincent
# @File  : classification.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ML.xjtupy.ml.supervised.svm.svm import SVM

"""
非线性数据
"""
if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False
    train_path = 'train_data.txt'
    train_data = np.loadtxt(train_path, dtype=float, delimiter=',')
    train_x = train_data[:, 0:2]
    train_y = train_data[:, 2]
    svm = SVM(train_x, train_y)
    svm.smo()
    # 预测数据
    test_path = 'test_data.txt'
    test_data = np.loadtxt(test_path, dtype=float, delimiter=',')
    test_x = test_data[:, 0:2]
    test_y = test_data[:, 2]
    X = test_x[:, 0]
    Y = test_x[:, 1]
    plt.figure(figsize=(12, 5), dpi=100)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('原始测试数据')
    for i in range(len(test_y)):
        if test_y[i] > 0:
            ax1.scatter(X[i].tolist(), Y[i].tolist(), color='red', marker='+')
        else:
            ax1.scatter(X[i].tolist(), Y[i].tolist(), color='blue', marker='_')

    predict_result = svm.predict_nonlinear(test_x, test_y)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('测试结果，绿色x为错误点')
    for i in range(len(test_y)):
        if predict_result[i] != test_y[i]:
            ax2.scatter(X[i].tolist(), Y[i].tolist(), color='green', marker='x')
        else:
            if test_y[i] > 0:
                ax2.scatter(X[i].tolist(), Y[i].tolist(), color='red', marker='+')
            else:
                ax2.scatter(X[i].tolist(), Y[i].tolist(), color='blue', marker='_')
    plt.show()
