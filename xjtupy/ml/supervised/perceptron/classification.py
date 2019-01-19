#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/15 10:39
# @Author: Vincent
# @File  : classification.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ML.xjtupy.ml.supervised.perceptron.perceptron import Perceptron

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False
    data = np.loadtxt('data.txt', dtype=float, delimiter=',')
    train_x, train_y = data[:80, :2], data[:80, 2]
    test_x, test_y = data[80:, :2], data[80:, 2]
    fig = plt.figure('训练数据')
    plt.title('感知机')
    ax = fig.add_subplot(111)
    X = train_x[:, 0]
    Y = train_x[:, 1]
    type2_x, type2_y, type3_x, type3_y = [], [], [], []
    for i in range(len(train_y)):
        if train_y[i] > 0:
            type2_x.append(X[i].tolist())
            type2_y.append(Y[i].tolist())
        else:
            type3_x.append(X[i].tolist())
            type3_y.append(Y[i].tolist())
    type2 = ax.scatter(type2_x, type2_y, color='red', marker='+')
    type3 = ax.scatter(type3_x, type3_y, color='green', marker='_')
    p = Perceptron()
    # 训练数据
    w = p.original_form(train_x, train_y)
    print(w)
    # 测试数据
    p.predict(test_x, test_y, w)
    o1 = w[0]
    o2 = w[1]
    o3 = w[2]
    x = np.linspace(3, 6, 50)
    y = (-o1 * x - o3) / o2
    ax.plot(x, y)
    plt.legend((type2, type3), ('正样本', '负样本'))
    plt.show()
