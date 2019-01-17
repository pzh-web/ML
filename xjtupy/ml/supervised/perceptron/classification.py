#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/15 10:39
# @Author: Vincent
# @File  : classification.py

import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.supervised.perceptron.perceptron import Perceptron

if __name__ == '__main__':
    data = np.loadtxt('data.txt', dtype=float, delimiter=',')
    train_x, train_y = data[:80, :2], data[:80, 2]
    test_x, test_y = data[80:, :2], data[80:, 2]
    fig = plt.figure('训练数据')
    ax = fig.add_subplot(111)
    X = train_x[:, 0]
    Y = train_x[:, 1]
    for i in range(len(train_y)):
        if train_y[i] > 0:
            ax.scatter(X[i].tolist(), Y[i].tolist(), color='red', marker='+')
        else:
            ax.scatter(X[i].tolist(), Y[i].tolist(), color='green', marker='_')
    p = Perceptron()
    # 训练数据
    w = p.original_form(train_x, train_y)
    # 测试数据
    p.predict(test_x, test_y, w)
    o1 = w[0]
    o2 = w[1]
    o3 = w[2]
    x = np.linspace(3, 6, 50)
    y = (-o1 * x - o3) / o2
    ax.plot(x, y)
    plt.show()
