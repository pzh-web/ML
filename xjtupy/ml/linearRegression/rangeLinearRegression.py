#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/23 21:29
# @Author: peng yang
# @File  : oneEleLinearRegression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.linearRegression.gradientDescent import GradientDescent

if __name__ == '__main__':
    path = 'multiEleData.csv'
    # 使用pandas读取数据
    data = pd.read_csv(path, delimiter=',')
    x = data[['x1', 'x2']]
    y = data['y']
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # 步长
    stepSize = 0.001
    g = GradientDescent(stepSize)
    t0, t1, t2 = g.gradient_descent(x, y)
    # 给出待预测的特征（4个），并求得预测值
    predict_x = pd.DataFrame([[0, 0], [0, 11], [11, 0], [11, 11]])
    predict_y = predict_x[0] * t1 + predict_x[1] * t2 + t0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 真实数据
    ax.scatter(x[:, :1], x[:, 1:2], y, s=150)
    # 预测数据
    ax.scatter(predict_x[0], predict_x[1], predict_y, color='red', s=200)
    # 拟合平面
    # np.meshgrid----生成网格点坐标矩阵
    x1, x2 = np.meshgrid(x[:, :1], x[:, 1:2])
    z = t0 + t1 * x1 + t2 * x2
    ax.plot_surface(x1, x2, z, alpha=0.04, color='red')
    plt.show()
