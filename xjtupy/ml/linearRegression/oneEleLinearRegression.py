#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/23 21:29
# @Author: peng yang
# @File  : oneEleLinearRegression.py
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ML.xjtupy.ml.linearRegression.gradientDescent import GradientDescent
from ML.xjtupy.ml.linearRegression.normalEquation import NormalEquation

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    path = 'oneEleData.csv'
    # 使用pandas读取数据
    data = pd.read_csv(path, delimiter=',')
    x = data['YearsExperience']
    y = data['Salary']
    x = np.array(x, dtype=float).reshape(-1, 1)
    y = np.array(y, dtype=float)
    plt.grid()
    plt.plot(x, y, 'ro')
    # 梯度下降法
    g = GradientDescent(random=False, ridge=True)
    t0, t1 = g.gradient_descent(x, y)
    fx = t0 + t1 * x
    plt.plot(x, fx, 'g-', label='梯度下降法-岭回归')

    # 正规方程法
    n = NormalEquation(True)
    fx = n.normal_equation_method(x, y)
    plt.plot(x, fx, linestyle='--', color='blue', label='正规方程法-正则化')
    plt.legend(loc='lower right')
    plt.show()
