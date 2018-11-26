#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/23 21:29
# @Author: peng yang
# @File  : oneEleLinearRegression.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ML.xjtupy.ml.linearRegression.gradientDescent import GradientDescent

if __name__ == '__main__':
    path = 'oneEleData.csv'
    # 使用pandas读取数据
    data = pd.read_csv(path, delimiter=',')
    x = data['YearsExperience']
    y = data['Salary']
    x = np.array(x, dtype=float).reshape(-1, 1)
    y = np.array(y, dtype=float)
    plt.grid()
    plt.plot(x, y, 'ro')
    # 步长
    stepSize = 0.001
    g = GradientDescent(stepSize, False)
    t0, t1 = g.gradient_descent(x, y)
    y = t0 + t1 * x
    plt.plot(x, y, 'g-')
    plt.show()
