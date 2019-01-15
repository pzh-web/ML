#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/15 10:39
# @Author: peng yang
# @File  : classification.py

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ML.xjtupy.ml.supervised.perceptron.perceptron import Perceptron

if __name__ == '__main__':
    df = pd.read_csv('iris.data', header=None)
    y = df.iloc[:, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, 1:3].values
    min_x = X[:, 1].min()
    max_x = X[:, 1].max()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    ax.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    w = Perceptron().original_form(X, y)
    b = np.ones(len(X))
    X = np.insert(X, 0, values=b, axis=1)
    plt.plot(np.linspace(min_x, max_x, len(X)), np.dot(X, w))
    plt.xlabel('petal length')
    plt.ylabel('sepal lenght')
    plt.legend(loc='upper left')
    plt.show()
