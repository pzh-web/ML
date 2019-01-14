#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/14 15:26
# @Author: peng yang
# @File  : regression.py

"""
回归
"""
import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.supervised.decisionTree.regTree import RegTree

if __name__ == "__main__":
    path = 'regData.txt'
    # 使用pandas读取数据
    data = np.loadtxt(path, dtype=float, delimiter=',')
    regTree = RegTree()
    tree = regTree.create_tree(data)
    print(tree)
    plt.scatter(data[:, 1], data[:, 2])
    plt.show()