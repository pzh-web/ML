#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/13 14:47
# @Author: peng yang
# @File  : pruning.py
import numpy as np
from ML.xjtupy.ml.supervised.decisionTree.treeGenerate import TreeGenerate
from ML.xjtupy.ml.supervised.decisionTree.visualDecisionTree import VisualDecisionTree

if __name__ == '__main__':
    path = 'watermelon_pruning.txt'
    # 使用numpy读入西瓜的数据
    data = np.loadtxt(path, dtype=str, delimiter=',')
    data[:, [1, 5]] = data[:, [5, 1]]
    # 取出相应的属性(不包含连续属性)
    A = [c + '_' + str(index) for index, c in enumerate(data[0:1, 1:-3][0])]
    # 获取数据集
    train_d = data[1:11, [1, 2, 3, 4, 5, 6, 9]]
    test_d = data[11:, [1, 2, 3, 4, 5, 6, 9]]
    treeGenerate = TreeGenerate(train_d)
    tree = treeGenerate.tree_generate(train_d, A)
    # 绘制
    VisualDecisionTree().create_plot(tree)
