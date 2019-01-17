#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/12 17:10
# @Author: Vincent
# @File  : classification.py

import numpy as np

from ML.xjtupy.ml.supervised.decisionTree.treeGenerate import TreeGenerate
from ML.xjtupy.ml.supervised.decisionTree.visualDecisionTree import VisualDecisionTree


"""
分类
"""
if __name__ == '__main__':
    path = 'watermelon.txt'
    # 使用numpy读入西瓜的数据
    data = np.loadtxt(path, dtype=str, delimiter=',')
    # 取出相应的属性(不包含连续属性)
    # A = [c + '_' + str(index) for index, c in enumerate(data[0:1, 1:-3][0])]
    # 获取数据集(不包含连续属性)
    # D = data[1:, [1, 2, 3, 4, 5, 6, 9]]

    # 取出相应的属性(包含连续属性)
    A = [c + '_' + str(index) for index, c in enumerate(data[0:1, 1:-1][0])]
    # 获取数据集(不包含连续属性)
    D = data[1:, 1:]
    treeGenerate = TreeGenerate(D)
    tree = treeGenerate.tree_generate(D, A)
    # 绘制
    VisualDecisionTree().create_plot(tree, '二分类决策树（包含连续属性）')
