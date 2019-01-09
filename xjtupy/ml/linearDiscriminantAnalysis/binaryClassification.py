#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/9 16:35
# @Author: peng yang
# @File  : binaryClassification.py

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.linearDiscriminantAnalysis.linearDiscriminantAnalysis import LinearDiscriminantAnalysis

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 二分类问题
    train_path = 'horseColicTraining.txt'
    test_path = 'horseColicTest.txt'
    # 使用numpy读入马训练的数据
    data = np.loadtxt(train_path, dtype=float, delimiter=',')
    # 取出相应的属性值和类别值
    train_x, train_y = np.split(data, (21,), axis=1)
    LDA = LinearDiscriminantAnalysis()

    # 测试
    test_path = 'horseColicTest.txt'
    data = np.loadtxt(test_path, dtype=float, delimiter=',')
    test_x, test_y = np.split(data, (21,), axis=1)
