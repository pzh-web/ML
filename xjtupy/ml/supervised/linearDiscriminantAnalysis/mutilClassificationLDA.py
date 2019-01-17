#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/10 10:13
# @Author: Vincent
# @File  : mutilClassificationLDA.py

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.supervised.linearDiscriminantAnalysis.mutilLDA import MutilLDA


def abalone_type(s):
    it = {b'M': 1, b'F': 2, b'I': 3}
    return it[s]


if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 多分类问题
    train_path = 'abalone.data'
    # 使用numpy读入鲍鱼的数据，第一列使用abalone_type单独处理
    data = np.loadtxt(train_path, dtype=float, delimiter=',', converters={0: abalone_type})
    y, x = np.split(data, (-8,), axis=1)
    mutilLDA = MutilLDA()
    # 训练数据
    train_x = x[:4100, :]
    train_y = y[:4100]
    # 获取类别
    mutilLDA.get_category(train_y)
    mutilLDA.get_weight(train_x, train_y, 3)
    # 测试数据
    test_x = x[4100:, :]
    test_y = y[4100:]
    predict_result = mutilLDA.predict_mutil(test_x, test_y)
    plt.figure(figsize=(12, 5), dpi=100)
    # 分别对应：1. 2. 3. 这几个类别
    color = ['r', 'g', 'b']
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('原始结果')
    ax1.scatter(test_x[:, 1], test_x[:, 6], c=[color[int(i[0]) - 1] for i in test_y], marker='o')
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('预测结果')
    ax2.scatter(test_x[:, 1], test_x[:, 6], c=[color[int(i) - 1] for i in predict_result], marker='o')
    plt.show()
