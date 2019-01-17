#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/27 21:24
# @Author: Vincent
# @File  : mutilClassification.py

from ML.xjtupy.ml.supervised.logisticRegression.mutilLR import MutilLR
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def abalone_type(s):
    it = {b'M': 1, b'F': 2, b'I': 3}
    return it[s]


if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 多分类问题
    train_path = 'abalone.data'
    # 使用numpy读入鲍鱼的数据
    data = np.loadtxt(train_path, dtype=float, delimiter=',', converters={0: abalone_type})
    y, x = np.split(data, (-8,), axis=1)
    mutilLR = MutilLR()
    # 训练数据
    train_x = x[:4100, :]
    train_y = y[:4100]
    # 获取类别
    mutilLR.get_category(train_y)
    mutilLR.one_vs_rest(train_x, train_y)
    # 测试数据
    test_x = x[4100:, :]
    test_y = y[4100:]
    predict_result = mutilLR.predict(test_x, test_y)
    print(predict_result)
    plt.figure(figsize=(12, 5), dpi=100)
    # 分别对应：1. 2. 3. 这几个类别
    color = ['m', 'b', 'k', 'r', 'g']
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('原始结果')
    ax1.scatter(test_x[:, 1], test_x[:, 6], c=[color[int(i[0]) - 1] for i in test_y], marker='o')
    # 错误点显示
    error_dot = []
    for x, y in zip(test_y, predict_result):
        if int(x[0]) != y:
            if y == 4:
                error_dot.append(4)
            else:
                error_dot.append(3)
        else:
            error_dot.append(y - 1)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('预测结果（红色：错误点，绿色：未预测出）')
    ax2.scatter(test_x[:, 1], test_x[:, 6], c=[color[i] for i in error_dot], marker='o')
    plt.show()
