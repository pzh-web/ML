#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/27 21:24
# @Author: peng yang
# @File  : mutilClassification.py
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.logisticRegression.logisticRegression import LogisticRegression
from ML.xjtupy.ml.logisticRegression.mutilLR import MutilLR
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 多分类问题
    train_path = 'glass.data'
    # 使用numpy读入花的数据，第四列使用iris_type单独处理
    data = np.loadtxt(train_path, dtype=float, delimiter=',')
    x, y = np.split(data, (10,), axis=1)
    mutilLR = MutilLR()
    # 训练数据
    train_x = x[:-24, 1:]
    train_y = y[:-24]
    # 获取类别
    mutilLR.get_category(train_y)
    mutilLR.one_vs_rest(train_x, train_y)
    # 测试数据
    test_x = x[-24:, 1:]
    test_y = y[-24:]
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
