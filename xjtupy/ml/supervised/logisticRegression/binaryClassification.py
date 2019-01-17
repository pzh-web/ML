#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/27 21:24
# @Author: Vincent
# @File  : binaryClassificationLDA.py
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from ML.xjtupy.ml.supervised.logisticRegression.logisticRegression import LogisticRegression

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 二分类问题
    train_path = 'horseColicTraining.txt'
    test_path = 'horseColicTest.txt'
    # 使用numpy读入马训练的数据
    data = np.loadtxt(train_path, dtype=float, delimiter=',')
    # 取出相应的属性值和类别值
    '''
    split(ary, indices_or_sections, axis=0) 
    把一个数组从左到右按顺序切分 
    参数： 
    ary:要切分的数组 
    indices_or_sections:如果是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置 
    axis：沿着哪个维度进行切向，默认为0，横向切分
    '''
    train_x, train_y = np.split(data, (21,), axis=1)
    logisticR = LogisticRegression()
    # 牛顿法
    # beta = logisticR.newton_method(train_x, train_y)
    # 随机梯度法
    beta = logisticR.newton_method(train_x, train_y)

    # 测试
    test_path = 'horseColicTest.txt'
    data = np.loadtxt(test_path, dtype=float, delimiter=',')
    test_x, test_y = np.split(data, (21,), axis=1)
    predict_result = logisticR.predict(test_x, test_y)
    plt.figure(figsize=(12, 5), dpi=100)
    # 红：0；蓝：1
    color = ['r', 'b', 'g']
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('原始结果')
    ax1.scatter(test_x[:, 3], test_x[:, 4], c=[color[int(i[0])] for i in test_y], marker='o')
    # 错误点显示
    error_dot = []
    for x, y in zip(test_y, predict_result):
        if int(x[0]) != y:
            error_dot.append(2)
        else:
            error_dot.append(y)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('预测结果（绿色点为错误点）')
    ax2.scatter(test_x[:, 3], test_x[:, 4], c=[color[i] for i in error_dot], marker='o')
    plt.show()
