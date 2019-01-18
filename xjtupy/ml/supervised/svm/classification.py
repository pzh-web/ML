#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/18 10:34
# @Author: Vincent
# @File  : classification.py

import numpy as np

from ML.xjtupy.ml.supervised.svm.svm import SVM

if __name__ == '__main__':
    train_path = 'train_data.txt'
    test_path = 'test_data.txt'
    train_data = np.loadtxt(train_path, dtype=float, delimiter=',')
    train_x = train_data[:, 0:2]
    train_y = train_data[:, 2]
    svm = SVM(train_x, train_y)
    svm.smo()
    print(svm.alphas)
