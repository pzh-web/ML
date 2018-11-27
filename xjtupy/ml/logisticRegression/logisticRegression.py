#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018/11/27 21:19
# @Author: peng yang
# @File  : logisticRegression.py

import numpy as np

'''
logistic回归
'''


class LogisticRegression(object):
    def __init__(self):
        pass

    def sigmoid(self, wTx):
        return 1.0 / (1 + np.exp(-wTx))
