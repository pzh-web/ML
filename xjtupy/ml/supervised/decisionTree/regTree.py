#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/14 16:40
# @Author: Vincent
# @File  : regTree.py
import numpy as np


class RegTree(object):

    def fleaf(self, data_set):
        """
        计算给定数据的叶节点数值, 这里为均值
        """
        data_set = np.array(data_set)
        return np.mean(data_set[:, -1])

    def ferr(self, data_set):
        """
        计算数据集的误差
        """
        data_set = np.array(data_set)
        return np.var(data_set[:, -1]) * data_set.shape[0]

    def split_data_set(self, data_set, feat_idx, value):
        """
        根据给定的特征编号和特征值对数据集进行分割
        :param feat_idx: 选取特征
        :param value: 特征值
        :return: 划分的数据集
        """
        ldata, rdata = [], []
        for data in data_set:
            if data[feat_idx] < value:
                ldata.append(data)
            else:
                rdata.append(data)
        return ldata, rdata

    def choose_best_feature(self, data_set, opt):
        """
        :param data_set: 待划分的数据集
        :param opt: 回归树参数.
        :return: 选取最佳分割特征和特征值
        """
        data_set = np.array(data_set)
        m, n = data_set.shape
        err_tolerance, n_tolerance = opt['err_tolerance'], opt['n_tolerance']
        err = self.ferr(data_set)
        best_feat_idx, best_feat_val, best_err = 0, 0, float('inf')
        # 遍历所有特征
        for feat_idx in range(n - 1):
            values = data_set[:, feat_idx]
            # 遍历所有特征值
            for val in values:
                # 按照当前特征和特征值分割数据
                ldata, rdata = self.split_data_set(data_set.tolist(), feat_idx, val)
                if len(ldata) < n_tolerance or len(rdata) < n_tolerance:
                    # 如果切分的样本量太小
                    continue
                # 计算误差
                new_err = self.ferr(ldata) + self.ferr(rdata)
                if new_err < best_err:
                    best_feat_idx = feat_idx
                    best_feat_val = val
                    best_err = new_err
        # 如果误差变化并不大归为一类
        if abs(err - best_err) < err_tolerance:
            return None, self.fleaf(data_set)
        # 检查分割样本量是不是太小
        ldata, rdata = self.split_data_set(data_set.tolist(), best_feat_idx, best_feat_val)
        if len(ldata) < n_tolerance or len(rdata) < n_tolerance:
            return None, self.fleaf(data_set)
        return best_feat_idx, best_feat_val

    def create_tree(self, data_set, opt=None):
        """
        递归创建树结构
        #data_set: 待划分的数据集
        #opt: 回归树参数.
            err_tolerance: 最小误差下降值;
            n_tolerance: 数据切分最小样本数
        """
        if opt is None:
            opt = {'err_tolerance': 1, 'n_tolerance': 4}
        # 选择最优化分特征和特征值
        feat_idx, value = self.choose_best_feature(data_set, opt)

        # 触底条件
        if feat_idx is None:
            return value
        # 创建回归树
        tree = {'feat_idx': feat_idx, 'feat_val': value}
        # 递归创建左子树和右子树
        ldata, rdata = self.split_data_set(data_set, feat_idx, value)
        ltree = self.create_tree(ldata, opt)
        rtree = self.create_tree(rdata, opt)
        tree['left'] = ltree
        tree['right'] = rtree
        return tree
