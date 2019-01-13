#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/12 19:14
# @Author: peng yang
# @File  : treeGenerate.py
import numpy as np

"""
决策树生成
"""


class TreeGenerate(object):

    def tree_generate(self, D, A):
        """
        :param D: 数据集 {（x1,y1）,(x2,y2),...,(xm,ym)}
        :param A: 属性集 {a1,a2,...,ad}
        :return: 决策树
        """
        # 判断D中样本是不是属于同一类
        if D[:, -1].tolist().count(D[:, -1][0]) == len(D):
            return D[:, -1][0]

        if len(A) == 0:
            return self.most_category(D)

        # 选择最优属性:对应原始索引
        optimalAttrIndex, cur_index = self.ID3(D, A)
        # 生成一个节点
        optimalAttr = A[cur_index].split('_')[0]
        tree = {optimalAttr: {}}
        # 从属性集合中删除最优属性
        A.remove(A[cur_index])
        # 遍历当前属性的所有取值
        for attrValue in np.unique(D[:, optimalAttrIndex]):
            sub_labels = A[:]
            # 递归生成决策树
            tree[optimalAttr][attrValue] = self.tree_generate(
                self.split_data_set(D, optimalAttrIndex, attrValue), sub_labels)
        return tree

    def ID3(self, D, A):
        """
        使用信息增益选择最优属性
        """
        # 原始属性集的索引
        optimalAttrIndex = 0
        # 删除某些属性之后的索引
        optimalCurAttrIndex = 0
        information_gain = 0
        # 数据集合的信息熵
        information_entropy = self.cal_information_entropy(D)
        # 获取最优属性索引
        for cur_index, attr in enumerate(A):
            original_index = int(attr.split('_')[1])
            attrs = np.unique(D[:, original_index])
            infor_entropy_sum = 0
            for attr in attrs:
                cur_data = self.split_data_set(D, original_index, attr)
                # 计算当前属性在当前取值下的信息熵
                cur_information_entropy = self.cal_information_entropy(cur_data)
                infor_entropy_sum += len(cur_data) / len(D) * cur_information_entropy
            if (information_entropy - infor_entropy_sum) > information_gain:
                information_gain = information_entropy - infor_entropy_sum
                optimalAttrIndex = original_index
                optimalCurAttrIndex = cur_index
        return optimalAttrIndex, optimalCurAttrIndex

    def C4_5(self, D, A):
        """
        C4.5决策树算法使用增益率来选择最优属性
        """
        pass

    def most_category(self, D):
        """
        :return: 返回数据集中数据量最多的类别
        """
        category = ''
        num = 0
        for c in np.unique(D[:, -1]):
            category = c if D[:, -1].tolist().count(c) > num else category
            num = D[:, -1].tolist().count(c)
        return category

    def split_data_set(self, D, optimalAttrIndex, attrValue):
        """
        :param D:
        :param optimalAttrIndex: 最优属性索引
        :param attrValue: 最优属性的某一个取值
        :return:
        """
        return np.array([row for row in D if row[optimalAttrIndex] == attrValue])

    def cal_information_entropy(self, D):
        """
        :param D:
        :return: 返回信息熵
        """
        category = np.unique(D[:, -1])
        return -np.sum(
            [D[:, -1].tolist().count(c) / len(D[:, -1]) * np.log2(D[:, -1].tolist().count(c) / len(D[:, -1])) for c in
             category])
