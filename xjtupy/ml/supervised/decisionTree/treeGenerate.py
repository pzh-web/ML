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
        # optimalAttrIndex, cur_index = self.ID3(D, A)
        optimalAttrIndex, cur_index = self.C4_5(D, A)
        # optimalAttrIndex, cur_index = self.CATR(D, A)
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
        # 原始属性集的索引
        optimalAttrIndex = 0
        # 删除某些属性之后的索引
        optimalCurAttrIndex = 0
        # 当前属性集的信息增益和增益率
        information_gain = []
        gain_ratio = []
        # 数据集合的信息熵
        information_entropy = self.cal_information_entropy(D)
        for attr in A:
            original_index = int(attr.split('_')[1])
            attrs = np.unique(D[:, original_index])
            infor_entropy_sum = 0
            # 固有值
            IV = 0
            for attr in attrs:
                cur_data = self.split_data_set(D, original_index, attr)
                # 计算当前属性在当前取值下的信息熵
                cur_information_entropy = self.cal_information_entropy(cur_data)
                infor_entropy_sum += len(cur_data) / len(D) * cur_information_entropy
                IV += len(cur_data) / len(D) * np.log2(len(cur_data) / len(D))
            # 计算当前属性的信息增益和增益率
            attr_information_gain = information_entropy - infor_entropy_sum
            attr_gain_ratio = -attr_information_gain / IV
            information_gain.append(attr_information_gain)
            gain_ratio.append(attr_gain_ratio)

        # 1、找出信息增益高于平均水平的属性
        ave_information_entropy = np.sum(information_gain) / len(A)
        # new_attr中元素格式：色泽_index1_index2
        # index1：代表原始属性集中当前属性的索引
        # index2：代表当前属性集（子树集合）中属性索引
        new_attr = [A[index] + '_' + str(index) for index, ig in enumerate(information_gain) if
                    ig > ave_information_entropy]
        # 2、在从中找出增益率最高的属性
        temp_gain_ratio = 0
        for attr in new_attr:
            original_index = int(attr.split('_')[1])
            child_tree_index = int(attr.split('_')[2])
            if temp_gain_ratio < gain_ratio[child_tree_index]:
                optimalAttrIndex = original_index
                optimalCurAttrIndex = child_tree_index
                temp_gain_ratio = gain_ratio[child_tree_index]

        return optimalAttrIndex, optimalCurAttrIndex

    def CATR(self, D, A):
        """
        使用基尼指数选择最优属性
        """
        # 原始属性集的索引
        optimalAttrIndex = 0
        # 删除某些属性之后的索引
        optimalCurAttrIndex = 0
        # 获取最优属性索引
        gini_index = 9999999
        for cur_index, attr in enumerate(A):
            original_index = int(attr.split('_')[1])
            attrs = np.unique(D[:, original_index])
            gini_value_sum = 0
            for attr in attrs:
                cur_data = self.split_data_set(D, original_index, attr)
                # 计算当前属性在当前取值下的基尼值
                cur_gini_value = self.cal_gini_index(cur_data)
                gini_value_sum += len(cur_data) / len(D) * cur_gini_value
            if gini_value_sum < gini_index:
                gini_index = gini_value_sum
                optimalAttrIndex = original_index
                optimalCurAttrIndex = cur_index
        return optimalAttrIndex, optimalCurAttrIndex

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

    def cal_gini_index(self, D):
        """
        计算基尼指数
        """
        return 1 - np.sum([(D[:, -1].tolist().count(c) / len(D[:, -1])) ** 2 for c in np.unique(D[:, -1])])
