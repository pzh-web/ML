#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/12 19:14
# @Author: peng yang
# @File  : treeGenerate.py
import numpy as np

from ML.xjtupy.ml.util.strOperate import StrOperate

"""
决策树生成
"""


class TreeGenerate(object):

    def __init__(self, train_data, test_data=None, pre_pruning=False):
        self.__train_data = train_data
        self.__test_data = test_data
        self.is_continuity = False  # 是否是连续值
        self.pre_pruning = pre_pruning  # 是否进行预剪枝

        if pre_pruning:
            # 第一次将节点划分为类别最多的一类,为进一步用属性划分提供划分前精度
            for c in np.unique(train_data[:, -1]):
                if len(np.where(train_data[:, -1] == c)[0]) >= (len(train_data) / 2):
                    node_category = c
                    break
            # 验证集预测,划分前的精度
            self.pre_precision = len(np.where(self.__test_data == node_category)[0]) / len(self.__test_data)

    def tree_generate(self, D, A, threshold=0):
        """
        :param D: 数据集 {（x1,y1）,(x2,y2),...,(xm,ym)}
        :param A: 属性集 {a1,a2,...,ad}
        :return: 决策树
        """
        # 第一个返回条件：判断D中样本是不是属于同一类
        if D[:, -1].tolist().count(D[:, -1][0]) == len(D):
            return D[:, -1][0]

        # 第二个返回条件：属性集为空
        if len(A) == 0:
            return self.most_category(D)

        # 选择最优属性:对应原始索引
        optimalAttrIndex, cur_index = self.ID3(D, A)
        # optimalAttrIndex, cur_index = self.C4_5(D, A)
        # 处理了连续属性
        # optimalAttrIndex, cur_index, threshold = self.CATR(D, A)
        # 生成一个节点
        optimalAttr = A[cur_index].split('_')[0]

        ##########################################################
        # 预剪枝
        if self.pre_pruning:
            if self.__pre_pruning(D, optimalAttrIndex) is False:
                return self.most_category(D)
        ##########################################################

        tree = {optimalAttr: {}}
        if self.is_continuity:
            sub_labels = A[:]
            d1 = np.array([row for index, row in enumerate(D) if float(D[index][optimalAttrIndex]) > threshold])
            d2 = np.array([row for index, row in enumerate(D) if float(D[index][optimalAttrIndex]) < threshold])
            if len(d1) == 0:
                tree[optimalAttr]['大于%s' % str(threshold)] = self.most_category(D)
            else:
                tree[optimalAttr]['大于%s' % str(threshold)] = self.tree_generate(d1, sub_labels)
            if len(d2) == 0:
                tree[optimalAttr]['小于%s' % str(threshold)] = self.most_category(D)
            else:
                tree[optimalAttr]['小于%s' % str(threshold)] = self.tree_generate(d2, sub_labels)
        else:
            # 从属性集合中删除最优属性
            A.remove(A[cur_index])
            # 遍历当前属性的所有取值
            for attrValue in np.unique(self.__train_data[:, optimalAttrIndex]):
                # 划分数据
                child_D = self.split_data_set(D, optimalAttrIndex, attrValue)
                # 第三个返回条件：当前属性取值在数据集中没有相应数据
                if len(child_D) == 0:
                    # 将父节点属性集中最多的类别标记为当前节点类别
                    tree[optimalAttr][attrValue] = self.most_category(D)
                else:
                    sub_labels = A[:]
                    # 递归生成决策树
                    tree[optimalAttr][attrValue] = self.tree_generate(child_D, sub_labels)
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
        加入连续属性的划分
        """
        # 原始属性集的索引
        optimalAttrIndex = 0
        # 删除某些属性之后的索引
        optimalCurAttrIndex = 0
        # 划分阈值
        threshold = 0
        # 获取最优属性索引
        gini_index = 9999999
        for cur_index, attr in enumerate(A):
            original_index = int(attr.split('_')[1])
            gini_value_sum = 0
            # 在当前数据集下判断是否是连续属性
            if StrOperate.is_number(D[0][original_index]):  # 连续的
                #  
                D = D[D[:, original_index].argsort()]
                divid_t = []
                cur_attr_value = D[:, original_index].tolist()
                for i in range(len(cur_attr_value) - 1):
                    divid_t.append((float(cur_attr_value[i]) + float(cur_attr_value[i + 1])) / 2)
                for j in range(len(divid_t)):
                    if j > 0 and divid_t[j - 1] == divid_t[j]:
                        continue
                    # 按当前阈值划分数据集
                    d1 = np.array([row for index, row in enumerate(D) if float(D[index][original_index]) > divid_t[j]])
                    d2 = np.array([row for index, row in enumerate(D) if float(D[index][original_index]) < divid_t[j]])
                    # 计算当前属性在当前划分下的基尼值
                    gini_value1 = self.cal_gini_index(d1)
                    gini_value2 = self.cal_gini_index(d2)
                    gini_value_sum = len(d1) / len(D) * gini_value1 + len(d2) / len(D) * gini_value2
                    if gini_value_sum < gini_index:
                        gini_index = gini_value_sum
                        threshold = divid_t[j]
                        optimalAttrIndex = original_index
                        optimalCurAttrIndex = cur_index
                        self.is_continuity = True
            else:  # 离散的
                attrs = np.unique(D[:, original_index])
                for attr in attrs:
                    cur_data = self.split_data_set(D, original_index, attr)
                    # 计算当前属性在当前取值下的基尼值
                    cur_gini_value = self.cal_gini_index(cur_data)
                    gini_value_sum += len(cur_data) / len(D) * cur_gini_value
                if gini_value_sum < gini_index:
                    gini_index = gini_value_sum
                    optimalAttrIndex = original_index
                    optimalCurAttrIndex = cur_index
                    self.is_continuity = False
        return optimalAttrIndex, optimalCurAttrIndex, threshold

    def most_category(self, D):
        """
        :return: 返回数据集中数据量最多的类别
        """
        category = ''
        num = 0
        for c in np.unique(D[:, -1]):
            category = c if D[:, -1].tolist().count(c) >= num else category
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

    def __pre_pruning(self, D, attr_index):
        """
        预剪枝函数
        :param D: 当前划分节点属性集
        :param attr_index: 当前划分属性
        :return: Boolean
        """
        right_count = 0
        # 获取当前属性集的属性值
        attr_values = np.unique(D[:, attr_index])
        # 将属性集按照属性值划分，并标记每个属性值划分集合的类别
        for attr_value in attr_values:
            cur_classification = np.array(
                [row for index, row in enumerate(D) if D[index][attr_index] == attr_value])
            for c in np.unique(D[:, -1]):
                if len(np.where(cur_classification[:, -1] == c)[0]) >= (len(cur_classification) / 2):
                    node_category = c
                    # 验证集预测
                    right_count += len([row for index, row in enumerate(self.__test_data)
                                        if self.__test_data[index][attr_index] == attr_value
                                        and self.__test_data[index][-1] == node_category
                                        ])
                    break
        cur_precision = right_count / len(self.__test_data)
        print('划分前的精度：%s' % self.pre_precision)
        print('划分后的精度：%s' % cur_precision)
        if cur_precision > self.pre_precision:
            self.pre_precision = cur_precision
            return True
        else:
            return False

    def post_pruning(self, tree):
        """
        后剪枝
        :param tree: 生成的决策树
        """
        pass