#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/12 17:17
# @Author: peng yang
# @File  : visualDecisionTree.py

import matplotlib.pyplot as plt
import matplotlib

"""
绘制树形图
"""


class VisualDecisionTree(object):
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    matplotlib.rcParams['axes.unicode_minus'] = False

    def __init__(self):
        self.decision_node = dict(boxstyle="sawtooth", fc="0.8")
        self.leaf_node = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        self.createPlot = {}
        self.plotTree = {}

    # 获取树的叶子结点个数（确定图的宽度）
    def get_leaf_num(self, tree):
        leaf_num = 0
        first_key = list(tree.keys())[0]
        next_dict = tree[first_key]
        for key in next_dict.keys():
            if type(next_dict[key]).__name__ == "dict":
                leaf_num += self.get_leaf_num(next_dict[key])
            else:
                leaf_num += 1
        return leaf_num

    # 获取数的深度（确定图的高度）
    def get_tree_depth(self, tree):
        depth = 0
        first_key = list(tree.keys())[0]
        next_dict = tree[first_key]
        for key in next_dict.keys():
            if type(next_dict[key]).__name__ == "dict":
                thisdepth = 1 + self.get_tree_depth(next_dict[key])
            else:
                thisdepth = 1
            if thisdepth > depth:
                depth = thisdepth
        return depth

    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.createPlot['ax1'].annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                                        xytext=centerPt, textcoords='axes fraction',
                                        va="center", ha="center", bbox=nodeType, arrowprops=self.arrow_args)

    # 在父子节点间填充文本信息
    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.createPlot['ax1'].text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

    def plot_tree(self, myTree, parentPt, nodeTxt):
        numLeafs = self.get_leaf_num(myTree)
        depth = self.get_tree_depth(myTree)
        firstStr = list(myTree.keys())[0]
        cntrPt = (
            self.plotTree['xOff'] + (1.0 + float(numLeafs)) / 2.0 / self.plotTree['totalW'], self.plotTree['yOff'])
        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, self.decision_node)
        secondDict = myTree[firstStr]
        self.plotTree['yOff'] = self.plotTree['yOff'] - 1.0 / self.plotTree['totalD']
        for key in secondDict.keys():
            if type(secondDict[
                        key]).__name__ == 'dict':
                self.plot_tree(secondDict[key], cntrPt, str(key))
            else:
                self.plotTree['xOff'] = self.plotTree['xOff'] + 1.0 / self.plotTree['totalW']
                self.plotNode(secondDict[key], (self.plotTree['xOff'], self.plotTree['yOff']), cntrPt, self.leaf_node)
                self.plotMidText((self.plotTree['xOff'], self.plotTree['yOff']), cntrPt, str(key))
        self.plotTree['yOff'] = self.plotTree['yOff'] + 1.0 / self.plotTree['totalD']

    def create_plot(self, inTree, name):
        fig = plt.figure(name, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.createPlot['ax1'] = plt.subplot(111, frameon=False, **axprops)
        self.plotTree['totalW'] = float(self.get_leaf_num(inTree))
        self.plotTree['totalD'] = float(self.get_tree_depth(inTree))
        self.plotTree['xOff'] = -0.5 / self.plotTree['totalW']
        self.plotTree['yOff'] = 1.0
        self.plot_tree(inTree, (0.5, 1.0), '')
        plt.show()
