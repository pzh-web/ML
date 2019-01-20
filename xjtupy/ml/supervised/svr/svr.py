#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/20 15:45
# @Author: Vincent
# @File  : svr.py

from numpy import *

"""
GitHub上的代码：https://github.com/Drzgy/epsilon-svr
没看懂
"""
def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue


def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    return row_stack((column_stack((kernelMatrix, -kernelMatrix)), column_stack((-kernelMatrix, kernelMatrix))))


class SvrStruct:
    def __init__(self, dataSet, labels, C, epsilon, toler, kernelOption):
        self.X = dataSet
        self.Y = labels
        self.epsilon = epsilon  # 不敏感损失
        self.toler = toler  # 满足迭代结束的精度
        self.numSamples = dataSet.shape[0]  # 样本数量
        self.C = mat(C * ones([self.numSamples * 2, 1]))  # 惩罚系数
        self.alphas = mat(zeros([self.numSamples * 2, 1]))  # 拉格朗日乘子 α=[α,α*]
        self.z = row_stack((ones([self.numSamples, 1]), -1 * ones([self.numSamples, 1])))  # z=1 α=α,z=-1 α=α*
        self.G = row_stack((self.epsilon * ones([self.numSamples, 1]) - self.Y,
                            self.epsilon * ones([self.numSamples, 1]) + self.Y))  # 梯度
        self.b = row_stack((self.epsilon * ones([self.numSamples, 1]) - self.Y,
                            self.epsilon * ones([self.numSamples, 1]) + self.Y))  # F=α.T*Q*α+p*α b->p
        self.B = 0  # f(X)=α*k(x,xi)+b B->b
        self.alphasState = mat(-1 * ones((self.alphas.shape[0],
                                          1)))  # 当αi等于惩罚参数C时，alpha_status[i]等于1；当αi=0时，alpha_status[i]等于-1；当0<αi<C时，alpha_status[i]等于0。
        self.kernelOpt = kernelOption  # 核函数的设置
        self.kernelMat = calcKernelMatrix(self.X, self.kernelOpt)  # 核矩阵 kernelMat=[K,-K;-K，K]
        self.obj = 1e5  # F=α.T*Q*α+p*α 目标函数值


class SvrResult:  # 输出结果
    def __init__(self, svr):
        self.X = svr.X
        self.C = svr.C
        self.B = svr.B
        self.eplison = svr.epsilon
        self.toler = svr.toler
        self.numSamples = svr.Y.shape[0]
        self.alphas = svr.alphas[0:self.numSamples] - svr.alphas[self.numSamples: 2 * self.numSamples]
        self.kernel = svr.kernelMat
        self.kernelOpt = svr.kernelOpt
        self.obj = svr.obj


def isUpBound(svr, i):
    if svr.alphasState[i] == 1:
        return True
    else:
        return False


def isLowBound(svr, i):
    if svr.alphasState[i] == -1:
        return True
    else:
        return False


def updateAlphasState(svr, i):
    if svr.alphas[i] == svr.C[i]:
        svr.alphasState[i][0] = 1
    elif svr.alphas[i] == 0:
        svr.alphasState[i][0] = -1
    else:
        svr.alphasState[i][0] = 0


def selectAlphaIJ(svr):
    mi = -1
    mj = -1
    Pm1 = -1e10
    Pm2 = -1e10
    for i in range(svr.alphas.shape[0]):
        if svr.z[i] > 0:
            if (not isUpBound(svr, i)) & (-svr.G[i] > Pm1):
                Pm1 = -svr.G[i]
                mi = i
            if (not isLowBound(svr, i)) & (svr.G[i] > Pm2):
                Pm2 = svr.G[i]
                mj = i
        else:
            if (not isUpBound(svr, i)) & (-svr.G[i] > Pm2):
                Pm2 = -svr.G[i]
                mj = i
            if (not isLowBound(svr, i)) & (svr.G[i] > Pm1):
                Pm1 = svr.G[i]
                mi = i
    print('Pm1', Pm1)
    print('Pm2', Pm2)
    return mi, mj, (Pm1 + Pm2) < svr.toler


def calcB(svr):
    free = 0
    freeSV = 0
    bUp = 0
    bLow = 0
    for i in range(svr.alphas.shape[0]):
        zp = svr.z[i] * svr.G[i]
        if isLowBound(svr, i):
            if svr.z[i] > 0:
                bUp = min(bUp, zp)
            else:
                bLow = max(bLow, zp)
        elif isUpBound(svr, i):
            if svr.z[i] < 0:
                bUp = min(bUp, zp)
            else:
                bLow = max(bLow, zp)
        else:
            free += 1
            freeSV += zp
    if free > 0:
        svr.B = freeSV / free
    else:
        svr.B = (bUp + bLow) * 0.5


def SvrTrain(train_x, train_y, C, epsilon, toler, maxIter, kernelOption=('rbf', 1.0)):
    svr = SvrStruct(mat(train_x), mat(train_y), C, epsilon, toler, kernelOption)
    iter = 0
    print("初始化梯度")
    for i in range(svr.alphas.shape[0]):
        if not isLowBound(svr, i):
            for j in range(svr.alphas.shape[0]):
                svr.G[j] += svr.alphas[i] * svr.kernelMat[i, j]
    print("开始优化")
    print('tol', svr.toler)
    while True:
        iter += 1
        i, j, stop = selectAlphaIJ(svr)

        if (stop) | ((iter > maxIter) & (maxIter != -1)):
            print('tol', svr.toler)
            print('优化结束', stop)
            break
        print('第', iter, '次迭代，选择工作集：', i, j)
        alphaOldI = svr.alphas[i].copy()
        alphaOldJ = svr.alphas[j].copy()
        if svr.z[i] != svr.z[j]:
            eta = 2 * svr.kernelMat[i, j] + svr.kernelMat[i, i] + svr.kernelMat[j, j]
            d = (-svr.G[i] - svr.G[j]) / max(abs(eta), 1e-10)
            diff = svr.alphas[i] - svr.alphas[j]
            svr.alphas[i] += d
            svr.alphas[j] += d
            if (diff > 0) & (svr.alphas[j] < 0):
                svr.alphas[j] = 0
                svr.alphas[i] = diff
            elif (diff < 0) & (svr.alphas[i] < 0):
                svr.alphas[i] = 0
                svr.alphas[j] = -diff
            if (diff > svr.C[i] - svr.C[j]) & (svr.alphas[i] > svr.C[i]):
                svr.alphas[i] = svr.C[i]
                svr.alphas[j] = svr.C[i] - diff
            elif (diff <= svr.C[i] - svr.C[j]) & (svr.alphas[j] > svr.C[j]):
                svr.alphas[j] = svr.C[j]
                svr.alphas[i] = svr.C[j] - diff
        else:
            eta = -2 * svr.kernelMat[i, j] + svr.kernelMat[i, i] + svr.kernelMat[j, j]
            d = (svr.G[i] - svr.G[j]) / max(abs(eta), 1e-10)
            summ = svr.alphas[i] + svr.alphas[j]
            svr.alphas[i] -= d
            svr.alphas[j] += d
            if (summ > svr.C[i]) & (svr.alphas[i] > svr.C[i]):
                svr.alphas[i] = svr.C[i]
                svr.alphas[j] = summ - svr.C[i]
            elif (summ <= svr.C[i]) & (svr.alphas[j] < 0):
                svr.alphas[i] = summ
                svr.alphas[j] = 0
            if (summ > svr.C[j]) & (svr.alphas[j] > svr.C[j]):
                svr.alphas[i] = summ - svr.C[j]
                svr.alphas[j] = svr.C[j]
            elif (summ <= svr.C[j]) & (svr.alphas[i] < 0):
                svr.alphas[j] = summ
                svr.alphas[i] = 0

        updateAlphasState(svr, i)
        updateAlphasState(svr, j)
        print('alphaOldI:', alphaOldI, 'alphaI:', svr.alphas[i])
        print('alphaOldJ:', alphaOldJ, 'alphaJ:', svr.alphas[j])
        di = svr.alphas[i] - alphaOldI
        dj = svr.alphas[j] - alphaOldJ
        print('di:', di, '  dj:', dj)
        svr.G += svr.kernelMat[i, :].T * di + svr.kernelMat[j, :].T * dj

    calcB(svr)
    svr.obj = (svr.alphas.T * (svr.G + svr.b)) * 0.5
    print('obj:', svr.obj)

    print('nSV', svr.numSamples - svr.alphasState.tolist().count([1]))
    print('nBSV', svr.alphasState.tolist().count([0]))
    return SvrResult(svr)


def predict(svr, testX, testY=None):
    testX = mat(testX)
    testY = mat(testY)
    numTestSamples = testX.shape[0]
    supportVectorsIndex = nonzero(svr.alphas.A != 0)[0]
    supportVectors = svr.X[supportVectorsIndex]
    supportVectorAlphas = svr.alphas[supportVectorsIndex]
    matchCount = 0
    result = []
    for i in range(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, testX[i, :], svr.kernelOpt)
        predict = float(sum(kernelValue.T * supportVectorAlphas) - svr.B)
        matchCount += abs(predict - testY[i])
        result.append(predict)
    mse = float(matchCount) / numTestSamples
    return result, mse
