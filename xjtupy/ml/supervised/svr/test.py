from numpy import *
from sklearn.datasets import load_boston
from sklearn import svm, model_selection

from ML.xjtupy.ml.supervised.svr.svr import SvrTrain, predict

if __name__ == '__main__':
    boston = load_boston()
    dataSet = mat(boston.data)
    labels = mat(boston.target).T
    train_x, test_x, train_y, test_y = model_selection.train_test_split(dataSet, labels, test_size=0.2, random_state=1)

    C = 1e1
    epsilon = 1
    toler = 1e-3
    maxIter = -1
    svr = SvrTrain(train_x, train_y, C, epsilon, toler, maxIter, kernelOption=('rbf', 1))
    yhat, loss = predict(svr, train_x, train_y)
    print('loss', loss)
    test = array(train_y.T)[0]
    print('svrDemo相关系数矩阵', corrcoef([test, yhat]))

    esvr = svm.SVR(C=C, epsilon=epsilon, tol=toler, kernel='rbf', verbose=1)
    his = esvr.fit(train_x, train_y)
    yhat = esvr.predict(train_x)
    test = array(train_y.T)[0]
    l = sum([(a - b) * (a - b) for a, b in zip(test, yhat)])
    print('sklearn相关系数矩阵', corrcoef([test, yhat]))
