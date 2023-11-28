import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []                        # X矩阵[x0, x1, x2]
    labelMat = []                       # y矩阵[yi]
    fr = open('testSet.txt')
    for line in fr.readlines():
        arrLine = line.strip().split()
        dataMat.append([1.0, float(arrLine[0]), float(arrLine[1])])
        labelMat.append(int(arrLine[2]))
    return dataMat, labelMat            # 返回两个列表

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n,1))        # weights初始为长度为n的列向量
    for k in range(maxCycles):
        z = dataMatrix * weights       # (m,n) * (n,1)
        hz = sigmoid(z)
        # 直接计算求导后的结果
        error = (labelMat - hz)
        weights = weights + alpha * dataMatrix.T * error
    return weights

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    bestWeights = gradAscent(dataArr, labelMat)
    plotBestFit(bestWeights)
