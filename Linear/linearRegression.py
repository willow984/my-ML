import numpy
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    # 将以tab分割的数据文件转化成特征矩阵X和目标值向量Y
    numFeat = len(open(fileName).readline().split(',')) - 1            # 默认数据文件每一行最后一个数据是目标值y
    dataMat = []        # 数据(特征矩阵)
    labelMat = []       # y向量
    fr = open(fileName)
    for line in fr.readlines():
        lineList = []                           # 存储每一行数据的特征向量
        curLine = line.strip().split(',')      # curLine是一个字符串列表
        for i in range(numFeat):
            lineList.append(float(curLine[i]))
        dataMat.append(lineList)                # 将每个特征向量加入特征矩阵
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat                    # 返回X(二维列表)和Y(列表)

def standRegress(xArr, yArr):
    # xArr: X的二维列表
    # yArr: Y的一维列表
    XMat = mat(xArr)
    YMat = mat(yArr).T
    xTx = XMat.T * XMat
    if linalg.det(xTx) == 0.0:                  # 求行列式
        print("X不可逆")
        return
    ws = xTx.I * (XMat.T * YMat)
    return ws

def calCost(ws, y):
    vecNum = len(ws)
    cost = 0
    for i in range(vecNum):
        cost += (ws[i] - y[i]) ** 2
    return cost


if __name__ == '__main__':
    XArr, YArr = loadDataSet('dataset1_3features_100.csv')
    # print("X:")
    # for line in XArr:
        # print(line)
    # print("Y:")

    ws = standRegress(XArr, YArr)
    vecNum = len(XArr[0])
    calY = numpy.dot(XArr, ws)
    print(calY)
    cost = 0.0
    for i in range(len(YArr)):
        cost += (calY[i] - YArr[i]) ** 2
    print(cost)

    xMat = mat(XArr)
    yMat = mat(YArr)
    yHat = xMat * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1], yHat)

    plt.show()
