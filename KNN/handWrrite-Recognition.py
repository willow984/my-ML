import numpy as np
import os
import operator


def img2Vector(fileName):
    returnVect = np.zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i*32+j] = int(lineStr[j])
    return returnVect


def classify(X, dataSet, labels, K):
    # 构造测试向量矩阵(n*测试向量)
    dataSetSize = dataSet.shape[0]
    # 距离矩阵(欧式)=sqrt(SUM((真实矩阵-测试矩阵)^2))
    diffMat = dataSet - np.tile(X, (dataSetSize, 1))
    sqDiffMat = diffMat ** 2
    sumOfDiffMat = sqDiffMat.sum(axis=1)        # 1是按行求和返回一个一维数组
    distances = sumOfDiffMat ** 0.5
    sortedDistancesIndex = distances.argsort()  # 从小排序并返回排序后的索引(原索引，索引并未排序)
    # 统计前K个
    KClassCount = {}
    for i in range(K):
        curLabelIndex = sortedDistancesIndex[i]
        curLabel = labels[curLabelIndex]
        KClassCount[curLabel] = KClassCount.get(curLabel, 0) + 1
    sortedKClassCount = sorted(KClassCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedKClassCount[0][0]


def handwritingClassTest(trainingFileDir, testFileDir):
    hwLabels = []  # 用于标记某一行真实值是哪个数字
    # 获取路径下目录(列表)
    trainFileList = os.listdir(trainingFileDir)
    testFileList = os.listdir(testFileDir)
    m = len(trainFileList)
    trainingMat = np.zeros((m, 1024))  # m * 1024的矩阵用于存储m个手写数字

    for i in range(m):
        # 真实标签列表
        fileNameStr = trainFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        # 训练矩阵
        trainingMat[i, :] = img2Vector(trainingFileDir + '/' + fileNameStr)  # 构造训练矩阵

    errorCount = 0.0
    mTest = len(testFileList)
    # 对mTest个测试数据依次进行测试：
    for i in range(mTest):
        testFileNameStr = testFileList[i]
        trueClassNum = int(testFileNameStr.split('_')[0])  # 真实值从test的文件名获取
        testVector = img2Vector(testFileDir + '/' + testFileNameStr)
        classifyResult = classify(testVector, trainingMat, hwLabels, 3) # 经过测试，3最小
        print("result: ", classifyResult, "true value: ", trueClassNum)
        if classifyResult != trueClassNum:
            errorCount += 1.0

    errorRate = errorCount / mTest
    print("Error rate: ", errorRate)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    handwritingClassTest('handWrite_trainingDigits', 'handWrite_testDigits')