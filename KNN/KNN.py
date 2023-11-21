import numpy as np
import pandas as pd
import operator


def createDataSet(fileName):
    dataSet = pd.read_csv(fileName).iloc[:, 1:].values.tolist()
    labels = pd.read_csv(fileName).iloc[:, 0].values.tolist()
    return dataSet, labels

def classify(inX, dataSet, labels, k):
    # inX是输入的要进行K临近分类的特征向量
    dataSetSize = len(dataSet)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet   # 距离矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)                  # axis=0是对列求和，axis=1是对行求和， 距离矩阵每行平方和组成的列表
    distances = sqDistances ** 0.5                       # 开方(欧氏距离)
    sortedDistIndicies = distances.argsort()             # 返回数组元素从小到大排序后的索引数组(即K临近)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    dataSet, labels = createDataSet('dataSet1_NBA_player.csv')
    testX = [195, 90, 85, 80, 80, 80]
    position = classify(testX, dataSet, labels, 10)
    # print(dataSet)
    # print(labels)
    print(position)
