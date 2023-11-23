import numpy as np
import operator

def createDataSet(fileName):
    with open(fileName) as fr:
        arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    dataSet = np.zeros((numberOfLines, 3))      # 行数，列数
    labels = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()                     # 删除每一行开头结尾的空白(包括tab 换行)
        listFromLine = line.split('\t')         # 行列表(特征向量)
        dataSet[index,:] = listFromLine[0:3]
        labels.append(int(listFromLine[-1]))
        index += 1
    return dataSet, labels

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
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

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - np.tile(minVals, (m, 1))
    normalDataSet = normalDataSet / np.tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = createDataSet('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(norMat[i,:], norMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games? "))
    iceCream = float(input("liters of ice cream consumed per year? "))
    ffMiles = float(input("frequent flier miles earned per year? "))
    datingDataMat, datingLabels = createDataSet('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm((datingDataMat))
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

if __name__ == '__main__':
    datingClassTest()
    classifyPerson()


