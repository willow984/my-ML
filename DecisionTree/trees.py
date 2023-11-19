from math import log
import operator


def createDataSet():
    """创建一个固定的数据集(也可以自定义)"""
    dataSet = [
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']           # 不浮出水面是否可以生存、 是否有脚蹼
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算当前给定数据集dataSet的信息熵：
        1.熵表示“混乱程度”，“信息量”，熵越大表示不确定性越大
        2.某个数据集的信息熵就是其中所有类(标签)的熵之和
        3.每个类(标签)的熵=prob*log(prob,2)，prob是这一类占总样本的比例
    """
    numEntries = len(dataSet)                       # 样本个数
    labelCount = {}                                 # 用一个空字典记录标签数量
    for featVec in dataSet:
        currentLabel = featVec[-1]                  # 取特征向量最后一列(标签)
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0            # 字典中不存在的标签加入字典
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries  # 计算当前数据集的熵
        shannonEnt -= prob * log(prob, 2)           # 概率必<=1，所以log出的为负值
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    划分数据集是以一个特征的某个特定值进行划分：
        dataSet: 要划分的数据集(二维列表/特征矩阵)
        axis: 用来划分的某个特征
        value: 用来划分的某个特征的指定值
        1.从数据集中选出所有axis特征为value的样本(特征向量)
        2.对所有的这些样本，将其特征向量从axis特征切开，取其前后两部分
        3.将这两部分拼起来，成为了没有axis这一列的新特征矩阵
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:                      # 取axis特征为value的样本
            reducedFeatVec = featVec[:axis]             # 取axis之前(不含axis)
            reducedFeatVec.extend(featVec[axis+1:])     # 取axis之后(不含axis)
            retDataSet.append(reducedFeatVec)           # 拼接
    return retDataSet


# 选择最优的数据集分割方式(信息增益)
def chooseBestFeatureToSplit(dataSet):
    """
    选择最优的划分特征(即按哪个特征划分使得数据集的熵更小)：
        1.计算原数据集的熵
        2.计算数据集按照每一个特征划分后的新数据集的熵
            按照某个特征划分后的新数据集的熵newEntropy += prob(特征下每个取值的概率) * calcShannonEnt(subDataSet)
        3.找到使得划分后数据集增益最大(熵最小)的特征
    """
    baseEntropy = calcShannonEnt(dataSet)                          # 数据集原来的熵
    bestInfoGain = 0.0                                             # 最大信息增益
    bestFeature = -1                                               # 最优的划分特征
    numFeatures = len(dataSet[0]) - 1                              # 特征的数量
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]             # 取每一个样本取其第i个特征值
        uniqueVals = set(featList)                                 # set()将列表转化为集合(去重复)，即得到第i个特征有多少种取值
        newEntropy = 0.0                                           # 划分后数据集的熵
        for value in uniqueVals:                                   # 将数据集按照第i个特征的每一种取值划分
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)        # 按照第i个特征的每一种取值划分后的熵之和，即为数据集按这种特征划分的熵
        infoGain = baseEntropy - newEntropy                        # 信息增益即为原数据集的熵和新数据集的熵之差
        if (infoGain > bestInfoGain):                              # 找出使得信息增益最大的特征(标号)
            bestInfoGain = i
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    多数表决法：当决策树划按所有特征划分后剩下的还不是同一类时(即最后一个叶子类不纯)：
        1.统计所有类及其个数
        2.按其个数进行排序，找出最多的那一类，把剩下的这些都划为这一类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树(决策树分类的过程就是递归的找最优划分来划分数据集的过程，本质上是个算法，没有建所谓的”树“)：
        labels:类别(标签)列表

        1.当前输入子集若全都是一类->则划分结束
        2.当前输入的子集并不全是一类，但所有特征已经划分完->用多数表决法决定
        3.当前输入的子集既不全是一类，特征也没有划分完->寻找最优特征再划分
    """

    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        print("只剩一类")
        return classList[0]

    if len(dataSet[0]) == 1:
        print("特征已用完")
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)                # 选出一个最佳划分特征
    bestFeatLabel = labels[bestFeat]
    print("这次的最佳划分特征是:", bestFeatLabel)
    myTree = {bestFeatLabel: {}}                                # 以最佳的特征划分树：形式是{ feat1:{ feat2:{ feat3:{ } }

    subLabels = labels[:]  # 创建标签的副本
    del (subLabels[bestFeat])  # 在副本上删除已选择的特征

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels[:])
    return myTree


if __name__ == '__main__':
    print('1')
    dateSet, labels = createDataSet()
    print(dateSet, labels)
    fishclass = createTree(dateSet, labels)
    print(fishclass)