import operator
from math import log
import pandas as pd


def calc_dataset_ent(dataSet):
    exampleNum = len(dataSet)
    labelCount = {}
    ent_of_dataset = 0.0
    for example in dataSet:
        label = example[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    for label in labelCount:
        prob = float(labelCount[label]) / float(exampleNum)
        ent_of_label = -prob * log(prob, 2)
        ent_of_dataset += ent_of_label
    return ent_of_dataset

def split_dataset(dataSet, axis, value):
    retDataSet = []
    for FeatVec in dataSet:
        if FeatVec[axis] == value:
            reducedFeatVec = FeatVec[:axis] + FeatVec[axis+1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet

def choose_best_split_feat(dataSet):
    bestGain = 0.0
    bestFeature = -1
    featuresNum = len(dataSet[0]) - 1
    lenOfOriginDataSet = len(dataSet)
    entOfOriginDataset = calc_dataset_ent(dataSet)
    for feature in range(featuresNum):
        entOfNewDataset = 0.0
        featureValueList = []
        # featValueList = [featVec[feature] for featVec in dataSet]
        for featVec in dataSet:
            featureValueList.append(featVec[feature])
        uniqueFeatureValue = set(featureValueList)
        for values in uniqueFeatureValue:
            subDataset = split_dataset(dataSet, feature, values)
            prob = float(len(subDataset)) / float(lenOfOriginDataSet)
            entOfNewDataset += prob * calc_dataset_ent(subDataset)
        Gain = entOfOriginDataset - entOfNewDataset
        if (Gain > bestGain):
            bestGain = Gain
            bestFeature = feature
    return bestFeature

def majority_cnt(labelList):
    labelCount = {}
    for label in labelList:
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    sortedLabelCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabelCount[0][0]

def create_tree(dataSet, labels):
    # 根据标签类书来判断递归的出入口
    # classList = [featVec[-1] for featVec in dataSet]
    classList = []
    for featVec in dataSet:
        classList.append(featVec[-1])

    # 第一种情况：只剩下一类了->直接返回该类
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 第二种情况：特征用完了-> 用多数表决
    if len(dataSet[0]) == 1:
        return majority_cnt(classList)

    # 其余情况：选最佳划分特征->划分树->删除标签->
    # 选特征
    bestFeat = choose_best_split_feat(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 删除已用标签(特征)
    subLabels = labels[:]
    del(subLabels[bestFeat])
    # 建树
    featValues = [featVec[bestFeat] for featVec in dataSet]
    uniqueFeatValues = set(featValues)
    # 递归建树
    for value in uniqueFeatValues:
        myTree[bestFeatLabel][value] = create_tree(split_dataset(dataSet, bestFeat, value), subLabels[:])
    return myTree

def classify(inputTree, featLabels, testVec):
    pass
    # ...


if __name__ == '__main__':
    file_path = './dataset2_weather100.csv'
    data = pd.read_csv(file_path)
    dataSet = data.values.tolist()
    '''
    lenOfFeatVec = len(dataSet[0]) - 1
    reducedDataSet = []
    for featVec in dataSet:
        reducedFeatVec = featVec[:lenOfFeatVec-1]
        reducedDataSet.append(reducedFeatVec)
    print(reducedDataSet)
    '''
    labels = ["Weather", "Temperature", "Humidity", "Wind", ]
    for featVec in dataSet:
        print(featVec)
    myTree = create_tree(dataSet, labels)
    print(myTree)
    # print(dataSet)
