import numpy as np
from numpy import log as log

'''
独立事件P(A), P(B)
条件概率：在B发生情况下A发生的概率
    P(A|B) = P(AB) / P(B)
贝叶斯公式：已知P(A|B)求P(B|A)
    P(B|A) = P(A|B) * P(B) / P(A)
'''

def loadDataSet():
    # 六段文本转换成单词向量
    postingList=[
                 ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]   # 标记前面六段话是否为侮辱(0为正常，1为侮辱)
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)                     # 取并集
    return list(vocabSet)                                       # 返回了包含dataSet中所有不重复词汇的列表

def setOfWords2Vec(vocabList, inputSet):
    # 不考虑词语出现次数，只考虑是否出现过(伯努利朴素贝叶斯)
    vocabNum = len(vocabList)
    returnVec = np.zeros(vocabNum)                            # 第i个位置上为1表示vocabList中第i个词出现了
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("这个单词%s不在词汇表中" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    # 考虑每个词出现的次数(多项式朴素贝叶斯)
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1
        return returnVec

def trainNB0(trainMatrix, trainCategory):
    # 朴素贝叶斯分类器训练函数
    # trainMatrix:是每句话基于整个词汇表的词向量，形如[1,0,1,0,...]
    numTrainDocs = len(trainMatrix)                             # 句子数量
    numWords = len(trainMatrix[0])                              # 特征数量，即每句话的词向量长度(词库不同词数)
    pAbusive = sum(trainCategory)/float(numTrainDocs)           # 侮辱性句子的概率，trainCategory中1表示侮辱性句子
    p0Num = np.ones(numWords)                                  # 用来累加所有正常句子词向量(赋为1防止乘以0)
    p1Num = np.ones(numWords)                                  # 用来累加所有侮辱句子词向量
    p0Denom = 2.0                                               # 正常句子的词语总数(由于p0p1被赋值为1，则这个值至少为2)
    p1Denom = 2.0                                               # 侮辱句子的词语总数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:       # 如果是侮辱性
            p1Num += trainMatrix[i]     # 这个侮辱句子的词向量累加进p1Num,p1Num就可以统计所有侮辱性句子所有词语出现的次数
            p1Denom += sum(trainMatrix[i])      # 侮辱性句子词语总数
        else:
            p0Num += trainMatrix[i]     # 这个正常句子的词向量累加进p0Num,p0Num就可以统计所有正常性句子所有词语出现的次数
            p0Denom += sum(trainMatrix[i])      # 正常句子词语总数
    p1Vect = log(p1Num/p1Denom)            # 计算每个词出现在侮辱性句子中的概率
    p0Vect = log(p0Num/p0Denom)            # 计算每个词出现在正常性句子中的概率
    return p0Vect, p1Vect, pAbusive     # 返回每个词出现在正常性句子中的概率、每个词出现在侮辱性句子中的概率、侮辱性句子的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 朴素贝叶斯分类函数
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)           # 生成词典
    trainMat = []                                       # 词典中每个句子的词向量
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))     # 以词典训练分类器
    testEntry = ['love', 'my', 'stupid', 'dog']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'love', 'help']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    testingNB()


