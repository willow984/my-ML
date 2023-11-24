import numpy as np

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
    vocabNum = len(vocabList)
    returnVec = np.zeros(vocabNum)                            # 第i个位置上为1表示vocabList中第i个词出现了
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("这个单词%s不在词汇表中" % word)
    return returnVec


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print("\n")
    print(setOfWords2Vec(myVocabList, listOPosts[0]))