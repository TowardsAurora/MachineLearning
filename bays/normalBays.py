from numpy import *
from math import *

"""
贝叶斯分类准则: 
* 如果 P(c1|x, y) > P(c2|x, y), 那么属于类别 c1; 
* 如果 P(c2|x, y) > P(c1|x, y), 那么属于类别 c2.
p(xy)=p(x|y)p(y)=p(y|x)p(x)
p(x|y)=p(y|x)p(x)/p(y)
"""


# 制造数据集和标签
# 1 stands for abusive  0 not
def createDataSet():
    dataSet = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    classVec = [0, 1, 0, 1, 0, 1]
    return dataSet, classVec


# 由于句子中可能有重复的word，所以借助set来对重复单词进行去重
def createVocabList(dataSet):
    vocabSet = set([])
    for line in dataSet:
        # vocabSet与每一组数据求并集得到不重复的words set
        vocabSet = vocabSet | set(line)
    # 将set集合转为list集合进行存储
    vocabList = list(vocabSet)
    return vocabList


# 判断输入集合中是否存在wordsList中存在的word
# 如果某个单词出现将其在vocabList的位置标记为1，其他则为0
def existWords2Vec(vocabList, inputSet):
    # 初始化existVec为长度为len(vocabList)的0向量
    existVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            existVec[vocabList.index(word)] = 1
        else:
            print("the word: '{}' is not in my vocabList!".format(word))
    return existVec


def trainNormalBays(trainMat, trainCat):
    trainMat = array(trainMat)
    trainCat = array(trainCat)

    numOfData = len(trainMat)
    numOfWords = len(trainMat[0])
    # 计算标记的侮辱性语言在数据集中的占比
    pAbusive = sum(trainCat) / float(numOfData)
    # 构造单词出现次数列表 使用ones()而不是zeros()以免发生math error
    p0NumsList = ones(numOfWords)
    p1NumsList = ones(numOfWords)

    # 整个数据集单词出现总数

    c0Nums = 0.0
    c1Nums = 0.0

    for i in range(numOfData):
        if trainCat[i] == 1:
            p1NumsList += trainMat[i]
            c1Nums += sum(trainMat[i])
        else:
            p0NumsList += trainMat[i]
            c0Nums += sum(trainMat[i])
    p1Vec = p1NumsList / c1Nums
    p0Vec = p0NumsList / c0Nums
    return p0Vec, p1Vec, pAbusive


"""
    使用算法: 
    # 将乘法转换为加法
    乘法: P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
    加法: P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    vecToClassify: 待测数据[0,1,1,1,1...]，即要分类的向量
    p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    pClass1/pAbusive: 类别1，侮辱性文件的出现概率
    return: 类别1 or 0
"""


def classifier(vecToClassify, p0Vec, p1Vec, pAbusive):
    sum1 = 0.0
    sum2 = 0.0
    for i in range(len(vecToClassify)):
        if vecToClassify[i] * p1Vec[i] != 0:
            sum1 += log(vecToClassify[i] * p1Vec[i])
    for i in range(len(vecToClassify)):
        if vecToClassify[i] * p1Vec[i] != 0:
            sum2 += log(vecToClassify[i] * p0Vec[i])
    p1 = sum1 + log(pAbusive)  # P(w|c1) * P(c1)
    p0 = sum2 + log(1 - pAbusive)  # P(w|c0) * P(c0)
    # 还原（可以不用还原，只需要比较结果即可）
    # p1 = exp(p1)
    # p0 = exp(p0)
    # print("p0=", p0, "\np1=", p1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    dataSet, classVec = createDataSet()
    vocab_list = createVocabList(dataSet)
    # print(vocab_list)
    # for test
    # vec = existWords2Vec(vocab_list, dataSet[0])
    # print(vec)
    trainMat = []
    for line in dataSet:
        words_vec = existWords2Vec(vocab_list, line)
        trainMat.append(words_vec)
    p0Vec, p1Vec, pA = trainNormalBays(array(trainMat), array(classVec))
    # print("p0:", p0Vec, "\np1", p1Vec, "pAbusive", pA)
    testEntrySet = ['stupid', 'garbage', 'buying', 'worthless', 'dog']
    testEntrySetVec = existWords2Vec(vocab_list, testEntrySet)
    myClass = classifier(array(testEntrySetVec), p0Vec, p1Vec, pA)
    print(testEntrySet, 'classified as: ', myClass)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(existWords2Vec(vocab_list, testEntry))
    print(testEntry, 'classified as: ', classifier(thisDoc, p0Vec, p1Vec, pA))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(existWords2Vec(vocab_list, testEntry))
    print(testEntry, 'classified as: ', classifier(thisDoc, p0Vec, p1Vec, pA))
