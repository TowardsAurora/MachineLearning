from math import *


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calc_Shanon_Ent(dataSet):
    data_length = len(dataSet)
    labels_count = {}

    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labels_count:
            labels_count[curLabel] = 1
        else:
            labels_count[curLabel] += 1

    shanonEnt = 0.0
    # Ent = ∑ -p*log2 p
    for key in labels_count:
        value = labels_count.get(key)
        prob = value / data_length
        shanonEnt += (-prob) * log(prob, 2)
    return shanonEnt


"""
根据属性进行划分
当前根据 数据的第 index 个属性进行划分，当 data[index] == value 时
--->value为set()集合中的值，即第 index 个属性所拥有的值
reducedFeaVec extend 排除当前属性的值
再将其加入subDataSet得到划分矩阵
实际上针对该划分在后面计算的有效属性为 len(subDataSet)
"""


def splitData(dataSet, index, value):
    subDataSet = []
    for data in dataSet:
        if data[index] == value:
            reducedFeaVec = data[:index]
            reducedFeaVec.extend(data[index:])
            subDataSet.append(reducedFeaVec)
    return subDataSet


"""
选择最佳划分属性
得到基础的信息熵，计算每一个属性划分后的信息熵，得到每一个属性的信息增益
进行比较，选择信息增益最大的属性作为划分属性
"""


def chooseBestFeature(dataSet):
    # 得到影响决策的属性个数
    featureCount = len(dataSet[0]) - 1
    # 得到初始信息熵
    baseEnt = calc_Shanon_Ent(dataSet)
    bestInfoGainRatio = 0.0
    # 将初始最佳属性设为-1
    bestFeature = -1
    for i in range(featureCount):
        featureList = [feature[i] for feature in dataSet]
        uniqueValOfFeature = set(featureList)
        divide_feature_Ent = 0.0
        IV_Ent = 0.0
        # 计算根据某个属性划分后的信息熵
        for value in uniqueValOfFeature:
            subDataSet = splitData(dataSet, i, value)
            divide_prob = len(subDataSet) / len(dataSet)
            divide_feature_Ent += divide_prob * calc_Shanon_Ent(subDataSet)
            IV_Ent += (-divide_prob) * log(divide_prob, 2)
        # 得到信息增益
        infoGain = baseEnt - divide_feature_Ent
        # print(infoGain, "  ", IV_Ent)
        # print("当前划分属性信息熵:", divide_feature_Ent, "feature:", i, "当前划分属性信息增益：", infoGain)
        # 计算信息增益率
        if IV_Ent == 0.0:
            break
        infoGainRatio = infoGain / IV_Ent
        print("当前划分属性信息熵:", divide_feature_Ent, "当前划分依据:", labels[i], "当前划分属性信息增益率：{:2f}%".format(infoGainRatio*100))
        # 将信息增益与最佳信息增益进行对比，进行更新
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature


def majorCounts(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList:
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[0], reverse=True)
    print("sortedClassCount:", sortedClassCount)
    print("sortedClassCount[0][0]:", sortedClassCount[0][0])
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [data[-1] for data in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return
    best_feature = chooseBestFeature(dataSet)
    # print("createTree in bestFea index:",best_feature)
    # print(len(labels))
    best_feature_Label = labels[best_feature]

    myTree = {best_feature_Label: {}}
    # 根据最佳属性划分结束后删除最佳划分属性
    # del (labels[best_feature]) # 这行没必要

    # 取出最佳划分属性列，根据其branch(分支)做分类
    featureVal = [data[best_feature] for data in dataSet]
    uniqueVal = set(featureVal)
    for value in uniqueVal:
        # 取出剩余标签
        subLabels = labels[:]
        # 进行递归处理
        # in the tree
        # key = best_feature_Label , value = value
        myTree[best_feature_Label][value] = createTree(splitData(dataSet, best_feature, value), subLabels)
    return myTree


"""
Args:
    inputTree  决策树模型
    featureLabel Feature标签对应的名称
    testVec    测试输入的数据
Returns:
    classLabel 分类的结果值，需要映射label才能知道名称
"""


def classify(inputTree, featureLabel, testVec):
    # 获取树的根结点的值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到节点值
    secDict = inputTree.get(firstStr)
    # print(featureLabel)
    # print(firstStr)
    featureIndex = featureLabel.index(firstStr)
    # print("feaIndex",featureIndex)
    key = testVec[featureIndex]
    value = secDict.get(key)
    # print('firstStr:', firstStr, 'secDict:', secDict, 'key:', key, 'value:', value)
    # 判断得到的value是否还有branch
    if isinstance(value, dict):
        classLabel = classify(value, featureLabel, testVec)
    else:
        classLabel = value
    return classLabel


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    feature = chooseBestFeature(dataSet)
    # print(feature)
    tree = createTree(dataSet, labels)
    print(tree)
    print("DivisionClass:",classify(tree, labels, [0, 0]))
    from plot import createPlot
    createPlot(tree)