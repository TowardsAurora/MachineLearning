from math import *

from matplotlib import pyplot as plt
from numpy import *
import numpy as np


# 创建数据集
def createDataSet(fileName):
    dataMat = []
    labelMat = []
    f = open(fileName)
    for line in f.readlines():
        # print(line.strip().split(" "))
        lineList = line.strip().split(" ")
        # label = sigmoid(w0+w1*x1+w2*x2)
        dataMat.append([float(lineList[0]), float(lineList[1])])
        labelMat.append(int(lineList[-1]))
    return dataMat, labelMat


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + exp(-x))


def plotOrigin(dataMat, labelMat):
    x1, x2 = [], []
    for item in dataMat:
        x1.append(item[0])
        x2.append(item[1])
    plt.scatter(x1, x2, c=labelMat)

    # 设置图表标题和坐标轴标签
    plt.title("Scatter Plot", fontsize=24)
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)

    # 显示图形
    plt.show()


# 梯度下降法
def gradAscent(dataMat, labelMat, alpha=0.001, maxCycles=500):
    data = []
    for item in dataMat:
        data.append([1.0, item[0], item[1]])
    # print(data)
    dataMat = mat(data)
    labelMat = mat(labelMat).transpose()  # 将labelMat从行向量转为列向量
    m, n = shape(dataMat)  # 得到dataMat的行和列数，即m-->数据量数，n-->因子数
    # print(m, n)

    # alpha = 0.001  # 向目标移动的步长
    # maxCycles = 500  # 最大迭代次数

    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和因子数相同的矩阵，其中的数全部都是 1
    weights = ones((n, 1))
    for eachRound in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
    return array(weights)


def plotMyLine(dataMat, labelMat, weights):
    x1, x2 = [], []
    for item in dataMat:
        x1.append(item[0])
        x2.append(item[1])
    plt.scatter(x1, x2, c=labelMat)

    # 设置图表标题和坐标轴标签
    plt.title("Scatter Plot", fontsize=24)
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)

    x = np.linspace(-3, 3, 50)
    y = -(weights[0] + weights[1] * x) / weights[2]
    plt.plot(x, y)
    plt.show()


def classifier(dataLine, weights):
    print("sum:", sum(dataLine * weights))
    prob = sigmoid(sum(dataLine * weights))
    print("prob:", prob)
    if prob >= 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    dataMat, labelMat = createDataSet('./data.txt')
    # print(dataMat)
    # print("\n", labelMat)

    plotOrigin(dataMat, labelMat)

    ascent = gradAscent(dataMat, labelMat, 0.0001, 1000)
    print(ascent)
    print(shape(ascent))
    transpose_ascent = mat([[1.0], [-0.017612], [14.053064]]).transpose() * ascent
    print("sigmoid",sigmoid(transpose_ascent))
    plotMyLine(dataMat, labelMat, ascent)
    transpose = mat([[1.0], [-0.017612], [14.053064]]).transpose()
    myClass = classifier(transpose, weights=ascent)
    print(myClass)

    print("sigmoid",sigmoid(mat([[1.0], [2], [0]]).transpose() * ascent))
    myClass = classifier(mat([[1.0], [2], [0]]).transpose(), weights=ascent)
    print(myClass)

