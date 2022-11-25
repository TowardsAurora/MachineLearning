import numpy as np
from matplotlib import pyplot as plt


def create_data(data):
    X = []
    tag = []
    for data_i in data:
        X.append(data_i[0])
        tag.append(data_i[1])
    X = np.array(X)
    tag = np.array(tag)
    return X, tag


def split_data(data):
    positive = []
    negative = []
    for data_i in data:
        if data_i[1] == 1:
            positive.append(data_i)
        else:
            negative.append(data_i)
    return positive, negative


class solution:
    def __init__(self):
        self.W = np.array([0, 0])
        self.bias = 0
        self.learning_rate = 0.1

    def update_params(self, W, X, bias, tag):
        self.W = W + self.learning_rate * X * tag
        self.bias = bias + self.learning_rate * tag

    @staticmethod
    def calculate_output(W, X, bias):
        output = np.sign(np.dot(W, X) + bias)
        return output

    def train(self, X, tag):
        stop = True
        while stop:
            count = len(X)
            for i in range(len(X)):
                if self.calculate_output(self.W, X[i], self.bias) * tag[i] <= 0:
                    print("当前是第{}个数据点 {} 分类错误".format(i+1,X[i]))
                    self.update_params(self.W, X[i], self.bias, tag[i])
                else:
                    count -= 1
                if count == 0:
                    stop = False
        print(" W = {} , bias = {}".format(self.W, self.bias))
        return self.W, self.bias

    def test(self, X, tag):
        tes = []
        for i in range(len(X)):
            te = self.calculate_output(self.W, X[i], self.bias) * tag[i] > 0
            tes.append(te)
        return tes


if __name__ == '__main__':
    data = [
        ((1, 1), 1),
        ((0, 1), -1),
        ((1, 0), -1),
        ((0, 0), -1)
    ]
    X, tag = create_data(data)
    model = solution()
    W, bias = model.train(X, tag)
    test_as = model.test(X, tag)
    positive, negative = split_data(data)
    positiveX, tag_p = create_data(positive)
    negativeX, tag_n = create_data(negative)
    print(positiveX,negativeX)
    # tag = 1
    plt.scatter(positiveX[:, 0], positiveX[:, 1], marker='.', color='red', s=100)
    # tag = -1
    plt.scatter(negativeX[:, 0], negativeX[:, 1], marker='.', color='blue', s=100)
    plt.xlabel("x1")
    plt.ylabel("x2")
    """
    w1*x1+w2*x2+bias=0
    x2=(-bias-w1*x1)/w2
    """
    x1 = np.linspace(-5, 5, 50)
    x2 = (-bias - W[0] * x1) / W[1]
    plt.plot(x1, x2)
    # 当
    print("{}*x1+{}*x2+{}=0".format(W[0], W[1], bias))
    print(test_as)
    plt.show()
