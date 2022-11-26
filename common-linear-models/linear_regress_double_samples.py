import numpy as np
from matplotlib import pyplot as plt

"""
这里我们讨论二元情况
即: y_pre=kx+b
target: minimize sum (y - y_pre)^2
min sum (y-(kx+b))^2
min sum (y^2-2*y*(kx+b)+(kx+b)^2)
分别对 k,b求导 令其为0
Partial y/k: k*(x^2).avg =b*x.avg-(y*x).avg 
Partial y/b: b=k*x.avg-y
solve:
k=((x*y).avg-x.avg*y.avg)
b=k*x.avg-y
"""


def spilt_data(data, spilt_point):
    train_data = data[0:spilt_point]
    test_data = data[spilt_point:len(data)]
    return train_data, test_data


def create_data(data):
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y


class linear_regression():
    def __init__(self):
        self.k = None
        self.b = None

    @staticmethod
    def calculate_avg_x(X):
        sum = 0
        for x in X:
            sum += x
        avg_x = sum / (len(X))
        return avg_x

    @staticmethod
    def calculate_avg_y(Y):
        sum = 0
        for y in Y:
            sum += y
        avg_y = sum / (len(Y))
        return avg_y

    @staticmethod
    def calculate_avg_xx(X):
        sum = 0
        for x in X:
            sum += x * x
        avg_xx = sum / (len(X))
        return avg_xx

    @staticmethod
    def calculate_avg_xy(X, Y):
        sum = 0
        for x, y in zip(X, Y):
            sum += x * y
        avg_xy = sum / (len(X))
        return avg_xy

    def train(self, X, Y):
        avg_xy = self.calculate_avg_xy(X, Y)
        avg_xx = self.calculate_avg_xx(X)
        avg_x = self.calculate_avg_x(X)
        avg_y = self.calculate_avg_y(Y)
        self.k = (avg_xy - avg_x * avg_y) / (avg_xx - avg_x * avg_x)
        self.b = avg_y - self.k * avg_x
        return self.k, self.b

    def predict_y(self, X):
        Y = []
        for x in X:
            y = k * x + b
            Y.append(y)
        return Y

    def test(self, X, Y, bias):
        count = 0
        predict_y = self.predict_y(X)
        for y, y_pre in zip(Y, predict_y):
            if abs(y - y_pre) < bias:
                count += 1
        acc = count / len(X)
        return acc


if __name__ == '__main__':
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 9],
        [4, 11],
        [5, 14.5],
        [6, 18.3],
        [7, 21.2],
        [8, 25],
        [9, 26.6],
        [10, 31]
    ])
    model = linear_regression()
    length = len(data)
    X, Y = create_data(data)
    print("data length:", length)
    spilt_point = 7
    train_data, test_data = spilt_data(data, spilt_point)
    train_X, train_Y = create_data(train_data)
    print("train:", train_X, train_Y)
    test_X, test_Y = create_data(test_data)
    print("test:", test_X, test_Y)
    k, b = model.train(train_X, train_Y)
    model_predict_y = model.predict_y(test_X)
    print("predict in test :", model_predict_y)
    bias = 1
    accuracy = model.test(test_X, model_predict_y, bias)
    print("under bias {},the accuracy is {}%".format(bias, accuracy * 100))
    plt.scatter(train_X, train_Y, marker='.', color='red', s=40, label='Train')
    plt.scatter(test_X, test_Y, marker='.', color='blue', s=40, label='Test')
    plt.plot(X, model.predict_y(X), linewidth=1, color='green')
    plt.show()
