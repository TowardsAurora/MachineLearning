"""
多元情况下求解：
公式：y=W*X+b
====>y_pre=W_hat*X_hat
minimize sum(y-y_pre)^2
min sum(y-W_hat*X_hat)^2
对W_hat求偏导令其为0
Partial y/W_hat:(y-W_hat*X_hat).T*(y-W_hat*X_hat)=0
solve: W_hat=(X_hat.T*X_hat)^-1*X_hat.T*y
"""
import numpy as np


def spilt_data(data, spilt_point):
    train_data = data[0:spilt_point]
    test_data = data[spilt_point:len(data)]
    return train_data, test_data


def create_data(data):
    X = data[:, 0:len(data[0]) - 1]
    Y = data[:, -1]
    return X, Y


class solution:
    def __init__(self):
        self.W_hat = None

    @staticmethod
    def X_to_X_hat(X):
        b = np.ones(len(X))
        X = np.insert(X, 0, b, axis=1)
        return X

    def train(self, X, Y):
        x_hat = self.X_to_X_hat(X)
        self.W_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_hat.T, x_hat)), x_hat.T), Y)
        return self.W_hat

    def predict(self, X):
        x_hat = self.X_to_X_hat(X)
        pre = np.dot(x_hat, W_hat)
        return pre

    def test(self, X, Y, bias):
        predict_y = self.predict(X)
        count = 0
        for y, y_pre in zip(Y, predict_y):
            if abs(y - y_pre) < bias:
                count += 1
        accuracy = count / len(X)
        return accuracy


if __name__ == '__main__':
    data = np.array([
        [6, 2, 7],
        [8, 1, 9],
        [10, 0, 13],
        [14, 2, 17.5],
        [18, 0, 18],
        [8, 2, 11],
        [9, 0, 8.5],
        [11, 2, 15],
        [16, 2, 18],
        [12, 0, 11]
    ])
    train_data, test_data = spilt_data(data, 7)
    train_X, train_Y = create_data(train_data)
    print("train data:\n", "X:\n", train_X, "\nY:\n", train_Y)
    test_X, test_Y = create_data(test_data)
    print("test data:\n", "X:\n", test_X, "\nY:\n", test_Y)
    model = solution()
    W_hat = model.train(train_X, train_Y)
    print("W_hat = ", W_hat)
    model_predict = model.predict(test_X)
    print("the model predict y is:", model_predict)
    bias = 2
    acc = model.test(test_X, test_Y, bias)
    print("under bias {},the accuracy is {}%".format(bias, acc * 100))
