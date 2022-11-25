import numpy as np

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


print(data)

train_len = 7
train_data = data[0:7]
test_data = data[7:]

print(train_data)
print(test_data)

x = train_data[:, 0]
y = train_data[:, 1]
print("x= ", x, "\n", "y= ", y)

sum_x = 0
sum_xx = 0
for x_i in x:
    sum_x += x_i
    sum_xx += x_i * x_i
avg_x = sum_x / len(x)
avg_xx = sum_xx / len(x)

sum_y = 0
for y_i in y:
    sum_y += y_i
avg_y = sum_y / len(y)

print("avg_x= ", avg_x, "\navy_y= ", avg_y)

sum_xy = 0
for x_i, y_i in train_data:
    sum_xy += x_i * y_i
    # print("x_i= ",x_i,"y_i=",y_i)
avg_xy = sum_xy / len(train_data)

print("avg_xy=", avg_xy)
print("avg_xx=", avg_xx)
w1 = (avg_xy - avg_x * avg_y) / (avg_xx - avg_x * avg_x)
w0 = avg_y - w1 * avg_x
print("w0 = ", w0, "\nw1 = ", w1)
print("y={}+{}x".format(w0, w1))


def predict(test):
    prediction = w0 + w1 * test
    return prediction


pre = predict(test_data[:, 0])
print("true = ", test_data[:, 1])
print("prediction = ", pre)

array = np.array([pre, test_data[:, 1]]).T
print(array)
test_len = len(test_data)

count = 0
for t, p in array:
    if abs(t - p) < 1:
        count += 1
print("在误差为1的情况下预测的准确率为：{}%".format(count / test_len * 100))
