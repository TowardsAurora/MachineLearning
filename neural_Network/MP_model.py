import numpy as np


def calculate_output(input, threshold):
    output = 0
    weight = np.zeros(len(input))
    for x, w in zip(input, weight):
        output += x * w
        print(output)
    output = output - threshold
    return output


if __name__ == '__main__':
    x = [1, 2, 3, 4]
    threshold = 2
    output = calculate_output(x, threshold)
    print(output)
