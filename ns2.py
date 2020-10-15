import numpy as np
from sympy import *

x_train = np.array([[1, 0, 1],
                    [0, 0, 0],
                    [0, 1, 1],
                    [1, 1, 0]])
y_train = np.array([2, 0, 2, 2]).T

input_net = 3
output_net = 1

# инициализируем случайные веса
np.random.seed(1)
weight = 2 * np.random.random((input_net, output_net)) - 1


# функция активации
def activation(x):
    return 1 / (1 + np.exp(-x))  # sgd

# диф-л по w

def dif_w(warr, xtrainarr, ytrain):
    grad_w2 = np.zeros((input_net, output_net))
    output = 0
    for l_ in range(len(warr)):
        output = output + warr[l_] * xtrainarr[l_]
    for l in range(len(warr)):
        x, y = symbols('x y')
        y = xtrainarr[l] * (x - ytrain) ** 2
        dif = diff(y, x)
        f = lambdify(x, dif, 'numpy')
        grad_w2[l] = f(output)
    return grad_w2

epochs = 100
epsilon = 0.05

for i in range(epochs):
    print(weight)
    for k in range(len(x_train)):
        weight = weight - epsilon * dif_w(weight, x_train[k], y_train[k])

def prediction(arr):
    a = float(*np.dot(arr, weight))
    return a
