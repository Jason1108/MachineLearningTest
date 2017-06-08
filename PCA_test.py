#coding:utf-8

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 去中心化
def calc(data_1_n_lst):
    sum = 0
    for i in data_1_n_lst:
        sum += i
    avr = sum/len(data_1_n_lst)
    X1 = [round(i - avr, 2) for i in data_1_n_lst]
    return X1

if __name__ == '__main__':
    data = load_iris()
    label = data.target
    X = data.data
    pca = PCA(n_components=2)
    reduce_X = pca.fit_transform(X)

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []

    for i in range(len(reduce_X)):
        if label[i] == 0:
            red_x.append(reduce_X[i][0])
            red_y.append(reduce_X[i][1])
        elif label[i] == 1:
            blue_x.append(reduce_X[i][0])
            blue_y.append(reduce_X[i][1])
        else:
            green_x.append(reduce_X[i][0])
            green_y.append(reduce_X[i][1])

    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()