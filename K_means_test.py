# -*- coding: utf-8 -*-
import numpy
from sklearn.cluster import KMeans

n_clusters = 3


def load_data(file_path):
    # fr = open(filePath, 'r+', encoding='utf-8')
    fr = open(file_path, 'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName

if __name__ == '__main__':
    data, cityName = load_data('data/city.txt')
    km = KMeans(n_clusters)
    label = km.fit_predict(data)
    expenses = numpy.sum(km.cluster_centers_, axis=1)

    CityCluster = [[] for i in range(n_clusters)]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])

    for i in range(n_clusters):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])

