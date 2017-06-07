# -*- coding: utf-8 -*-
import numpy
from sklearn.cluster import KMeans

n_clusters = 15

def loadData(filePath):
    # fr = open(filePath, 'r+', encoding='utf-8')
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName

if __name__ == '__main__':
    data, cityName = loadData('data/city.txt')
    km = KMeans(n_clusters)
    label = km.fit_predict(data)
    expenses = numpy.sum(km.cluster_centers_, axis=1)
    CityCluster = []
    for i in range(0, n_clusters):
        CityCluster.append([])
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])

    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
