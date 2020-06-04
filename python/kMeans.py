import numpy as np
import pandas as pd

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        floatLine = map(float,curLine)
        dataMat.append(list(floatLine))
    return dataMat
# pandas function
def loadDataSetInPandas(fileName, by = '\t', is_header = None):
    dataMat = pd.read_table(fileName, sep = by, header = is_header)
    colNum = dataMat.shape[1]
    newColName = []
    for i in range(0,colNum):
        newColName.append("X{0}".format(str(i+1)))
    dataMat.columns = newColName
    return dataMat
def distMeasure(vecA, vecB, order = 2):
    dist = np.sqrt(np.sum(np.power(vecA - vecB, order)))
    return dist #np.linalg.norm(vecA-vecB)
def assignCent(dataMat, k):
    n = dataMat.shape[1]
    centroids = np.zeros((k,n))
    for i in range(0,n):
        xMax = np.max(dataMat.iloc[:,i])
        xMin = np.min(dataMat.iloc[:,i])
        xRand = np.random.uniform(xMin, xMax, k)
        for j in range(0,k):
            centroids[j,i] = xRand[j]
    return centroids

dataMat = loadDataSetInPandas('testSet_kmeans.txt')

def kMeans(dataMat, k):
    k = 6
    m = dataMat.shape[0]
    p = dataMat.shape[1]
    centroids = assignCent(dataMat, k)
    clusterChanged = True
    iteTime = 0
    print('Random Centroids assigned.')
    print(centroids)
    clusterDoc = copy.copy(dataMat)
    clusterDoc['minIndex'] = -1
    clusterDoc['minDist'] = np.Inf
    while clusterChanged:
        iteTime += 1
        clusterChanged = False

        for i in range(0, m): # loop through each data point to assign it to the nearest centroid
            minDist, minIndex = np.Inf, -1
            distList = np.zeros((1, k))
            for j in range(0, k):
                curDist = distMeasure(centroids[j], dataMat.iloc[i, :], order=2)
                distList[0, j] = curDist
                minIndex = np.argmin(distList)
                minDist = np.min(distList)
            if minDist < clusterDoc.iloc[i, p+1]:
                clusterChanged = True
                clusterDoc.iloc[i, p+1] = minDist
                clusterDoc.iloc[i, p] = minIndex

        for cent in range(0, k): # loop through each clusters to update centroids
            centroids[cent] = np.mean(clusterDoc[clusterDoc['minIndex'] == cent].iloc[:, 0:p], axis=0)

        print('Cluster Assignment Changed.')
        print('Iteration {0}'.format(str(iteTime)))

    return centroids, clusterDoc

kMeans(dataMat,6)








