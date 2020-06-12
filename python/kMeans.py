import numpy as np
import pandas as pd
import copy, os
os.chdir('/Users/eloncha/Documents/GitHub/Elon/python')

### Kmeans
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        floatLine = map(float,curLine)
        dataMat.append(list(floatLine))
    return dataMat
def loadDataSetInPandas(fileName, by = '\t', is_header = None):
    dataMat = pd.read_table(fileName, sep = by, header = is_header)
    colNum = dataMat.shape[1]
    newColName = []
    for i in range(0,colNum):
        newColName.append("X{0}".format(str(i+1)))
    dataMat.columns = newColName
    return dataMat
def distMeasure(vecA, vecB, order = 2):
    dist = np.sqrt(np.sum(np.abs(np.power(vecA - vecB, order))))
    return dist #np.linalg.norm(vecA-vecB)
def proximityMatrix(dataMat, order=2):
    n = dataMat.shape[0]
    Mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Mat[i,j] = distMeasure(dataMat.iloc[i,:], dataMat.iloc[j,:])
    return Mat
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
def kMeans(dataMat, k):
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
dataMat = loadDataSetInPandas('testSet_kmeans.txt')
centroids, cluster = kMeans(dataMat,6)


### Bi-Kmeans
def rankSSE(dataMat1, dataMat2):
    SSE1 = np.sum(dataMat1['minDist'])
    SSE2 = np.sum(dataMat2['minDist'])
    if SSE1 > SSE2:
        print('Cluster 1 has higher SSE')
        return dataMat1, dataMat2
    elif SSE1 <= SSE2:
        print('Cluster 2 has higher SSE')
        return dataMat2, dataMat1
def biKmeans(dataMat, k):
    curK = 0
    m = dataMat.shape[0]
    p = dataMat.shape[1]
    cList = np.zeros((k,p))
    dList = pd.DataFrame(columns = kMeans(dataMat, 2)[1].columns)
    while curK < k:
        centroids, cluster = kMeans(dataMat.iloc[:,0:p], 2)
        dataMat1, dataMat2 = cluster[cluster['minIndex'] == 0], cluster[cluster['minIndex'] == 1]
        deadMat, surviveMat = rankSSE(dataMat1, dataMat2)
        cdf = centroids[surviveMat['minIndex'].values[0]]
        cList[curK] = cdf
        surviveMat.iloc[:,p] = curK
        dList = surviveMat.merge(dList, how = 'outer')
        curK += 1
        dataMat = copy.copy(deadMat)
    return cList, dList
cList, dList = biKmeans(dataMat, 6)


### Kmeans in scikit.learn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(6)
kmeans = KMeans(n_clusters = 6, init = 'k-means++', n_init = 1).fit(dataMat)
kmeans.cluster_centers_
kmeans.labels_

plt.scatter(kmeans.cluster_centers_[0:6,0], kmeans.cluster_centers_[0:6,1])
plt.show()

### Vector Quantization (LVQ)
import scipy as sp
from sklearn import cluster

from scipy.misc import face
face = face(gray=True)
plt.figure(1, figsize = (3,2.2))
plt.imshow(face, cmap = plt.cm.gray, vmin = 0, vmax = 250)
plt.show()

X = face.reshape((-1,1))
n_clusters = 3
kmeans = KMeans(n_clusters = n_clusters, n_init = 1).fit(X)
values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

plt.figure(2, figsize = (3,2.2))
plt.imshow(face_compressed, cmap = plt.cm.gray, vmin = 0, vmax = 250)
plt.show()