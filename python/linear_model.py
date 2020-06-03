import pandas as pd
import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt

def sampleData(data, fraction = 0.75):
    allSet = data
    trainingSet = allSet.sample(frac = fraction).sort_index()
    testSet = pd.DataFrame(columns = allSet.columns)
    testindex = []
    for i in data.index.values:
        if i in trainingSet.index.values:
            allSet = allSet.drop(i)
    testSet = allSet
    print ('Training Set: {0}, Test Set: {1}'.format(len(trainingSet),len(testSet)))
    return (trainingSet, testSet)

def loadDataSet(data):
    numFeat = len(data.columns) - 1
    xArr = data.iloc[:,0:numFeat]
    yArr = data.iloc[:,numFeat]
    return(xArr, yArr)

# Linear Regression on TESL prostate dataset (source code)
prostate_url = 'http://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
delimiter = '\t'
data = pd.read_csv(prostate_url, sep = delimiter).drop(['Unnamed: 0', 'train'], axis = 1)
data.head(5)
data.to_csv(os.getcwd() + 'prostate.csv')

dataMat, labelMat = loadDataSet(data)

def OLSRegress(dataMat, labelMat, corrThreshold = 0.7):
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat).T
    xTx = xMat.T * xMat
    xTy = xMat.T * yMat
    # singularity check
    if np.linalg.det(xTx) == 0:
        print ("This matrix is singular, cannot do inverse")
        return
    else:
        beta = np.linalg.inv(xTx) * xTy # Alternative: beta = np.linalg.solve(xTx, xTy)
    # correlation check
    corr = dataMat.corr()
    for i in np.arange(0, len(dataMat.columns)):
        for j in np.arange(i+1, len(dataMat.columns)):
            if corr.iloc[i,j] >= corrThreshold:
                iname = corr.columns.values[i]
                jname = corr.columns.values[j]
                print ("Caution! {0} and {1} have a correlation coefficient equal to {2}.".format(iname, jname, corr.iloc[i,j]) + '\n')
    print ('Estimated Coefficient: ')
    print (beta)
    return beta

beta = OLSRegress(dataMat, labelMat)


# Scikit-learn 1.1.1 OLS
reg = sklearn.linear_model.LinearRegression()
data_train, data_test = sampleData(data, fraction = 0.7)
dataMat_train, labelMat_train = loadDataSet(data_train)
dataMat_test, labelMat_test = loadDataSet(data_test)
reg.fit(dataMat_train, labelMat_train)
reg.coef_
pred = reg.predict(dataMat_train).transpose()

sklearn.metrics.mean_squared_error(labelMat_train, pred)
sklearn.metrics.r2_score(labelMat_train, pred)

plt.scatter(dataMat_train.iloc[:,0], pred)
plt.show()

# Scikit-learn 1.1.2 Ridge
reg = sklearn.linear_model.Ridge(alpha = 0.5)