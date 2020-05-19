
import pandas as pd
import numpy as np
import sklearn

# Linear Regression on TESL prostate dataset (source code)
prostate_url = 'http://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
delimiter = '\t'
data = pd.read_csv(prostate_url, sep = delimiter).drop(['Unnamed: 0', 'train'], axis = 1)
data.head(5)

def loadDataSet(data):
    numFeat = len(data.columns) - 1
    xArr = data.iloc[:,0:numFeat]
    yArr = data.iloc[:,numFeat]
    return(xArr, yArr)

dataMat, labelMat = loadDataSet(data)

def OLSRegress(dataMat, labelMat):
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat).T
    xTx = xMat.T * xMat
    xTy = xMat.T * yMat
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    else:
        beta = np.linalg.inv(xTx) * xTy # Alternative: beta = nnp.linalg.solve(xTx, xTy)
    return beta

beta = OLSRegress(dataMat, labelMat)

def pointEst(beta, dataMat):
