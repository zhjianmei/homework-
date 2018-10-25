import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

def loadDataSet():
    dataMatrix = []
    datalabel = []
    #style.use('ggplot')
    iris = load_iris()
    data = iris.data
    target = iris.target
    X = data[0:100]#取前100行，有4个特征，100*4
    Y = target[0:100]
    datalabel = np.mat(Y)
    datalabel=np.transpose(datalabel)
    dataMatrix = np.mat(X)
    minmax_x_train = MinMaxScaler()
    x_train_std = minmax_x_train.fit_transform(dataMatrix)
    dataMatrix = np.mat(x_train_std)
    return dataMatrix,datalabel

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def graAscent(dataMatrix,matLabel,num):
    m,n=np.shape(dataMatrix)
    w=np.ones((n,1))
    alpha=0.01
    for i in range(num):
        error=sigmoid(dataMatrix*w)-matLabel
        w=w-alpha*dataMatrix.transpose()*error
    return w

def predict(w,X):
    m = X.shape[0]#取列数
    Y_prediction = np.zeros((m,1))#初始化预测值，初始化为0
    A = sigmoid(np.dot(X,w))
    for i in range(A.shape[0]):
        if A[i,0]>0.5:
            Y_prediction[i ,0]=1
        else:
            Y_prediction[i ,0]=0
    return Y_prediction

def loss(X,Y,num,print_cost=False):
   #costs=loss(weight,dataMatrix,matLabel, num)
    m, n = np.shape(dataMatrix)
    w = np.ones((n, 1))
    alpha = 0.01
    #assert (cost.shape == ())
    costs = []
    print_cost=0
    for i in range(num):
        # 记录成本
        error = sigmoid(dataMatrix * w) - matLabel
        w = w - alpha * dataMatrix.transpose() * error
        A = sigmoid(np.dot(X, w))
        w = np.array(w)
        A = np.array(A)
        Y= np.array(Y)
        cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
        if i % 100== 0:
            costs.append(cost)
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))
    return costs

if __name__ == '__main__':
    dataMatrix,matLabel=loadDataSet()
    print(dataMatrix.shape)
    print(matLabel.shape)
    num = 2000
    #weight=graAscent(dataMatrix,matLabel)
    weight= graAscent(dataMatrix,matLabel,num)
    #print(weight)
    print(weight.shape)
    y=predict(weight,dataMatrix)
    print(y.T)
    print("准确度为：", format(100 - np.mean(np.abs(y - matLabel) * 100)), "%")
    costs = loss(dataMatrix, matLabel, num)
    # costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    # lr=0.01
    # plt.title("学习率：",'lr')
    plt.show()