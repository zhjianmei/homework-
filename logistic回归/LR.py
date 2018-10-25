import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

def loadDataSet():
    dataMatrix = []
    datalabel = []
    style.use('ggplot')
    train = pd.read_csv('train.csv')
    train = train.fillna(0)#把数据中null的设为0
    date= train.ix[:, 2:15]#取出I1-I13的数字特征
    label = train.ix[:, 1:2]#取出真实标记
    datalabel = np.mat(label)
    dataMatrix = np.mat(date)
    #对数据进行归一化处理
    minmax_x_train = MinMaxScaler()
    x_train_std = minmax_x_train.fit_transform(dataMatrix)
   # X_normalized = normalize(dataMatrix, norm='l2')
    dataMatrix = np.mat(x_train_std)
    return dataMatrix,datalabel

def loadTest():
    dataMatrix = []
    datalabel = []
    style.use('ggplot')
    test = pd.read_csv('test.csv')
    label = pd.read_csv('submission.csv')
    test= test.fillna(0)
    date= test.ix[:, 1:14]
    label =label.loc[:, ['Label']]
    datalabel = np.mat(label)
    dataMatrix = np.mat(date)
    minmax_x_train = MinMaxScaler()
    x_train_std = minmax_x_train.fit_transform(dataMatrix)
    dataMatrix = np.mat(x_train_std)
    return dataMatrix,datalabel


def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def graAscent(dataMatrix,matLabel,num):
    m,n=np.shape(dataMatrix)#1599,13
    w=np.ones((n,1))#13,1
    alpha=0.01
    for i in range(num):
        E=dataMatrix.transpose()*(sigmoid(dataMatrix*w)-matLabel)#梯度
        w=w-alpha*E
    return w

def predict(w,X):
    m = X.shape[0]#取行数
    Y_prediction = np.zeros((m,1))
    A = sigmoid(np.dot(X, w))
    for i in range(A.shape[0]):
        if A[i,0]>0.5:
            Y_prediction[i ,0]=1
        else:
            Y_prediction[i ,0]=0
    return Y_prediction

def loss(X,Y,num,print_cost=False):
   #costs=loss(weight,dataMatrix,matLabel, num)
    m, n = np.shape(dataMatrix)#1599,13
    w = np.ones((n, 1))#13,1
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
        #cost = np.squeeze(cost)
        if i % 100== 0:
            costs.append(cost)
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))
    return costs

if __name__ == '__main__':
    dataMatrix,matLabel=loadDataSet()
    num=2000
    #weight=graAscent(dataMatrix,matLabel)
    weight= graAscent(dataMatrix,matLabel,num)
    print(weight)
    print(weight.shape)
    dataMatrix1, matLabel1= loadTest()
    #draw(weight)
    y=predict(weight,dataMatrix)
    y1=predict(weight,dataMatrix1)
    print(y.T)
    print(y1.T)
    print("训练集的准确度为：", format(100 - np.mean(np.abs(y - matLabel) * 100)), "%")
    print("测试集的准确度为：", format(100 - np.mean(np.abs(y1 - matLabel1) * 100)), "%")
    costs = loss(dataMatrix,matLabel,num)
    #costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    #lr=0.01
    #plt.title("学习率：",'lr')
    plt.show()
