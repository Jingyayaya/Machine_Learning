import matplotlib.pyplot as plt
from tkinter import _flatten
import numpy as np
import math
from statistics import mean, stdev

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat)  # regularization
    # print(yMean)
    yMat = yMat - yMean
    # print(xMat)
    # regularize X's
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar  # （feature-mean）/std
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):  # test different lambda，get the coefficient
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def get_dataset(filename):
    input_file = open(filename, 'r')
    x = []
    y=[]
    for line in input_file:
        lineSplit = line.strip('\n').split(' ')
        x.append(float(lineSplit[0]))
        y.append(float(lineSplit[1]))
    return x,y
def get_test(filename):
    input_file = open(filename, 'r')
    x = []
    y=[]
    for line in input_file:
        lineSplit = line.strip('\n').split(' ')
        x.append(float(lineSplit[0]))
        y.append(float(lineSplit[1]))
    return x,y

def main():
    # import data
    test_file="dataset_test2.txt"
    x_test,y_test=get_test(test_file)
    x_test=np.array(x_test)
    filename="dataset_50.txt"
    x_input,y_input=get_dataset(filename)
    x_arange = np.array(x_input)
    y_True = (np.cos(x_arange) + 2) / (np.cos(1.4 * x_arange) + 2)
    x_Prec = np.linspace(-4, 4.8, 200)  # plot range

    Num = 100
    n = 11  # 10 order
    lamda = [np.exp(1),np.exp(0),np.exp(-2),np.exp(-4),np.exp(-6),np.exp(-10),np.exp(-12)]  # different lambda
    phi = np.mat(np.zeros((x_arange.size, n)))  # phi matrix
    x = np.mat(x_arange).T  # input matrix

    # phi matrix computation
    for i_n in range(n):
        for y_n in range(x_arange.size):
            phi[y_n, i_n] = x[y_n, 0] ** i_n

    plt.figure(figsize=(5, 8))
    e_in_list=[]
    e_out_list=[]

    numTestPts = 20
    lamda_list=[]
    for index in range(numTestPts):
        i_lamda=np.exp(index - 10)
        lamda_list.append(index - 10)
        plt.title("lambda = %f" % i_lamda)
        #plt.plot(x_Prec, (np.cos(x_Prec) + 2) / (np.cos(1.4 * x_Prec) + 2), color='g')

        y = np.mat(y_input).T
        # w
        W = (phi.T * phi + i_lamda * np.eye(n)).I * phi.T * y

        ploy = list(_flatten(W.T.tolist()))
        ploy.reverse()
        p = np.poly1d(ploy)

        prec_y = p(x_arange)
        f = (np.cos(x_arange) + 2) / (np.cos(1.4 * x_arange) + 2)
        e_in=0
        for i in range(len(x_arange)):
            e_in += (prec_y[i]-f[i]) * (prec_y[i]-f[i]) / len(x_arange)
        e_in_list.append(e_in)

        #plt.plot(x_arange, f, color='g', label='f_sample')
        #plt.plot(x_arange, prec_y, color='r', label='prec_sample')
        #plt.show()

        prec_y = p(x_test)
        f = (np.cos(x_test) + 2) / (np.cos(1.4 * x_test) + 2)
        e_out=0
        for i in range(len(x_test)):
            e_out += (prec_y[i]-f[i]) * (prec_y[i]-f[i]) / len(x_test)
        e_out_list.append(e_out)

        plt.plot(x_test, f, color='g', label='f')
        plt.xlabel("x")
        plt.plot(x_test, prec_y, color='r',label='prediction')
        plt.ylabel("value")
        plt.legend()
        plt.show()

    #plt.plot(lamda_list,e_in_list, color='r', label='e_in')
    #plt.plot(lamda_list,e_out_list,color='g',label='e_out')
    #plt.xlabel("log(lamda)")
    #plt.ylabel("error")
    #plt.legend()
    #plt.show()
def e_in(x_input, w, y_prediction):
    f=(np.cos(x_input) + 2) / (np.cos(1.4 * x_input) + 2)

    print()
def e_out(x_input, y_true, w, y_prediction):
    print()
if __name__ == '__main__':
    main()
