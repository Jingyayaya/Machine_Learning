import sys
import math
import random
from statistics import mean, stdev
def get_input(num):
    train_data=[]
    for i in range(num):
        row=[]
        row.append(1)
        line_splits=sys.stdin.readline().split(' ')
        for number in line_splits:
            row.append(float(number))
        train_data.append(row)
    return train_data

def initial_weights(num):
    weights=[]
    for i in range(num):
        #weights.append(random.uniform(-1,1))
        weights.append(0)
    return weights

def sigmoid(z):

    return  1 / (1 + math.exp(-z))

def compute_dot(weights, x):
    val=0
    for i in range(len(weights)):
        val += weights[i] * x[i]
    return val

def update_w(train_data,weights):
    gradient=[0] * len(weights)
    for row in train_data:
        z=compute_dot(weights,row[:-1])
        for index in range(len(weights)):
            gradient[index] = gradient[index] + (1-sigmoid(row[-1] * z)) * row[-1] * row[index]
            #sum += (1-sigmoid(row[-1] * z)) * row[-1] * row[index]

    #new_w = weights[index] + eta * sum/len(train_data)
    return gradient

def gradient_asc(train_data):
    eta=1
    max_cycle=500
    weights=initial_weights(len(train_data[0])-1)

    for i in range(max_cycle):
        try:
            new_weights=[]
            gradient=update_w(train_data,weights)
            for index in range(len(weights)):
                new_w=weights[index] + eta * gradient[index] / len(train_data)
                new_weights.append(new_w)
            weights = new_weights
        except:
            weights=initial_weights(len(train_data[0])-1)

    return weights

def print_weights(weights):
    for i in range(len(weights)-1):
        print(str(weights[i]), end=' ')
    print(weights[-1])

def initial_matrix(n_rows, n_coloums):
    matrix=[]
    for i in range(n_rows):
        row=[]
        for j in range(n_coloums):
            row.append(0)
        matrix.append(row)
    return matrix

def transpose(matrix):
    new_matrix=initial_matrix(len(matrix[0]), len(matrix))

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[j][i]=matrix[i][j]
    return new_matrix

def preprocess(train_data):
    train_data=transpose(train_data)
    means_list=[]
    stds_list=[]
    for i in range(1,len(train_data)-1):
        mean_data=mean(train_data[i])
        means_list.append(mean_data)
        std_data=stdev(train_data[i])
        stds_list.append(std_data)
        for j in range(len(train_data[i])):
            train_data[i][j]=(train_data[i][j]-mean_data)/std_data
    return means_list, stds_list,train_data
def main():
    line_split=sys.stdin.readline().split(' ')
    num=int(line_split[0])
    dimension=int(line_split[1])
    train_data=get_input(num)
    means_list, stds_list,train_data=preprocess(train_data)
    train_data=transpose(train_data)
    weights=gradient_asc(train_data)
    for i in range(len(means_list)):
        weights[0] = weights[0] - (weights[i + 1] * (means_list[i] / stds_list[i]))
        weights[i + 1] = weights[i + 1] / stds_list[i]
    print_weights(weights)

if __name__ == '__main__':
    main()