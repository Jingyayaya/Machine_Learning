import sys
import math
import random
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
        # weights.append(random.uniform(-1,1))
        weights.append(0)
    return weights

def sigmoid(z):

    return  1 / (1 + math.exp(-z))

def compute_dot(weights, x):
    val=0
    for i in range(len(weights)):
        val += weights[i] * x[i]
    return val

def update_w(train_data,index,weights, alpha):
    sum=0
    for row in train_data:
        z=compute_dot(weights,row[:-1])
        sum += (sigmoid(z)-row[-1]) * row[index]
    new_w = weights[index] - alpha * sum / len(train_data)
    return new_w

def gradient_desc(train_data):
    alpha=0.1
    max_cycle=5000
    weights=initial_weights(len(train_data[0])-1)

    for i in range(max_cycle):
        try:
            new_weights=[]
            for index in range(len(weights)):
                new_w=update_w(train_data,index,weights,alpha)
                new_weights.append(new_w)
            weights = new_weights
        except:
            weights=initial_weights(len(train_data[0])-1)

    return weights

def print_weights(weights):
    for i in range(len(weights)-1):
        print(str(weights[i]), end=' ')
    print(weights[-1])


def main():
    line_split=sys.stdin.readline().split(' ')
    num=int(line_split[0])
    dimension=int(line_split[1])
    train_data=get_input(num)
    weights=gradient_desc(train_data)
    print_weights(weights)

if __name__ == '__main__':
    main()