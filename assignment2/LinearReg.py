import sys

def get_dataset(n):
    x_data=[]
    y_data=[]
    for i in range(n):
        data=[]
        line_splits = sys.stdin.readline().split(' ')
        for j in range(len(line_splits)-1):
            data.append(float(line_splits[j]))
        y_data.append(float(line_splits[-1]))
        x_data.append(data)
    return x_data,y_data

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

def calculate_mean(data):
    sum=0
    for i in data:
        sum += i
    return sum / len(data)

# Simple Linear Regression implementation
def SLR(x_data, y_data):
    trans_x_data=transpose(x_data)
    y_mean=calculate_mean(y_data)
    w=[]
    w.append(y_mean)

    for x in trans_x_data:
        x_mean=calculate_mean(x)
        nominator=0
        denominator=0
        for i in range(len(x)):
            nominator += (x[i]-x_mean) * (y_data[i]-y_mean)
            denominator += (x[i] -x_mean) ** 2
        res= nominator / denominator
        w.append(res)
        w[0] -= res* x_mean
    return w


def main():
    line_splits = sys.stdin.readline().split(' ')
    n=int(line_splits[0])
    dimension=int(line_splits[1])
    x_data,y_data=get_dataset(n)
    w=SLR(x_data,y_data)

    for i in w:
        print(i, end=' ')

if __name__ == '__main__':
    main()