import sys
def isValid(split_line):
    try:
        for v in split_line:
            float(v)
        return True
    except ValueError:
        return False
def get_train_data(line_num):
    train_data=[]
    for i in range(line_num):
        x=[]
        x.append(1)
        lineSplit=sys.stdin.readline().strip('\n').split(' ')
        if isValid(lineSplit):
            for j in range(len(lineSplit)):
                x.append(float(lineSplit[j]))
            train_data.append(x)
    return train_data
def compute_dot(W, x):
    val=0
    for i in range(len(W)):
        val += W[i] * x[i]
    return val

def update_w(W, x):
    new_w=[]
    for i in range(len(W)):
        new_val=W[i] + x[-1] * x[i]
        new_w.append(new_val)
    return new_w

def PLA(train_data):
    #initialize the W
    W=[]
    for i in range(len(train_data[0])-1):
        W.append(1)
    count=0
    while True:
        count += 1
        if count>1000:
            break
        is_complete= True
        for i in range(len(train_data)):
            x=train_data[i]
            y_tmp=compute_dot(W,x)
            if y_tmp * train_data[i][-1] > 0:
                continue
            else:
                is_complete=False
                W=update_w(W,x)
        if is_complete:
            break
    result=''
    for val in W:
        result =result + str(val) +' '
    print(result)

def main():
    firLineSplit=sys.stdin.readline().strip('\n').split(' ')
    line_num=int(firLineSplit[0])
    dimension=int(firLineSplit[1])

    train_data=get_train_data(line_num)
    PLA(train_data)

if __name__ == '__main__':
    main()