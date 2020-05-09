import numpy as np
import matplotlib.pyplot as plt
def get_dataset(filename):
    total_train_data=[]
    input_file = open(filename, 'r')
    train_data = []
    for line in input_file:
        lineSplit = line.strip('\n').split(' ')
        x=[]
        for i in range(1,len(lineSplit)):
           x.append(float(lineSplit[i]))
        train_data.append(x)
        if len(train_data) % 100 == 0:
            train_data = np.asarray(train_data)
            total_train_data.append(train_data)
            train_data = []
    return total_train_data

def pocket(mat, iter_num = 2000):
    '''
    pocket algorithm implement.
    what is different from PLA is that pocket pick a random point each time
    and update if the new point causes less mistakes
    '''

    m,n = mat.shape
    w = np.zeros(n)
    #w=[-0.2838768815928495, -0.4726050922908023, 0.3263571831129865]
    w=[-0.09623991573802122, -0.27821656367531816, 0.43339352018536703]
    bias = np.ones(m)
    x_vec = np.c_[bias,mat] # add the bias term
    update_cnt = 0

    out = np.sign(x_vec[:,0:-1] @ w)
    out[out == 0 ] = -1 # for sign(0) = -1 here
    pre_error = sum((out - x_vec[:,-1]) != 0) # calculate the error

    flag=False
    for i in range(iter_num):
    #while True:
        false_id = np.where(out != x_vec[:,-1])[0] # the indices of mistakes
        if not false_id.any():
            break
        rand_id = false_id[np.random.randint(len(false_id))] # randomly pick one false point
        w += x_vec[rand_id,-1] * x_vec[rand_id,0:-1] # updating the weight
        update_cnt +=1

        out = np.sign(x_vec[:,0:-1] @ w)
        out[out == 0] = -1
        new_error = sum((out - x_vec[:,-1]) != 0)
        if new_error < pre_error:
            w_pocket = w.copy() # w_pocket's base should not be w
            pre_error = new_error
            flag=True
    #if flag == False:
       # w_pocket=[-0.2838768815928495, -0.4726050922908023, 0.3263571831129865]
    return w_pocket,update_cnt,pre_error
def get_test_dataset(filename):
    test_dataset = []
    input_file = open(filename, 'r')
    for line in input_file:
        lineSplit = line.strip('\n').split(' ')
        x=[]
        for v in lineSplit:
           x.append(float(v))
        test_dataset.append(x)
    test_dataset=np.asarray(test_dataset)
    return test_dataset
def calculate_e_out(test_dataset,w_list):
    sum_total=0
    for test_row in test_dataset:
        sum=0
        for w in w_list:
            if np.dot(w,test_row[:-1])*test_row[-1] < 0:
                sum += 1
        sum_total += sum / len(w_list)
    e_out=sum_total / len(test_dataset)
    print("average e_out: "+str(e_out))

    e_out_list=[]
    for w in w_list:
        sum=0
        for test_row in test_dataset:
            if np.sign(np.dot(w,test_row[:-1])) != test_row[-1]:
                sum += 1
        e_out_list.append(sum/len(test_dataset))
    return e_out_list

def main():
    filename="dataset_outlier.txt"
    test_filename='dataset_test.txt'
    toral_data=get_dataset(filename)
    w_pocket_list = []
    update_cnt_list = []
    pre_error_list = []
    for mat in toral_data:
        w_pocket,update_cnt,pre_error = pocket(mat)
        print(w_pocket)
        print("e_in:"+str(pre_error/len(mat)))
        print("update_cnt:" + str(update_cnt))
        w_pocket_list.append(w_pocket)
        update_cnt_list.append(update_cnt)
        pre_error_list.append(pre_error/len(toral_data[0]))
    test_dataset=get_test_dataset(test_filename)
    e_out_list=calculate_e_out(test_dataset,w_pocket_list)
    s=0
    for c in update_cnt_list:
        s+=c
    print("average update iterations: " + str(s / len(update_cnt_list)))
    s=0
    for e in pre_error_list:
        s+=e
    print("average e_in: "+str(s/ len(pre_error_list)))

    plt.bar(range(len(e_out_list)), e_out_list)
    plt.show()
    plt.bar(range(len(update_cnt_list)), update_cnt_list)
    plt.show()
    plt.bar(range(len(pre_error_list)), pre_error_list)
    plt.show()
    #print(w, '\n', iteration,'\n',error)
if __name__ == '__main__':
    main()