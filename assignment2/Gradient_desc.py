import sys
import random

# print w to query the f(w) and the corresponding gradient
def print_w(w):
    for i in range(len(w)-1):
        print(str(w[i]), end=' ')
    print(w[-1])

# update the w using the gradient and step size eta
def update_w(old_w, gradient,eta):
    w=[]
    for i in range(len(old_w)):
        #w.append(round(old_w[i] - eta * gradient[i],6))
        w.append(old_w[i] - eta * gradient[i])
    return w

# check if the w exceeds the constrictions of L and G
def check_constrain(w,l_list,g_list):
    for i in range(len(w)):
        if w[i] < l_list[i] or w[i] > g_list[i]:
            w[i] = random.uniform(l_list[i], g_list[i])
    return w

# Implement the gradient descent algorithm
def gradient_descent(coef, eta, l_list, g_list,n_iters=2000):
    w=coef
    w=check_constrain(w,l_list,g_list)
    print_w(w)
    sys.stdout.flush()
    respond1=sys.stdin.readline()
    respond2=sys.stdin.readline().split(' ')
    #f_value = float(respond1)
    gradient = get_input_list(respond2)

    i_iters=1
    while i_iters < n_iters:
        w = update_w(w,gradient,eta)
        w=check_constrain(w, l_list, g_list)

        print_w(w)
        print_w(w)
        sys.stdout.flush()
        respond1 = sys.stdin.readline()
        respond2 = sys.stdin.readline().split(' ')
        #f_value = float(respond1)
        gradient = get_input_list(respond2)

        i_iters +=1

# generate the initial w
def get_initial_w(l_list,g_list):
    initial_w=[]
    for i in range(len(l_list)):
        #initial_w.append(random.uniform(l_list[i], g_list[i]))
        initial_w.append(random.uniform(l_list[i], g_list[i]))
    return initial_w

# transform the input string into float values list
def get_input_list(input):
    result=[]
    for i in range(len(input)):
        result.append(float(input[i]))
    return result

# main function
def main():
    dimension=sys.stdin.readline()
    l=sys.stdin.readline().split(' ')
    g=sys.stdin.readline().split(' ')
    l_list=get_input_list(l)
    g_list=get_input_list(g)

    initial_weights = get_initial_w(l_list,g_list)
    eta = 0.6
    gradient_descent(initial_weights, eta,l_list, g_list)


if __name__ == '__main__':
    main()

