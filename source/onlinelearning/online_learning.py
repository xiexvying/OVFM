import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from onlinelearning.ftrl_adp import *
from sklearn import svm

def svm_classifier(train_x, train_y,test_x, test_y):
    best_score=0
    best_C=-1
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = svm.LinearSVC(C=C,max_iter=100000)
        clf.fit(train_x,train_y)
        score = clf.score(test_x, test_y)
        if score>best_score:
            best_score=score
            best_C=C
    return  best_score,best_C

def calculate_svm_error(X_input, Y_label,n):
    length=int(0.7*n)
    X_train = X_input[:length, :]
    Y_train = Y_label[:length]
    X_test = X_input[length:, :]
    Y_test = Y_label[length:]
    best_score, best_C = svm_classifier(X_train, Y_train, X_test, Y_test)
    error = 1.0 - best_score
    return error,best_C

def generate_X(n,X_input,Y_label,decay_choice,contribute_error_rate):
    errors=[]
    decays=[]
    predict=[]
    mse=[]

    classifier=FTRL_ADP(decay = 1.0, L1=0., L2=0., LP = 1., adaptive =True,n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]
        p, decay,loss,w = classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    svm_error,_ =calculate_svm_error(X_input[:,1:], Y_label, n)

    return X_Zero_CER, svm_error

def generate_cap(n,X_input,Y_label,decay_choice,contribute_error_rate):
    errors=[]
    decays=[]

    classifier=FTRL_ADP(decay = 1.0, L1=0., L2=0., LP = 1., adaptive =True,n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row].data
        y = Y_label[row]
        p, decay,loss,w= classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)

    Z_imp_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return Z_imp_CER
def generate_tra(n, X_input, Y_label, decay_choice, contribute_error_rate):
    errors = []
    classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label[row]
        p, decay, loss ,w= classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    imp_CER=np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return imp_CER

def draw_cap_error_picture(Z_imp_CER,X_Zero_CER,svm_error):
    n=len(Z_imp_CER)
    plt.figure(figsize=(6, 4))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")  #
    x = range(n)

    plt.plot(x, Z_imp_CER  , color='green')    # the error of z_imp
    plt.plot(x, X_Zero_CER, color='blue')     # the error of x_zero
    plt.plot(x, [svm_error] * n ,color='red') # the error of svm

    plt.title("The error of z_imp_CER,x_zero_CER,SVM_CER")
    plt.show()
    plt.clf()

def draw_tra_error_picture(error_arr_Z,error_arr_X):
    n=len(error_arr_Z)
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 1.0))
    plt.xlim((0, n))
    plt.ylabel("CER")  #

    x = range(n)
    plt.plot(x, error_arr_Z, color='green')   # z_imp error picture

    plt.plot(x, error_arr_X, color='blue')    # x_imp error picture

    plt.title("The CER of trapezoid data stream")
    plt.show()
    plt.clf()