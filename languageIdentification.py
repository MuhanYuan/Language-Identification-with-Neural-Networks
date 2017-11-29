# yuanmh
# Muhan Yuan

import numpy as np
import random
import math
from random import shuffle
import sys


def onehotencode(mat,char_list,des = True):
    out_mat = []
    for row in mat:
        if des == True:
            new_row = [0]*len(row)*len(char_list)
            for i in range(len(row)):
                new_row[i*len(char_list)+char_list.index(row[i])] = 1
            out_mat.append(new_row)
        else:
            new_row = [0] * len(char_list)
            new_row[char_list.index(row)] = 1
            out_mat.append(new_row)
    return out_mat

def max_index(l):
    ind = 0
    cur_max = l[0]
    for i in range(len(l)):
        if l[i]> cur_max:
            cur_max = l[i]
            ind = i
    return ind

def sample(l,ratio = 0.05):
    out = []
    count = 0
    for i in l:
        if count % (1/ratio)== 0:
            out.append(i)
        count += 1
    return out

# loading data
with open(sys.argv[1],"r") as trainfile:
    train = []
    for l in trainfile:
        if len(l.strip().split(" ",1))>1:
            train.append(l.strip().split(" ",1))
with open(sys.argv[2],"r") as devfile:
    dev = []
    for l in devfile:
        if len(l.strip().split(" ",1))>1:
            dev.append(l.strip().split(" ",1))
with open(sys.argv[3],"r") as testfile:
    test = []
    for l in testfile:
        test.append(l)

with open("languageIdentification.data/test_solutions","r") as testsolu:
    test_solu = []
    for l in testsolu:
        if len(l.strip().split(" ",1))>1:
            test_solu.append(l.strip().split(" ",1))

train_table = []
for line in train:
    for unit in range(len(line[1])-4):
        train_table.append([line[1][unit:unit+5], line[0]])
# train_table = sample(train_table)
# print len(train_table)

random.seed(722)
num = 5
d = 100
eta = 0.1

# build up c_list
c_list = []
for t in train:
    for char in t[1]:
        if char not in c_list:
            c_list.append(char)
for t in dev:
    for char in t[1]:
        if char not in c_list:
            c_list.append(char)
for t in test:
    for char in t:
        if char not in c_list:
            c_list.append(char)
c_list = sorted(c_list)
res_list = ["ENGLISH","FRENCH","ITALIAN"]

# initialize W B
W1_matrix = np.random.random((d,num*len(c_list))) # (100, 650)
b1_matrix = np.random.random((d,1)) #(100,1)
W2_matrix = np.random.random((len(res_list),d))  #(3,100)
b2_matrix = np.random.random((len(res_list),1))   #(3,1)

# train data for acc test
count = 0
train_t_mat = []
train_t_res = []
train_sentence_level_index = []
train_sentence_level_res = []
for line in train:
    train_sentence_level_res.append(line[0])
    temp_list = []
    for unit in range(len(line[1])-4):
        train_t_mat.append(line[1][unit:unit+5])
        train_t_res.append(line[0])
        temp_list.append(count)
        count +=1
    train_sentence_level_index.append(temp_list)
train_t_In_matrix = onehotencode(train_t_mat,c_list)
train_t_res_matrix = onehotencode(train_t_res,res_list,des=False)

# dev data for acc test
count = 0
dev_mat = []
# dev_res = []
sentence_level_index = []
sentence_level_res = []
for line in dev:
    sentence_level_res.append(line[0])
    temp_list = []
    for unit in range(len(line[1])-4):
        dev_mat.append(line[1][unit:unit+5])
        # dev_res.append(line[0])
        temp_list.append(count)
        count +=1
    sentence_level_index.append(temp_list)
dev_In_matrix = onehotencode(dev_mat,c_list)
# dev_res_matrix = onehotencode(dev_res,res_list,des=False)

# test data for acc test
count = 0
test_mat = []
test_sentence_level_index = []
test_sentence_level_res = [i[1] for i in test_solu]
for line in range(len(test)):
    temp_list = []
    for unit in range(len(test[line])-4):
        test_mat.append(test[line][unit:unit+5])
        temp_list.append(count)
        count += 1
    test_sentence_level_index.append(temp_list)
test_In_matrix = onehotencode(test_mat,c_list)


# round 0 test acc
print "Before Training:"
try:
    dev_res_cache = []
    for entry in range(len(dev_In_matrix)):
        x_matrix = np.matrix(dev_In_matrix[entry]).transpose()
        H_prime_matrix = np.dot(W1_matrix,x_matrix) + b1_matrix
        H_matrix = 1/(1+np.exp(-H_prime_matrix))
        y_prime_matrix = np.dot(W2_matrix,H_matrix) + b2_matrix
        y_mat = []
        denom = 0
        for i in range(y_prime_matrix.shape[0]):
            denom += math.exp(y_prime_matrix.__getitem__((i,0)))
        for j in range(y_prime_matrix.shape[0]):
            y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
        dev_res_cache.append(y_mat)

    dev_res = []
    for i in range(len(sentence_level_index)):
        temp = [0,0,0]
        for j in sentence_level_index[i]:
            for ind in range(len(dev_res_cache[j])):
                temp[ind]+= dev_res_cache[j][ind]
        r = max_index(temp)
        if r== 0:
            dev_res.append('ENGLISH')
        elif r == 1:
            dev_res.append('FRENCH')
        else:
            dev_res.append('ITALIAN')
    count = 0
    for i in range(len(dev_res)):
        if dev_res[i]== sentence_level_res[i]:
            count +=1
    print "Dev acc:"
    print float(count)/len(dev_res)
except:
    pass
# round 0 train acc
try:
    train_res_cache = []
    for entry in range(len(train_t_In_matrix)):
        x_matrix = np.matrix(train_t_In_matrix[entry]).transpose()
        H_prime_matrix = np.dot(W1_matrix,x_matrix) + b1_matrix
        H_matrix = 1/(1+np.exp(-H_prime_matrix))
        y_prime_matrix = np.dot(W2_matrix,H_matrix) + b2_matrix
        y_mat = []
        denom = 0
        for i in range(y_prime_matrix.shape[0]):
            denom += math.exp(y_prime_matrix.__getitem__((i,0)))
        for j in range(y_prime_matrix.shape[0]):
            y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
        train_res_cache.append(y_mat)

    train_t_res = []
    for i in range(len(train_sentence_level_index)):
        temp = [0,0,0]
        for j in train_sentence_level_index[i]:
            for ind in range(len(train_res_cache[j])):
                temp[ind]+= train_res_cache[j][ind]
        r = max_index(temp)
        if r== 0:
            train_t_res.append('ENGLISH')
        elif r == 1:
            train_t_res.append('FRENCH')
        else:
            train_t_res.append('ITALIAN')
    count = 0
    for i in range(len(train_t_res)):
        if train_t_res[i]== train_sentence_level_res[i]:
            count +=1
    print "Train acc:"
    print float(count)/len(train_t_res)
except:
    pass

# round start
for ep in range(3):
    shuffle(train_table)
    train_mat = [i[0] for i in train_table]
    train_res = [i[1] for i in train_table]

    train_In_matrix = onehotencode(train_mat,c_list)
    train_res_matrix = onehotencode(train_res,res_list,des=False)


    for entry in range(len(train_In_matrix)):
        # print entry
        x_matrix = np.matrix(train_In_matrix[entry]).transpose()   # (650,1)
        H_prime_matrix = np.dot(W1_matrix,x_matrix) + b1_matrix
        H_matrix = 1/(1+np.exp(-H_prime_matrix))    # (100,1)
        y_prime_matrix = np.dot(W2_matrix,H_matrix) + b2_matrix  # (3, 1)
        y_mat = []
        denom = 0
        for i in range(y_prime_matrix.shape[0]):
            denom += math.exp(y_prime_matrix.__getitem__((i,0)))
        for j in range(y_prime_matrix.shape[0]):
            y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
        y_matrix = np.matrix(y_mat) # (1,3)
        y_hat_matrix = np.matrix(train_res_matrix[entry]) # (1,3)
        l_y_matrix = y_matrix - y_hat_matrix  # (1,3)
        L_y_prime = []
        for j in range(y_hat_matrix.shape[1]):
            temp = 0
            for i in range(y_hat_matrix.shape[1]):
                if i==j:
                    temp += l_y_matrix.__getitem__((0,i)) * y_matrix.__getitem__((0,i)) * (1 - y_matrix.__getitem__((0,j)))
                else:
                    temp += l_y_matrix.__getitem__((0,i)) * y_matrix.__getitem__((0,i)) * (0 - y_matrix.__getitem__((0,j)))
            L_y_prime.append(temp)
        L_y_prime_matrix = np.matrix(L_y_prime).transpose() #(1,3)
        # print L_y_prime_matrix

        # L -- W2
        L_W2_matrix = np.dot(L_y_prime_matrix,H_matrix.transpose())  #(3,100)
        # L --b2
        L_b2_matrix = L_y_prime_matrix #(3,1)
        # L -- H
        L_H_matrix = np.dot(W2_matrix.transpose() ,L_y_prime_matrix) #(100,1)
        # L -- H'
        L_H_prime = []
        for i in range(L_H_matrix.shape[0]):
            temp = L_H_matrix.__getitem__((i,0)) * H_matrix.__getitem__((i,0)) * (1- H_matrix.__getitem__((i,0)))
            L_H_prime.append(temp)
        L_H_prime_matrix = np.matrix(L_H_prime).transpose()   #(100,1)
        # L -- W1
        L_W1_matrix = np.dot(L_H_prime_matrix, x_matrix.transpose())   #(100,650)
        # L -- b1
        L_b1_matrix = L_H_prime_matrix #(100,1)
        ### update
        W1_matrix = W1_matrix - eta * L_W1_matrix #
        b1_matrix = b1_matrix - eta * L_b1_matrix
        W2_matrix = W2_matrix - eta * L_W2_matrix
        b2_matrix = b2_matrix - eta * L_b2_matrix


    # test after epoch
    print "Round "+str(ep+1)
    dev_res_cache = []
    for entry in range(len(dev_In_matrix)):
        x_matrix = np.matrix(dev_In_matrix[entry]).transpose()
        H_prime_matrix = np.dot(W1_matrix,x_matrix) + b1_matrix
        H_matrix = 1/(1+np.exp(-H_prime_matrix))
        y_prime_matrix = np.dot(W2_matrix,H_matrix) + b2_matrix
        y_mat = []
        denom = 0
        for i in range(y_prime_matrix.shape[0]):
            denom += math.exp(y_prime_matrix.__getitem__((i,0)))
        for j in range(y_prime_matrix.shape[0]):
            y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
        dev_res_cache.append(y_mat)

    dev_res = []
    for i in range(len(sentence_level_index)):
        temp = [0,0,0]
        for j in sentence_level_index[i]:
            for ind in range(len(dev_res_cache[j])):
                temp[ind]+= dev_res_cache[j][ind]
        r = max_index(temp)
        if r== 0:
            dev_res.append('ENGLISH')
        elif r == 1:
            dev_res.append('FRENCH')
        else:
            dev_res.append('ITALIAN')
    count = 0
    for i in range(len(dev_res)):
        if dev_res[i]== sentence_level_res[i]:
            count +=1
    print "Dev acc:"
    print float(count)/len(dev_res)


    try:
        train_res_cache = []
        for entry in range(len(train_t_In_matrix)):
            x_matrix = np.matrix(train_t_In_matrix[entry]).transpose()
            H_prime_matrix = np.dot(W1_matrix,x_matrix) + b1_matrix
            H_matrix = 1/(1+np.exp(-H_prime_matrix))
            y_prime_matrix = np.dot(W2_matrix,H_matrix) + b2_matrix
            y_mat = []
            denom = 0
            for i in range(y_prime_matrix.shape[0]):
                denom += math.exp(y_prime_matrix.__getitem__((i,0)))
            for j in range(y_prime_matrix.shape[0]):
                y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
            train_res_cache.append(y_mat)

        train_t_res = []
        for i in range(len(train_sentence_level_index)):
            temp = [0,0,0]
            for j in train_sentence_level_index[i]:
                for ind in range(len(train_res_cache[j])):
                    temp[ind]+= train_res_cache[j][ind]
            r = max_index(temp)
            if r== 0:
                train_t_res.append('ENGLISH')
            elif r == 1:
                train_t_res.append('FRENCH')
            else:
                train_t_res.append('ITALIAN')
        count = 0
        for i in range(len(train_t_res)):
            if train_t_res[i]== train_sentence_level_res[i]:
                count +=1
        print "Train acc:"
        print float(count)/len(train_t_res)
    except:
        pass


# test output
test_res_cache = []
for entry in range(len(test_In_matrix)):
    x_matrix = np.matrix(test_In_matrix[entry]).transpose()
    H_prime_matrix = np.dot(W1_matrix,x_matrix) + b1_matrix
    H_matrix = 1/(1+np.exp(-H_prime_matrix))
    y_prime_matrix = np.dot(W2_matrix,H_matrix) + b2_matrix
    y_mat = []
    denom = 0
    for i in range(y_prime_matrix.shape[0]):
        denom += math.exp(y_prime_matrix.__getitem__((i,0)))
    for j in range(y_prime_matrix.shape[0]):
        y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
    test_res_cache.append(y_mat)

test_res = []
for i in range(len(test_sentence_level_index)):
    temp = [0,0,0]
    for j in test_sentence_level_index[i]:
        for ind in range(len(test_res_cache[j])):
            temp[ind]+= test_res_cache[j][ind]
    r = max_index(temp)
    if r== 0:
        test_res.append('ENGLISH')
    elif r == 1:
        test_res.append('FRENCH')
    else:
        test_res.append('ITALIAN')

count = 0
for i in range(len(test_res)):
    if test_res[i].lower()== test_sentence_level_res[i].lower():
        count +=1
print "test acc:"
print float(count)/len(test_res)

with open("languageIdentificationPart1.output,","w") as outputfile:
    for i in range(len(test_res)):
        outputfile.write("Line"+str(i+1)+" "+test_res[i]+"\n")
