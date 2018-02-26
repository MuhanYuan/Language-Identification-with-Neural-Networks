import numpy as np
import random
import math
from random import shuffle

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

class NNclassifier():
    def __init__(self, ):
        self.W1_matrix = None
        self.b1_matrix = None
        self.W2_matrix = None
        self.b2_matrix = None


    def loading(self, trainfile, devfile = None, testfile = None, testsol = None):
        self.train = []
        with open(trainfile,"r") as trainfile:
            for l in trainfile:
                if len(l.strip().split(" ",1))>1:
                    self.train.append(l.strip().split(" ",1))


        if devfile:
            self.dev = []
            with open(devfile,"r") as devfile:
                for l in devfile:
                    if len(l.strip().split(" ",1))>1:
                        self.dev.append(l.strip().split(" ",1))

        if testfile:
            self.test = []
            with open(testfile,"r") as testfile:
                for l in testfile:
                    self.test.append(l)

        if testsol:
            self.test_solu = []
            with open(testsol,"r") as testsolu:
                for l in testsolu:
                    if len(l.strip().split(" ",1))>1:
                        self.test_solu.append(l.strip().split(" ",1))



    def initmat(self, num, d, eta):
        self.num = num
        self.d = d
        self.eta = eta
        c_list = []
        for t in self.train + self.dev + self.test:
            for char in t[1]:
                if char not in c_list:
                    c_list.append(char)
        self.c_list = sorted(c_list)
        self.res_list = ["ENGLISH","FRENCH","ITALIAN"]

        self.W1_matrix = np.random.random((d,num*len(self.c_list))) # (100, 650)
        self.b1_matrix = np.random.random((d,1)) #(100,1)
        self.W2_matrix = np.random.random((len(self.res_list),d))  #(3,100)
        self.b2_matrix = np.random.random((len(self.res_list),1))   #(3,1)




    def evaluate(self, data):
        count = 0
        data_mat = []
        sentence_level_index = []
        sentence_level_res = []
        for line in data:
            sentence_level_res.append(line[0])
            temp_list = []
            for unit in range(len(line[1])-4):
                data_mat.append(line[1][unit:unit+5])
                temp_list.append(count)
                count +=1
            sentence_level_index.append(temp_list)
        In_matrix = onehotencode(data_mat,self.c_list)

        res_cache = []
        for entry in range(len(In_matrix)):
            x_matrix = np.matrix(In_matrix[entry]).transpose()
            H_prime_matrix = np.dot(self.W1_matrix,x_matrix) + self.b1_matrix
            H_matrix = 1/(1+np.exp(-H_prime_matrix))
            y_prime_matrix = np.dot(self.W2_matrix,H_matrix) + self.b2_matrix
            y_mat = []
            denom = 0
            for i in range(y_prime_matrix.shape[0]):
                denom += math.exp(y_prime_matrix.__getitem__((i,0)))
            for j in range(y_prime_matrix.shape[0]):
                y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
            res_cache.append(y_mat)

        res = []
        for i in range(len(sentence_level_index)):
            temp = [0,0,0]
            for j in sentence_level_index[i]:
                for ind in range(len(res_cache[j])):
                    temp[ind]+= res_cache[j][ind]
            r = max_index(temp)
            if r== 0:
                res.append('ENGLISH')
            elif r == 1:
                res.append('FRENCH')
            else:
                res.append('ITALIAN')
        corr = 0
        for i in range(len(res)):
            if res[i]== sentence_level_res[i]:
                corr +=1
        return float(corr)/len(res)


    def training(self):
        shuffle(self.train)

        train_table = []
        for line in self.train:
            for unit in range(len(line[1])-4):
                train_table.append([line[1][unit:unit+5], line[0]])


        train_mat = [i[0] for i in train_table]
        train_res = [i[1] for i in train_table]

        train_In_matrix = onehotencode(train_mat,self.c_list)
        train_res_matrix = onehotencode(train_res,self.res_list,des=False)
        for entry in range(len(train_In_matrix)):
            # print entry
            x_matrix = np.matrix(train_In_matrix[entry]).transpose()   # (650,1)
            H_prime_matrix = np.dot(self.W1_matrix,x_matrix) + self.b1_matrix
            H_matrix = 1/(1+np.exp(-H_prime_matrix))    # (100,1)
            y_prime_matrix = np.dot(self.W2_matrix,H_matrix) + self.b2_matrix  # (3, 1)
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
            L_H_matrix = np.dot(self.W2_matrix.transpose() ,L_y_prime_matrix) #(100,1)
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
            self.W1_matrix = self.W1_matrix - self.eta * L_W1_matrix #
            self.b1_matrix = self.b1_matrix - self.eta * L_b1_matrix
            self.W2_matrix = self.W2_matrix - self.eta * L_W2_matrix
            self.b2_matrix = self.b2_matrix - self.eta * L_b2_matrix




if __name__ == "__main__":
    random.seed(722)
    nn = NNclassifier()
    nn.loading("train", "dev", "test", "test_solutions")
    nn.initmat(5,100,0.1)
    # print nn.evaluate(nn.dev)
    nn.training()
    print nn.evaluate(nn.dev)
    # print nn.


exit()
