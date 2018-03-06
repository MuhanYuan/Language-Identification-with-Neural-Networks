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

    def loadfile(self, filename):
        temp = []
        with open(filename,"r") as filedata:
            for l in filedata:
                if len(l.strip().split(" ",1))>1:
                    temp.append(l.strip().split(" ",1))
        return temp

    def loading(self, trainfile, devfile = None, testfile = None, testsol = None):
        self.train = self.loadfile(trainfile)
        if devfile:
            self.dev = self.loadfile(devfile)
        if testfile:
            self.test = self.loadfile(testfile)
        if testsol:
            self.test_solu = self.loadfile(testsol)

    def initmat(self, num, d, eta):
        self.num = num
        self.d = d
        self.eta = eta
        self.c_list = reduce(lambda x,y: x+y ,[t[1] for t in self.train + self.dev + self.test])
        self.c_list = sorted(set(self.c_list))

        self.res_list = list(set(t[0] for t in self.train))

        # re-initialize the weight and bias matrix
        self.W1_matrix = np.random.random((d,num*len(self.c_list))) # (100, 650)
        self.b1_matrix = np.random.random((d,1)) #(100,1)
        self.W2_matrix = np.random.random((len(self.res_list),d))  #(3,100)
        self.b2_matrix = np.random.random((len(self.res_list),1))   #(3,1)

    def forward(self, data_entry):
        x_matrix = np.matrix(data_entry).transpose()
        H_prime_matrix = np.dot(self.W1_matrix,x_matrix) + self.b1_matrix
        self.H_matrix = 1/(1+np.exp(-H_prime_matrix))
        y_prime_matrix = np.dot(self.W2_matrix,self.H_matrix) + self.b2_matrix

        # y_mat = []
        # denom = 0
        # for i in range(y_prime_matrix.shape[0]):
        #     denom += math.exp(y_prime_matrix.__getitem__((i,0)))
        # for j in range(y_prime_matrix.shape[0]):
        #     y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
        # y_matrix = np.matrix(y_mat)

        # print y_prime_matrix
        # print np.exp(y_prime_matrix)
        # print np.exp(y_prime_matrix).sum()
        y_matrix = (np.exp(y_prime_matrix) / np.exp(y_prime_matrix).sum()).transpose()

        return y_matrix

    def backprop(self,y_matrix, y_hat_matrix,data_entry):
        x_matrix = np.matrix(data_entry).transpose()
        l_y_matrix = y_matrix - y_hat_matrix
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
        L_W2_matrix = np.dot(L_y_prime_matrix,self.H_matrix.transpose())  #(3,100)
        # L --b2
        L_b2_matrix = L_y_prime_matrix #(3,1)
        # L -- H
        L_H_matrix = np.dot(self.W2_matrix.transpose() ,L_y_prime_matrix) #(100,1)
        # L -- H'
        L_H_prime = []
        for i in range(L_H_matrix.shape[0]):
            temp = L_H_matrix.__getitem__((i,0)) * self.H_matrix.__getitem__((i,0)) * (1- self.H_matrix.__getitem__((i,0)))
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




    def evaluate(self, data):
        X_mat = []
        y_mat = []
        for line in data:
            for unit in range(len(line[1])-4):
                X_mat.append(line[1][unit:unit+5])
                y_mat.append(line[0])
        # for line in data:
        #     sentence_level_res.append(line[0])
        #     temp_list = []
        #     for unit in range(len(line[1])-4):
        #         data_mat.append(line[1][unit:unit+5])
        #         temp_list.append(count)
        #         count +=1
        #     sentence_level_index.append(temp_list)
        # print len(data_mat)
        # print len(sentence_level_index)
        X_mat_enco = onehotencode(X_mat, self.c_list)
        res_cache = []
        for ind, entry in enumerate(X_mat_enco):
            res_cache.append(self.forward(entry).tolist())

        # print res_cache

        res = map(max_index, res_cache)
        res = [self.res_list[i] for i in res]

        corr = 0
        for i,v in enumerate(res):
            if v == y_mat[i]:
                corr +=1
        return float(corr)/len(res)


    def training(self):
        shuffle(self.train)
        X = []
        y = []
        for line in self.train:
            for unit in range(len(line[1])-4):
                X.append(line[1][unit:unit+5])
                y.append(line[0])

        X_matrix = onehotencode(X,self.c_list)
        y_hat = onehotencode(y,self.res_list,des=False)

        for entry, value in enumerate(X_matrix):
            # print entry
            # x_matrix = np.matrix(train_In_matrix[entry]).transpose()   # (650,1)
            # H_prime_matrix = np.dot(self.W1_matrix,x_matrix) + self.b1_matrix
            # H_matrix = 1/(1+np.exp(-H_prime_matrix))    # (100,1)
            # y_prime_matrix = np.dot(self.W2_matrix,H_matrix) + self.b2_matrix  # (3, 1)
            # y_mat = []
            # denom = 0
            # for i in range(y_prime_matrix.shape[0]):
            #     denom += math.exp(y_prime_matrix.__getitem__((i,0)))
            # for j in range(y_prime_matrix.shape[0]):
            #     y_mat.append(math.exp(y_prime_matrix.__getitem__((j,0)))/ denom)
            #
            # y_matrix = np.matrix(y_mat) # (1,3)
            y_matrix = self.forward(value)
            # print y_matrix


            y_hat_matrix = np.matrix(y_hat[entry]) # (1,3)
            self.backprop(y_matrix, y_hat_matrix, value)
            # backprop
            # l_y_matrix = y_matrix - y_hat_matrix  # (1,3)
            #
            # # print y_matrix
            # # print y_matrix.transpose()
            # print y_hat_matrix
            # print l_y_matrix
            # exit()
            #
            # L_y_prime = []
            # for j in range(y_hat_matrix.shape[1]):
            #     temp = 0
            #     for i in range(y_hat_matrix.shape[1]):
            #         if i==j:
            #             temp += l_y_matrix.__getitem__((0,i)) * y_matrix.__getitem__((0,i)) * (1 - y_matrix.__getitem__((0,j)))
            #         else:
            #             temp += l_y_matrix.__getitem__((0,i)) * y_matrix.__getitem__((0,i)) * (0 - y_matrix.__getitem__((0,j)))
            #     L_y_prime.append(temp)
            # L_y_prime_matrix = np.matrix(L_y_prime).transpose() #(1,3)
            # # print L_y_prime_matrix
            #
            # # L -- W2
            # L_W2_matrix = np.dot(L_y_prime_matrix,H_matrix.transpose())  #(3,100)
            # # L --b2
            # L_b2_matrix = L_y_prime_matrix #(3,1)
            # # L -- H
            # L_H_matrix = np.dot(self.W2_matrix.transpose() ,L_y_prime_matrix) #(100,1)
            # # L -- H'
            # L_H_prime = []
            # for i in range(L_H_matrix.shape[0]):
            #     temp = L_H_matrix.__getitem__((i,0)) * H_matrix.__getitem__((i,0)) * (1- H_matrix.__getitem__((i,0)))
            #     L_H_prime.append(temp)
            # L_H_prime_matrix = np.matrix(L_H_prime).transpose()   #(100,1)
            # # L -- W1
            # L_W1_matrix = np.dot(L_H_prime_matrix, x_matrix.transpose())   #(100,650)
            # # L -- b1
            # L_b1_matrix = L_H_prime_matrix #(100,1)
            # ### update
            # self.W1_matrix = self.W1_matrix - self.eta * L_W1_matrix #
            # self.b1_matrix = self.b1_matrix - self.eta * L_b1_matrix
            # self.W2_matrix = self.W2_matrix - self.eta * L_W2_matrix
            # self.b2_matrix = self.b2_matrix - self.eta * L_b2_matrix
            # print self.b2_matrix




if __name__ == "__main__":
    random.seed(722)
    nn = NNclassifier()
    nn.loading("train", "dev", "test", "test_solutions")
    nn.initmat(5,100,0.1)

    # print nn.c_list

    # dev accuracy before training
    print "Accuracy before training:"
    print nn.evaluate(nn.dev)
    print "first epoch"
    print nn.b2_matrix
    nn.training()
    print nn.b2_matrix
    print nn.evaluate(nn.dev)
    print "second epoch"
    nn.training()
    print nn.b2_matrix
    print nn.evaluate(nn.dev)
    print "third epoch"
    nn.training()
    print nn.b2_matrix
    print nn.evaluate(nn.dev)
    # print nn.
