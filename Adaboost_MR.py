# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujingdaily@gmail.com
@site: https://github.com/XuJing1022
@file: Adaboost_MR.py
@created: 18/11/30 下午11:13
"""
from functools import reduce

import operator
import numpy as np
from math import log, exp

from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from tools import load_data, one_error


class discrete_Adaboost_MR():
    """
    Implementation of discrete Adaboost.MR based on ranking loss
    """
    def __init__(self, X, y, class_list, T=20, model=DecisionTreeClassifier, pretrained=True):
        self.class_list = class_list
        self.k = len(self.class_list)
        self.m = X.shape[0]  # data num

        # construct (x,label)
        x_ = np.tile(X, (self.k, 1))  # k*m
        y_ = np.array([])  # k*m
        for class_ in self.class_list:
            for i in range(self.m):
                y_ = np.append(y_, 1 - 2 * np.sign(int(class_ not in y[i])))

        self.X = x_
        self.y = y_  # np.array
        self.T = T  # iteration times

        self.D_t = np.array([])  # m*K*K
        self.pairs = []
        for i in range(self.m):
            tmp = []
            for l_0 in range(self.k):
                zeors_ = np.ones([len(self.class_list)]) * (1/self.m)  # np.zeros([len(self.class_list)])
                if self.class_list[l_0] not in y[i]:
                    d = 1/(self.m*len(y[i])*len(set(self.class_list)-set(y[i])))
                    for l in y[i]:
                        zeors_[self.class_list.index(l)] = d
                        self.pairs.append((i, l_0, self.class_list.index(l)))
                tmp = np.append(tmp, zeors_)
            self.D_t = np.append(self.D_t, tmp)
        self.pairs = set(self.pairs)

        self.D_t = np.reshape(self.D_t, (self.m, self.k, self.k))
        self.D = np.append(np.array([]), self.D_t)

        self.model = model  # base training model
        self.h = []
        self.h_t = None
        self.alphas = np.array([])
        if pretrained:
            self.train()

    def train(self, max_depth=3):
        for i in range(self.T):
            pred_y = np.array([])  # k*m
            # Train weak learner using distribution D_t
            for j in range(self.k):
                self.h_t = self.model(max_depth=max_depth, presort=True)
                self.h_t.fit(self.X[j*self.m:(j+1)*self.m], self.y[j*self.m:(j+1)*self.m], sample_weight=np.sum(self.D_t, axis=1)[:, j])
                # Get weak hypothesis h_t
                self.h.append(self.h_t)

                # Choose alpha_t
                pred_y = np.append(pred_y, self.h_t.predict(self.X[j*self.m:(j+1)*self.m]))
            tmp_sum = 0
            for (x_i, l0, l1) in self.pairs:
                tmp_sum += self.D_t[x_i][l0][l1] * (pred_y[l1 * self.m + x_i] - pred_y[l0 * self.m + x_i])
            # for i in range(self.m):
            #     for l0 in range(self.k):
            #         for l1 in range(self.k):
            #             if abs(self.D_t[i][l0][l1]) < 0.0000001:
            #                 continue
            #             tmp_sum += self.D_t[i][l0][l1]*(pred_y[i*self.k+l1]-pred_y[i*self.k+l0])
            r_t = tmp_sum * 0.5

            if abs(r_t - 1) < 0.00000001:
                self.alpha_t = 0.5 * log((1 + r_t + 0.000001) / (1 - r_t + 0.000001))
            else:
                self.alpha_t = 0.5 * log((1 + r_t) / (1 - r_t))

            self.alphas = np.append(self.alphas, self.alpha_t)
            # Update
            for (x_i, l0, l1) in self.pairs:
                self.D_t[x_i][l0][l1] = self.D_t[x_i][l0][l1] * exp(0.5*self.alpha_t*(pred_y[l0 * self.m + x_i] - pred_y[l1 * self.m + x_i]))
            self.D_t /= np.sum(self.D_t)
            self.D = np.append(self.D, self.D_t)

    def predict(self, x):
        m = x.shape[0]
        pred = [self.alphas[t] * self.h[t*self.k+l].predict(x) for t in range(self.T) for l in range(self.k)]  # T*k*m
        pred = np.array(pred).reshape((self.T, self.k, m)).transpose(2, 1, 0)  # m*k*T
        pred = np.sum(pred, axis=-1)  # m*k
        pred = list(map(list, pred))
        ret_index = list(map(list, np.argsort(pred)))  #
        for i in range(m):
            ret_index[i] = sorted(self.class_list, key=lambda x: ret_index[i].index(self.class_list.index(x)))  # ret_index is label_ret

        return ret_index, np.sort(pred)

    def transfer(self, H, x_m):
        """
        transfer -1/0,+1 to class name
        m==len(H)
        """
        ret = []
        for i in range(x_m):
            ret.append([])
        for i in range(x_m):
            for j in range(self.k):
                if H[i][j] != -1:
                    ret[i].append(self.class_list[j])
        return ret


def test():
    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]])
    y = np.array([[1, 2], [2], [3, 1], [2, 3]])
    clf = discrete_Adaboost_MR(x, y, [1,2,3], T=200)
    ret_index, pred = clf.predict(np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]]))
    print(ret_index)
    print(pred)
    '''
    [[3, 1, 2], [1, 3, 2], [2, 1, 3], [1, 2, 3]]
    [[-7.27423079  7.27423079  7.27423079]
     [-7.27423079 -7.27423079  7.27423079]
     [-7.27423079  7.27423079  7.27423079]
     [-7.27423079  7.27423079  7.27423079]]
    '''


def SDSS():
    path = 'data/SDSS.csv'
    x, y, class_list = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    clf = discrete_Adaboost_MR(X_train, y_train, class_list, T=50)
    ret_index, pred = clf.predict(X_test)
    for i in range(len(X_test)):
        print(ret_index[i])
        print(y_test[i])
    scores = one_error(ret_index, y_test)
    print('----------SDSS one error-----------\n', scores)
    print(pred)


def yeast():
    path = 'data/yeast'
    X_train, X_test, y_train, y_test, class_list = load_data(path)
    clf = discrete_Adaboost_MR(X_train, y_train, class_list, T=50)
    ret_index, pred = clf.predict(X_test)
    for i in range(len(X_test)):
        print(ret_index[i])
        print(y_test[i])
    scores = one_error(ret_index, y_test)
    print('----------yeast one error-----------\n', scores)
    print(pred)

if __name__ == "__main__":
    # test()
    SDSS()
    yeast()