# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujingdaily@gmail.com
@site: https://github.com/XuJing1022
@file: Adaboost_MH.py
@created: 18/12/1 上午9:24
"""
from functools import reduce

import operator
import numpy as np
from math import log, exp

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from tools import load_data, one_error, evaluate


class discrete_Adaboost_MH:
    """
    Implementation of Adaboost as base model
    """
    def __init__(self, X, y, test_x, test_y, class_list, T=20, model=DecisionTreeClassifier, pretrained=True):
        self.test_X = test_x
        self.test_y = test_y
        self.class_list = class_list
        self.k = len(self.class_list)
        self.m = X.shape[0]  # data num

        # construct (x,label)
        x_ = X  # np.tile(X, (self.k,1))  # k*m
        y_ = np.array([])  # k*m
        for class_ in self.class_list:
            for i in range(self.m):
                y_ = np.append(y_, 1 - 2 * np.sign(int(class_ not in y[i])))

        self.X = x_
        self.y = y_  # np.array
        self.T = T  # iteration times

        self.D_t = np.array([1 / (self.m*self.k) for i in range(self.m) for j in range(self.k)])
        self.D = np.append(np.array([]), self.D_t)  # weighted matrixs

        self.model = model  # base training model
        self.h = []  # T*k
        self.h_t = []
        self.alpha_t = None
        self.alphas = np.array([])
        if pretrained:
            self.train()

    def train(self, max_depth=3):
        for i in range(self.T):
            pred_y = np.array([])  # k*m
            # Train weak learner using distribution D_t
            for j in range(self.k):
                self.h_t = self.model(max_depth=max_depth, presort=True)
                self.h_t.fit(self.X, self.y[j*self.m:(j+1)*self.m], sample_weight=self.D_t[j*self.m:(j+1)*self.m])
                # Get weak hypothesis h_t
                self.h.append(self.h_t)

                # Choose alpha_t
                pred_y = np.append(pred_y, self.h_t.predict(self.X))

            r_t = np.dot(self.D_t, np.multiply(self.y, pred_y))

            if abs(r_t - 1) < 0.00000001:
                self.alpha_t = 0.5 * log((1 + r_t + 0.000001) / (1 - r_t + 0.000001))
            else:
                self.alpha_t = 0.5 * log((1 + r_t) / (1 - r_t))

            self.alphas = np.append(self.alphas, self.alpha_t)
            # Update
            self.D_t = np.multiply(self.D_t, list(map(exp, -self.alpha_t * np.multiply(self.y, pred_y))))
            self.D_t /= np.sum(self.D_t)
            # self.D = np.append(self.D, self.D_t)

            ret_index = self.predict(self.test_X, i + 1)
            # for i in range(len(X_test)):
            #     print(ret_index[i])
            #     print(y_test[i])
            scores = one_error(ret_index, self.test_y)
            y_train = []
            at_n = 3
            for j in range(len(ret_index)):
                tmp = [0] * self.k
                for ll in ret_index[j]:
                    if ret_index[j].index(ll) < self.k - at_n:
                        continue
                    tmp[self.class_list.index(ll)] = 1
                ret_index[j] = tmp

                tmp = [0] * self.k
                for ll in self.test_y[j]:
                    tmp[self.class_list.index(ll)] = 1
                y_train.append(tmp)

            precision, recall, error = evaluate(np.array(ret_index), np.array(y_train))
            print(i, precision, recall, error, scores)

    def predict(self, x, T):
        m = x.shape[0]
        pred = [self.alphas[t] * self.h[t*self.k+l].predict(x) for t in range(T) for l in range(self.k)]  # T*k*m
        pred = np.array(pred).reshape((T, self.k, m)).transpose(2, 1, 0)  # m*k*T
        pred = np.sum(pred, axis=-1)  # m*k

        H = np.sign(pred)
        return self.transfer(H, len(x))

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
    clf = discrete_Adaboost_MH(x, y, [1,2,3], T=200)
    ret = clf.predict(np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]]))
    print(ret)
    # [[1, 2], [2], [1, 3], [2, 3]]


def SDSS():
    path = 'data/SDSS.csv'
    x, y, class_list = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    clf = discrete_Adaboost_MH(X_train, y_train, class_list, T=50)
    ret_index = clf.predict(X_test)
    for i in range(len(X_test)):
        print(ret_index[i])
        print(y_test[i])
    scores = one_error(ret_index, y_test, all_=True)
    print('----------SDSS one error-----------\n', scores)


def mill():
    path = 'data/mediamill'
    train_x, train_y, test_x, test_y, class_list = load_data(path)
    clf = discrete_Adaboost_MH(train_x, train_y, test_x, test_y, class_list, T=50)
    ret_index = clf.predict(test_x)
    for i in range(len(test_x)):
        print(ret_index[i])
        print(test_y[i])


def yeast():
    path = 'data/yeast'
    X_train, y_train, X_test, y_test, class_list = load_data(path)
    clf = discrete_Adaboost_MH(X_train, y_train, X_test, y_test, class_list, T=200)
    ret_index = clf.predict(X_test, clf.T)
    for i in range(len(X_test)):
        print(ret_index[i])
        print(y_test[i])
    scores = one_error(ret_index, y_test, all_=True)
    print('----------yeast one error-----------\n', scores)

if __name__ == "__main__":
    # SDSS()
    # mill()
    yeast()
