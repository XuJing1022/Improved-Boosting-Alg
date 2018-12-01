# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujingdaily@gmail.com
@site: https://github.com/XuJing1022
@file: Adaboost_MH_new.py
@created: 18/12/1 上午9:24
"""
from functools import reduce

import operator
import numpy as np
from math import log, exp
from sklearn.tree import DecisionTreeClassifier


class discrete_Adaboost_MH:
    """
    Implementation of Adaboost as base model
    """
    def __init__(self, X, y, T=20, model=DecisionTreeClassifier, pretrained=True):
        self.class_list = list(set(reduce(operator.add, y)))
        self.k = len(self.class_list)
        self.m = X.shape[0]  # data num

        # construct (x,label)
        x_ = np.tile(X, (self.k,1))  # k*m
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
                self.h_t.fit(self.X[j*self.m:(j+1)*self.m], self.y[j*self.m:(j+1)*self.m], sample_weight=self.D_t[j*self.m:(j+1)*self.m])
                # Get weak hypothesis h_t
                self.h.append(self.h_t)

                # Choose alpha_t
                pred_y = np.append(pred_y, self.h_t.predict(self.X[j*self.m:(j+1)*self.m]))

            r_t = np.dot(self.D_t, np.multiply(self.y, pred_y))

            if abs(r_t - 1) < 0.00000001:
                self.alpha_t = 0.5 * log((1 + r_t + 0.000001) / (1 - r_t + 0.000001))
            else:
                self.alpha_t = 0.5 * log((1 + r_t) / (1 - r_t))

            self.alphas = np.append(self.alphas, self.alpha_t)
            # Update
            self.D_t = np.multiply(self.D_t, list(map(exp, -self.alpha_t * np.multiply(self.y, pred_y))))
            self.D_t /= np.sum(self.D_t)
            self.D = np.append(self.D, self.D_t)

    def predict(self, x):
        m = x.shape[0]
        pred = [self.alphas[t] * self.h[t*self.k+l].predict(x) for t in range(self.T) for l in range(self.k)]  # T*k*m
        pred = np.array(pred).reshape((self.T, self.k, m)).transpose(2, 1, 0)  # m*k*T
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


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]])
    y = np.array([[1, 2], [2], [3, 1], [2, 3]])
    clf = discrete_Adaboost_MH(x, y, T=200)
    ret = clf.predict(np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]]))
    print(ret)
    # [[1, 2], [2], [1, 3], [2, 3]]
