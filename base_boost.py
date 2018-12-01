# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujing1@deepwise.com
@file: base_boost.py
"""


import numpy as np
from math import log, exp
from sklearn.tree import DecisionTreeClassifier


class Adaboost:
    """
    Implementation of Adaboost as base model
    """
    def __init__(self, X, y, T=20, model=DecisionTreeClassifier, pretrained=True):
        self.X = X
        self.m = self.X.shape[0]  # data num
        self.y = y.copy()  # np.array
        self.T = T  # iteration times

        self.D_t = np.array([1 / self.m for i in range(self.m)])
        self.D = np.append(np.array([]), self.D_t)  # weighted matrixs

        self.class_list = list(set(y))
        if len(self.class_list) != 2:
            raise TypeError('this is not a 2-classes problem')
        if [1,-1] == self.class_list:
            self.neg = -1
            self.pos = 1
        else:
            self.neg = self.class_list[0]  # set negative value with class[0]
            self.pos = self.class_list[1]
            self.y = [-1 if self.y[i] == self.neg else 1 for i in range(self.m)]

        self.model = model  # base training model
        self.h = []
        self.h_t = None
        self.alphas = np.array([])
        if pretrained:
            self.train()

    def train(self):
        for i in range(self.T):
            # Train weak learner using distribution D_t
            self.h_t = self.model(max_depth=3, presort=True)
            self.h_t.fit(self.X, self.y, sample_weight=self.D_t)
            # Get weak hypothesis h_t
            self.h.append(self.h_t)
            # Choose alpha_t
            self.alpha_t = self.choose_alpha()
            self.alphas = np.append(self.alphas, self.alpha_t)
            # Update
            pred_y = self.h_t.predict(self.X)
            self.D_t = np.multiply(self.D_t, list(map(exp, -self.alpha_t * np.multiply(self.y, pred_y))))
            self.D_t /= np.sum(self.D_t)
            self.D = np.append(self.D, self.D_t)

    def choose_alpha(self):
        pred_y = self.h_t.predict(self.X)
        r_t = np.dot(np.multiply(self.D_t, self.y), pred_y)
        if r_t == 1:
            alpha_t = 0.5 * log((1 + r_t + 0.001) / (1 - r_t + 0.001))
        else:
            alpha_t = 0.5 * log((1 + r_t) / (1 - r_t))
        return alpha_t

    def predict(self, x):
        self.pred_x = self.form(x)
        pred = [self.alphas[t] * self.h[t].predict(self.pred_x) for t in range(self.T)]
        H = np.sign(np.sum(pred, axis=0))
        return self.transfer(H, len(x))

    def form(self, data_):
        return data_

    def transfer(self, H, x_m):
        """
        transfer -1/0,+1 to class name
        m==len(H)
        """
        return [self.neg if H[i] == -1 else self.pos for i in range(x_m)]


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]])
    y = np.array([1, 2, 2, 1])
    clf = Adaboost(x, y)
    ret = clf.predict(np.array([[1, 7, 2, 8], [2, 5, 6, 9], [2, 3, 4, 5]]))
    print(ret)
    # [1, 1]
