# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujing1@deepwise.com
@file: Adaboost_MH.py
"""

import operator
import numpy as np
from functools import reduce
from sklearn.tree import DecisionTreeClassifier

from base_boost import Adaboost


class discrete_Adaboost_MH(Adaboost):
    """
    Implementation of discrete Adaboost.MH
    """

    def __init__(self, X, y, T=20, model=DecisionTreeClassifier):
        self.x_m = X.shape[0]
        self.class_name_list = list(set(reduce(operator.add, y)))
        self.k = len(self.class_name_list)
        # construct (x,label)
        x_ = np.column_stack((np.tile(X,(self.k, 1)), np.repeat(self.class_name_list, self.x_m)))
        y_ = np.array([])
        for class_ in self.class_name_list:
            for i in range(self.x_m):
                y_ = np.append(y_, 1-2*np.sign(int(class_ not in y[i])))

        super().__init__(x_, y_, T, model)

    def transfer(self, H, m):
        """
        transfer -1/0,+1 to class name
        """
        ret = []
        for i in range(m):
            ret.append([])
        for j in range(self.k):
            for i in range(m):
                if H[i+j*m] != -1:
                    ret[i].append(self.class_name_list[j])
        return ret

    def form(self, data_):
        return np.column_stack((np.tile(data_, (self.k, 1)), np.repeat(self.class_name_list, len(data_))))


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [2, 5, 7, 8]])
    y = np.array([[1, 2], [2], [3, 1], [2, 3]])
    clf = discrete_Adaboost_MH(x, y, T=100)
    ret = clf.predict(np.array([[1, 2, 3, 4], [6, 7, 8, 9], [1, 7, 2, 8], [2, 5, 6, 9]]))
    print(ret)
    # [[1, 2], [1, 3], [1, 2, 3], [2, 3]]
