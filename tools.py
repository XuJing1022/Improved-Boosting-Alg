# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujingdaily@gmail.com
@site: https://github.com/XuJing1022
@file: tools.py
@created: 18/12/1 下午6:05
"""
import pandas as pd
import numpy as np


def load_data(path):
    x=[]
    y=[]
    if path == 'data/spambase.data':
        pass
    elif path == 'data/SDSS.csv':
        data = pd.read_csv(path)
        y = np.array(data['class']).reshape((-1, 1))
        columns = data.columns.values.tolist()
        columns.remove('class')
        x = np.array(data[columns])
        return x, y, list(set(list(y.reshape((-1)))))
    elif path == 'data/mediamill':
        class_list = []
        train_x = []
        train_y = []
        with open('data/mediamill.train') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                [features, labels] = line.split('\t')
                x = []
                for i in features.split(','):
                    x.append(float(i))
                train_x.append(x)
                y = []
                for i in labels.split(','):
                    if not i:
                        continue
                    y.append(int(i))
                    class_list.append(int(i))
                train_y.append(y)
        test_x = []
        test_y = []
        with open('data/mediamill.test') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                [features, labels] = line.split('\t')
                x = []
                for i in features.split(','):
                    x.append(float(i))
                test_x.append(x)
                y = []
                for i in labels.split(','):
                    if not i:
                        continue
                    y.append(int(i))
                    class_list.append(int(i))
                test_y.append(y)
        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), np.array(class_list)
    elif path == 'data/yeast':
        class_list = [i+1 for i in range(14)]
        train_x = []
        train_y = []
        with open('data/yeast-train.arff') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                tmp = line.split(',')
                features = tmp[:103]
                labels = tmp[103:]
                x = []
                for i in features:
                    x.append(float(i))
                train_x.append(x)
                y = []
                for i in range(len(labels)):
                    if int(labels[i]) == 1:
                        y.append(class_list[i])
                train_y.append(y)
        test_x = []
        test_y = []
        with open('data/yeast-test.arff') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                tmp = line.split(',')
                features = tmp[:103]
                labels = tmp[103:]
                x = []
                for i in features:
                    x.append(float(i))
                test_x.append(x)
                y = []
                for i in range(len(labels)):
                    if int(labels[i]) == 1:
                        y.append(class_list[i])
                test_y.append(y)
        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), class_list


def one_error(pred_y, truth_y, all_=False):
    error = 0

    for i in range(len(truth_y)):
        flag = False
        k = 0
        if not all_:
            k = -len(truth_y[i])
        for ll in pred_y[i][k:]:
            if ll in truth_y[i]:
                flag = True
        if not flag:
            error += 1
            # print('predict wrong', pred_y[i], truth_y[i])
    error /= len(truth_y)
    return error


def n_mean(a):
    count = 0
    total = 0
    for i in a:
        if i>=0 and i<=1:
            total += i
            count += 1
    return (total/count)


def evaluate(h, Y):
    h = (h + 1) / 2
    Y = (Y + 1) / 2
    hY = h * Y
    TP = np.zeros(len(Y[0]))
    T = np.zeros(len(Y[0]))
    P = np.zeros(len(Y[0]))
    for k in range(len(Y[0])):
        TP[k] = sum(hY[:,k])
        T[k] = sum(Y[:,k])
        P[k] = sum(h[:,k])
    precision = TP / P
    recall = TP / T
    # print ('Precision: ' + str(n_mean(precision)))
    # print ('Recall: ' + str(n_mean(recall)))
    error = 0
    for i in range(len(h)):
        if (h[i] == Y[i]).all():
            error += 1
    # print ('One Error: ' + str(1 - error / len(h)))
    with open('DiscreteMH_mediamill.result', 'w+', encoding = 'utf8') as f:
        f.write("%.6f,%.6f,%.6f\n"%(n_mean(precision), n_mean(recall), 1 - error / len(h)))
    return n_mean(precision), n_mean(recall), 1 - error / len(h)

if __name__ == '__main__':
    load_data('data/yeast')