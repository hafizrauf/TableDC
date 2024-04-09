#!/usr/bin/env python
# coding: utf-8

from sklearn.cluster import DBSCAN
import hdbscan

from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from kneed import KneeLocator
import statistics
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
from sklearn.metrics.cluster import fowlkes_mallows_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from munkres import Munkres, print_matrix
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from collections import Counter
from scipy.special import comb

# set seed
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)


def cluster_acc2(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    # print(l1, l2)
    #为缺失类别进行补充，保障后面使用匈牙利可以一一映射
    if numclass1 > numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                # y_pred.append(i)
                # y_true.append(i)
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    if numclass1 < numclass2:
        print(l2)
        for i in l2:
            if i in l1:
                pass
            else:
                # y_pred.append(i)
                # y_true.append(i)
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    l1 = list(set(y_true))
    l2 = list(set(y_pred))
    numclass1 = len(l1)
    numclass2 = len(l2)
    # print(numclass1, numclass2)

   # if numclass1 != numclass2:
   #   print('error')
   #   return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    y_true = y_true[:len(y_true)-ind]
    new_predict = new_predict[:len(y_pred)-ind]
    y_pred = new_predict

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    # precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    # recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    # f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    # precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    # recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro



def acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size
    




def revised_rand_index(actual, pred):
    # Convert to integer type
    actual = actual.astype(int)
    pred = pred.astype(int)
    
    # Handle negative values in 'pred' (treat as outliers and convert to a unique label)
    unique_label = np.max(pred) + 1 if np.any(pred < 0) else np.max(pred)
    pred[pred < 0] = unique_label

    tp_plus_fp = comb(np.bincount(actual), 2).sum()
    tp_plus_fn = comb(np.bincount(pred), 2).sum()
    A = np.c_[(actual, pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(actual))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    print('tp = '+str(tp))
    print('tn = '+str(tn))
    print('fp = '+str(fp))
    print('fn = '+str(fn))
    
    print('Precision is = ' + str(tp/(tp+fp)))
    print('Recall is = ' + str(tp/(tp+fn)))
    p= tp/(tp+fp)
    r = tp/(tp+fn)
    Fmeasure = 2 * ((p*r)/(p+r))
    print('updated F1 is = ' + str(Fmeasure))

    return (tp + tn) / (tp + fp + fn + tn)



X=pd.read_csv('X.txt', header =None, sep = ' ')
Y=pd.read_csv('labels.txt', sep=' ', header = None)
Y= pd.DataFrame(Y)
Y = Y.to_numpy()
Y=Y.flatten()

n_clusters = 56 # select number of GT clusters

# print("-----------------------------------Birch----------------------------------------")
start = time()
from sklearn.cluster import Birch
brc = Birch(n_clusters = n_clusters) #
brc.fit(X)
brc.predict(X)
acc, f1 = cluster_acc2(Y, brc.predict(X))
print('Number of clusters '+str(len(np.unique(brc.predict(X)))))
print('Rand Score '+str(rand_score(Y, brc.predict(X))))
print('Rand Score '+str(revised_rand_index(Y, brc.predict(X))))
print('Adjusted Rand Score '+str(adjusted_rand_score(Y, brc.predict(X))))
print('Normalized mutual info scor '+str(normalized_mutual_info_score(Y, brc.predict(X))))
print('Accuracy '+str(acc))
print('F1 '+str(f1))
df = pd.DataFrame(brc.predict(X))
df.to_csv('Birch_pred.csv',sep=',', header = None)  
count=df.value_counts().tolist()
unary_clusters = count.count(1)
print('Number of unary clusters = '+str(unary_clusters))
print('Median cluster count = '+str(statistics.median(count)))
print('Mean cluster count = '+str(statistics.mean(count)))
end = time() 
print("Birch took ",{end-start}," sec to run.")


# print("-----------------------------------Kmean----------------------------------------")
# #------
from sklearn.cluster import KMeans
start = time()
kmeans = KMeans(n_clusters=n_clusters, n_init=2, random_state= 2).fit(X)
acc, f1 = cluster_acc2(Y, kmeans.labels_)
print('Number of clusters '+str(len(np.unique(kmeans.labels_))))
print('Rand Score '+str(rand_score(Y, kmeans.labels_)))
print('Rand Score '+str(revised_rand_index(Y, kmeans.labels_)))
print('Adjusted Rand Score '+str(adjusted_rand_score(Y, kmeans.labels_)))
print('Normalized mutual info scor '+str(normalized_mutual_info_score(Y, kmeans.labels_)))
print('Accuracy '+str(acc))
print('F1 '+str(f1))
df = pd.DataFrame(kmeans.labels_) 
df.to_csv('kmeans_pred.csv',sep=',', header = None)  
count=df.value_counts().tolist()
unary_clusters = count.count(1)
print('Number of unary clusters = '+str(unary_clusters))
print('Median cluster count = '+str(statistics.median(count)))
print('Mean cluster count = '+str(statistics.mean(count)))
end = time()
print("Kmeans took ",{end-start}," sec to run.")


print("--------------------DBSCAN Clustering---------------------")

min_samples = 56  # The number of samples in a neighborhood for a point to be considered as a core point.



nearest_neighbors = NearestNeighbors(n_neighbors=10)
neighbors = nearest_neighbors.fit(X)

distances, indices = neighbors.kneighbors(X)
distances = np.sort(distances[:,9], axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")


i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print("eps value ", distances[knee.knee])

eps = distances[knee.knee]  # if eps is 0 by knee method, just select 0.1, if still not then check 0.1 to 1.0 all parameters
#eps = 0.5
start = time()

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)

end = time()

labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(len(Y))
print(len(labels))
acc, f1 = cluster_acc2(Y, labels)
print('Number of clusters ' + str(n_clusters_))
print('Rand Score '+str(revised_rand_index(Y, labels)))
print('Adjusted Rand Score ' + str(adjusted_rand_score(Y, labels)))
print('Normalized mutual info score ' + str(normalized_mutual_info_score(Y, labels)))
print('Accuracy ' + str(acc))
print('F1 ' + str(f1))

df = pd.DataFrame(labels)
df.to_csv('DBSCAN_pred.csv', sep=',', header=None)

count = df.value_counts().tolist()
unary_clusters = count.count(1)
print('Number of unary clusters = '+str(unary_clusters))
print('Median cluster count = '+str(statistics.median(count)))
print('Mean cluster count = '+str(statistics.mean(count)))
print("DBSCAN took ", {end - start}, " sec to run.")
