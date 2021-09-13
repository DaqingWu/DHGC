# -*- encoding: utf-8 -*-

import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import recall_score, precision_score

from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

######################################## Sinkhorn ########################################
def sinkhorn(pred, lambdas, row, col):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, lambdas)
    
    u = np.ones(num_node)
    v = np.ones(num_class)

    for index in range(1000):
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target

######################################## Evaluation ########################################
def best_map(y_true, y_pred):
    """
    https://github.com/jundongl/scikit-feature/blob/master/skfeature/utility/unsupervised_evaluation.py
    Permute labels of y_pred to match y_true as much as possible
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    label_set = np.unique(y_true)
    num_class = len(label_set)

    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    A = linear_assignment(-G)
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
    return new_y_pred.astype(int)

def evaluation(y_true, y_pred):
    y_pred_ = best_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_)
    f1_macro = f1_score(y_true, y_pred_, average='macro')
    # f1_micro = f1_score(y_true, best_map(y_true, y_pred), average='micro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    return acc, nmi, ari, f1_macro

######################################## vMF ########################################
def pdf_norm(dim, kappas):
    numerator = torch.pow(kappas, dim/2 -1)
    denominator = torch.pow(torch.mul(torch.pow(torch.ones_like(kappas)*2*math.pi, dim/2), iv(dim/2 -1, kappas)), -1)
    return torch.mul(numerator, denominator)

def A_d(dim, kappas):
    numerator = iv(dim/2, kappas)
    denominator = torch.pow(iv(dim/2 -1, kappas), -1)
    return torch.mul(numerator, denominator)

def estimate_kappa(dim, kappas):
    r = A_d(dim, kappas)
    numerator = dim*r - torch.pow(r, 3)
    denominator = torch.pow(1 - torch.pow(r, 2), -1)
    return torch.mul(numerator, denominator)