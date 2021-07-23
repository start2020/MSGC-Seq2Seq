import logging
import pickle
import scipy.sparse as sp
import sys
import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

#######################################  DCRNN ###########################################
def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

###################################  STGCN ############################################
def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    W = W.astype(np.float32)
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacians
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    N,_ = L.shape
    L0, L1 = np.mat(np.identity(N)), np.mat(np.copy(L))
    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    print('file_path',file_path)
    try:
        _, _, W = load_graph_data(file_path)
        #W = pd.read_csv(file_path, header=None).values
        print(W.shape)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        df = pd.DataFrame(W)
        #df.to_excel('./W.xlsx')
        print("save W successfully")
        return W

######################################  RMASTGRU ##############################

# adj_mx:ndarray, L:ndarray
def transform(adj_mx, filter_type="laplacian"):
    if filter_type == "laplacian":
        L = calculate_scaled_laplacian(adj_mx, lambda_max=None)
    elif filter_type == "random_walk":
        L = calculate_random_walk_matrix(adj_mx).T
    elif filter_type == "dual_random_walk":
        L = calculate_random_walk_matrix(adj_mx.T).T
    elif filter_type == "scaled_laplacian":
        L = calculate_scaled_laplacian(adj_mx)
    else:
        L = adj_mx
    return L.A

# matrices:ndarray (B,P,N,N)
def all_transform(matrices, filter_type='random_walk'):
    B = matrices.shape[0]
    P = matrices.shape[1]
    Matrices = []
    for i in range(B):
        xp = []
        for j in range(P):
            adj_mx = matrices[i, j, ...]  # (N,N), ndarray
            t = transform(adj_mx, filter_type=filter_type) #(118,118)
            xp.append(t)
        xp = np.stack(xp, axis=0)
        Matrices.append(xp)
    Matrices = np.stack(Matrices, axis=0)
    return Matrices

# def prior(Q,P,A,t):
#     N = A.shape[0]
#     All = np.zeros(shape=(Q, P, N, N), dtype=np.float32)
#     for q in range(P+1,Q+P+1):
#         for p in range(1,P+1):
#             t1, t2, t3, t4 = (p-1)*t, p*t, (q-1)*t, q*t
#             S = np.zeros_like(A, shape=(N, N), dtype=np.float32)
#             for i in range(N):
#                 for j in range(N):
#                     av = A[i][j]
#                     if t1 - t4  >= av:
#                         S[i][j] = 0.
#                     elif t4 - t1 > av >= t3 - t1:
#                         S[i][j] = (t4 - (t1 + av))/t
#                     elif t1 + av < t3 <= t2 + av:
#                         S[i][j] = (t2 + av - t3) / t
#                     else:
#                         S[i][j] = 0.
#             #path = '.\\attention-{:02d}P--{:02d}Q.xlsx'.format(p,q)
#             #dfs = pd.DataFrame(S, index=df.index, columns=df.columns)
#             #dfs.to_excel(path)
#             All[q-(P+1),p-1,...]=S
#     return All

def prior(Q, P, A, t):
    t = float(t)
    N = A.shape[0]
    All = np.zeros(shape=(Q, P, N, N), dtype=np.float32)
    for q in range(P,Q+P):
        for p in range(0,P): #p=[0,...,P-1], q=[P,...,P+Q-1]
            S = np.zeros_like(A, dtype=np.float32)
            # S = np.zeros_like(A, shape=(N, N), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    t1, t2, t3, t4 = p * t, (p + 1) * t, q * t, (q + 1) * t
                    av = A[i][j]
                    if np.isnan(av) or av==-1 or np.isinf(av):#-1和nan都表示没有值
                        S[i][j] = 0.

                    else:
                        t1, t2 = t1 + av, t2 + av
                        if i==j:
                            S[i][j] = 1.0
                        else:
                            if t1 >= t4 or t2 <= t3:
                                S[i][j] = 0.
                            elif t3 <= t1 < t4:
                                S[i][j] = (t4 - t1) / t
                            elif t3 < t2 <= t4:
                                S[i][j] = (t2 - t3) / t
                            else:
                                print("Something Wrong!")
            All[q-P,p,...]=S
    return All