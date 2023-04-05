import numpy as np
import time
# np.random.seed(5)
from scipy.stats import spearmanr, pearsonr, kendalltau


a = np.random.rand(1, 30).reshape(-1) * 2
b = np.random.rand(1, 30).reshape(-1) * 3
c = np.random.rand(1, 30).reshape(-1) * 4

d = np.random.rand(1, 30).reshape(-1)
e = np.random.rand(1, 30).reshape(-1)
f = np.random.rand(1, 30).reshape(-1)


def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

def normalizeRows1(x):
    '''
    错误，无法保证 迹=3
    :param x:
    :return:
    '''
    minv = np.min(x, axis=1)
    maxv = np.max(x, axis=1)
    res = []
    for i in range(x.shape[1]):
        arr = x[:, i]
        arr = arr - minv
        base = (maxv-minv)
        arr = (arr * 1.0) / base # - 0.5
        res.append(arr)

    #a = np.asarray(res).T
    #a1 = a[0]
    #a2 = a[1]
    #a3 = a[2]
    return np.asarray(res).T

def standardizeRows(x):
    '''
    同样，也不行 无法保证约束
    :param x:
    :return:
    '''
    avg = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    res = []
    for i in range(x.shape[1]):
        arr = x[:, i]
        arr = arr - avg
        arr = arr/std
        res.append(arr)
    return np.asarray(res).T

def multirelation(y1, y2, y3):
    '''
    Y = [y1, y2, y3].T
    S = Normalized_row(Y)
    R = S * S.T
    r = 1 - Min(Var(R))
    cite: Multirelation - a correlation among more than two variables
    '''
    # assert len(y1) == len(y2) == len(y3)
    # Matrix_y = np.asarray(np.vstack((y1, y2, y3)))
    Matrix_y = np.asarray(np.vstack((y1, y2, y3)))

    #print(Matrix_y)
    Normalized_y = normalizeRows(Matrix_y)
    #Standardized_y = standardizeRows(Matrix_y)

    #print(Normalized_y)
    R_y = np.dot(Normalized_y, Normalized_y.T)  # a positive semi-definite symmetric matrix, k*k, 会形成复数解
    #R_y = np.dot(Standardized_y, Standardized_y.T)  # a positive semi-definite symmetric matrix n*n

    #print(R_y)
    eigenvalues, eigenvectors = np.linalg.eig(R_y)  # get the smallest eigenvalues
    #print(eigenvalues)
    #if min(eigenvalues) < 0:
    #    return 1 - abs(min(eigenvalues))
    return 1 - min(eigenvalues)
    # return min(eigenvalues)


#for i in range(100):
# print(multirelation(d, e, f))


def multiple_correlation_coefficient(d1, d2, d3, metric=kendalltau):
    '''
    :param d1: 1st variable vector
    :param d2: 2nd variable vector
    :param d3: 3rd variable vector
    :param metric: two-item correlation [spearmanr, pearsonr, kendalltau]
    :return: R: correlation of d1, d2 and d3, R is not an unbiased estimate
    ref: https://real-statistics.com/correlation/multiple-correlation/
    '''
    #assert len(d1) == len(d2) == len(d3)
    r12 = metric(d1, d2)[0]
    # r122 = metric(d1, d2).statistic
    r23 = metric(d2, d3)[0]
    r13 = metric(d1, d3)[0]
    #a = r12 ** 2 + r23 ** 2 - 2 * r12 * r23 * r13
    #b = 1 - r12 ** 2
    #c = np.sqrt(a/b)
    #print(a, b, c)
    r312 = np.sqrt((r12 ** 2 + r23 ** 2 - 2 * r12 * r23 * r13) / (1 - r12 ** 2))
    return r312

#print(multiple_correlation_coefficient(d, e, f, spearmanr))
#print(multiple_correlation_coefficient(e, f, d, spearmanr))
#print(multiple_correlation_coefficient(f, d, e, spearmanr))

def adjusted_multiple_correlation_coefficient(d1, d2, d3, metric=kendalltau):
    '''

    (R_adj)^2 = 1 - (1-R^2)(n-1)/(n-k-1)
    ref: https://real-statistics.com/correlation/multiple-correlation/
    :param d1: 1st variable vector
    :param d2: 2nd variable vector
    :param d3: 3rd variable vector
    :param metric: two-item correlation [spearmanr, pearsonr, kendalltau]
    :return: adjusted correlation of d1, d2 and d3, <0 useless
    '''

    r = multiple_correlation_coefficient(d1, d2, d3, metric)
    n = len(d1)
    k = 3
    #R_adj = 1 - (1 - r * r)(n - 1) / (n - k - 1)
    R_adj_2 = 1 - (1 - r * r) * (n - 1) / (n - k - 1)

    return R_adj_2

#print(adjusted_multiple_correlation_coefficient(d, e, f, spearmanr))
#print(adjusted_multiple_correlation_coefficient(e, f, d, spearmanr))
#print(adjusted_multiple_correlation_coefficient(f, d, e, spearmanr))

#a=np.array([1, 2, 3, 4])
#b=np.array([4, 5, 6, 7])
#c=np.array([7, 8, 9, 10])

#test2 = multirelation(a, b, c)
#print(test2)
















