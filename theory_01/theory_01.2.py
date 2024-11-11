import numpy as np

x_1 = np.array([1, 0.5])
x_2 = np.array([-0.5, -1])
x_3 = np.array([-0.5, 0])

y_all = np.array([1, 0, 0])

w_1 = (1 / np.sqrt(2)) * np.array([1, 1])
w_2 = np.array([1, 0])


def sgd(u):
    return 1 / (1 + np.exp(-u))


def neg_nll(w, x, y):
    p = sgd(np.dot(w, x.T))  # transponiert
    nll = -(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return nll


X = np.array([x_1, x_2, x_3])

nll_w1 = neg_nll(w_1, X, y_all)
nll_w2 = neg_nll(w_2, X, y_all)

print(nll_w1, nll_w2)

#nll_w1 = 1.1265
#nll_w2 = 1.2614
