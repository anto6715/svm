from cvxpy import *
import numpy as np
import sys

n_dataset = 40               # number of constraints

X = np.zeros(shape=(n_dataset, 2))
Y = np.zeros(shape=(n_dataset, 1))
A = np.zeros(shape=(n_dataset, 3))
u=0
for i in range(0, n_dataset):
    X[u] = np.loadtxt("../dataset", usecols=(0, 1))[i]
    Y[u] = np.loadtxt("../dataset", usecols=2)[i]
    A[u] = Y[u] * np.array([X[u][0], X[u][1], 1])
    u = u + 1

H = np.matrix('1, 0, 0;0, 1, 0; 0 ,0 ,0')

w = Variable(3)

obj = Minimize(0.5 * quad_form(w, H))
constraints = [A * w >= 1]  # va inserito vincolo Ax<b
prob = Problem(obj, constraints)
prob.solve()
print(w.value)
