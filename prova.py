import numpy as np
import matplotlib.pyplot as plt
import sys
# Fixing random state for reproducibility

n_constraints=10


#X = np.loadtxt("dataset",usecols=(0,1))
#Y = np.loadtxt("dataset",usecols=2, unpack=True)

X = np.zeros(shape=(n_constraints, 2))
Y = np.zeros(shape=(n_constraints, 1))


A= np.zeros(shape=(n_constraints,3))
#Y=Y.reshape(40,1)

#A= np.hstack((A,Y))
#print(A)

rank=1

u = 0
for i in range(rank * n_constraints, rank * n_constraints + n_constraints):
    X[u] = np.loadtxt("dataset completo", usecols=(0, 1))[i]
    Y[u] = np.loadtxt("dataset completo", usecols=2)[i]
    A[u] = Y[u]*np.array([X[u][0],X[u][1], 1])
    u = u + 1
print(Y)
print(A)