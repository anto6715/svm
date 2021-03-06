import numpy as np
from numpy.matlib import rand
from random import randint
def generateSPD(dim):
    A=rand(dim)
    A=0.5*(A+A.transpose())
    A=A+dim*np.eye(dim)
    A=A*randint(0,1000)
    return A

N=20
n=3
A = np.zeros(shape=(N,n,n))
B = np.zeros(shape=(N,1,n))

for i in range(0,N):
    A[i][:][:] = generateSPD(n)
    B[i][0][:] = np.random.randn(1,n)
    B[i][0][:] = B[i][0][:] * randint(0,800)

print(B)
np.save("A_matrices",A)
np.save("B_matrices",B)