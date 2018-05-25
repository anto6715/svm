import numpy as np

N=10
A = np.zeros(shape=(N,N))

#np.savetxt("lambda",A)

a = np.loadtxt("lambda")[1][1]
print(a)