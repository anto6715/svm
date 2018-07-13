import numpy as np
from numpy.matlib import rand

N=5
#lambda matrix created with all zero
A = np.zeros(shape=(N,N))
#lambda matrix created with random values
#A = rand(N,N)

np.savetxt("lambda",A)
