from cvxpy import *
import numpy as np

A = np.loadtxt("A")
A = -A

H = np.matrix('1, 0, 0;0, 1, 0; 0 ,0 ,0')
eps = 0.0001

c = np.zeros(shape=(1, 3))
c[0][0] = 0
c[0][1] = 0
c[0][2] = eps

w = Variable(3)

obj = Minimize(0.5 * quad_form(w, H))
constraints = [A * w >= 1]  # va inserito vincolo Ax<b
prob = Problem(obj, constraints)
prob.solve()
a=(-1.10839884)**2+(-2.15925935)**2
print(w.value)
print(a/2)
