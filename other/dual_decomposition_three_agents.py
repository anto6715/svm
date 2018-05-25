import numpy as np
import matplotlib.pyplot as plt

print("Implentation of dual decomposition algorithm")
iteration = 300
n = 2
lam = np.zeros(shape=(iteration, 1))

alpha = 0.0568
x_best = np.array([0,0])
x = np.zeros(shape=(iteration, 2))
x1 = np.zeros(shape=(iteration, 2))
x2 = np.zeros(shape=(iteration, 2))
x3 = np.zeros(shape=(iteration, 2))

A1 = np.matrix('2, 0;0,4')
A2 = np.matrix('3, 0;0,6')
A3 = np.matrix('4, 0;0,8')
B1 = np.array([1, 1])
B2 = np.array([2, 2])
B3 = np.array([3, 3])

lam12 = np.zeros(shape=(iteration, 2))
lam13 = np.zeros(shape=(iteration, 2))

lam21 = np.zeros(shape=(iteration, 2))
lam23 = np.zeros(shape=(iteration, 2))

lam31 = np.zeros(shape=(iteration, 2))
lam32 = np.zeros(shape=(iteration, 2))

fval1 = np.zeros(shape=(iteration, 1))
fval2 = np.zeros(shape=(iteration, 1))
fval3 = np.zeros(shape=(iteration, 1))
fval = np.zeros(shape=(iteration, 1))

l1 = np.zeros(shape=(iteration, 2))
l2 = np.zeros(shape=(iteration, 2))
l3 = np.zeros(shape=(iteration, 2))

for t in range(0,iteration-1):
    x1[t] = (-1 / 2) * np.linalg.inv(A1).dot(B1 + lam12[t] - lam21[t])
    x2[t] = (-1 / 2) * np.linalg.inv(A2).dot(B2 + lam21[t] + lam23[t] - lam32[t] - lam12[t])
    x3[t] = (-1 / 2) * np.linalg.inv(A3).dot(B3 + lam32[t] - lam23[t])

    lam12[t + 1] = lam12[t] + alpha * (x1[t] - x2[t])
    lam21[t + 1] = lam21[t] + alpha * (x2[t] - x1[t])
    lam13[t + 1] = lam13[t] + alpha * (x1[t] - x3[t])
    lam23[t + 1] = lam23[t] + alpha * (x2[t] - x3[t])
    lam31[t + 1] = lam31[t] + alpha * (x3[t] - x1[t])
    lam32[t + 1] = lam32[t] + alpha * (x3[t] - x2[t])

    x[t] = (x1[t]+x2[t]+x3[t])/3
    fval1[t] = (x[t]*A1).dot(x[t]) + B1.dot(x[t])
    fval2[t] = (x[t]*A2).dot(x[t]) + B2.dot(x[t])
    fval3[t] = (x[t]*A3).dot(x[t]) + B3.dot(x[t])
    fval[t]  = (fval1[t]+fval2[t]+fval3[t])

#print("x1: ", x1[t], "x2: ", x2[t], "x3: ", x3[t])

x_best = (-1 / 2) * np.linalg.inv((A1+A2+A3)).dot((B1 + B2+B3))
print(x)

prova=-1.5

plt.axhline(y=-1.5, color='r', linestyle='-', label="theory value")
plt.plot(fval, label="function value")
plt.axis([0,iteration-10, -1.502,-1.494])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()