import numpy as np
import matplotlib.pyplot as plt

print("Implentation of dual decomposition algorithm")
iteration = 200  #numero iterazioni
N=3  #numero agenti
n=2  #dimensione di R
n_lambda=N*N
alpha = 0.0568

x_avg = np.zeros(shape=(iteration, n))

x = np.zeros(shape=(iteration,N,n))

lambd = np.zeros(shape=(iteration, n_lambda, 2))
f_best = np.zeros(shape=(iteration, 1))

fval = np.zeros(shape=(iteration, N))

l = np.zeros(shape=(iteration, N, n))

A_tot = np.zeros(shape=(n,n))
A = np.zeros(shape=(N,n,n))
A[0][:]=([2,0],[0,4])
A[1][:]=([3,0],[0,6])
A[2][:]=([4,0],[0,8])

B_tot = np.zeros(shape=(1))
B = np.zeros(shape=(N,n))
B[0]=[1,1]
B[1]=[2,2]
B[2]=[3,3]



for t in range(0,iteration-1):
   # print("prova")
    for i in range(0,N):
        k = i*N
        k_temp = k
        k_temp2 = i
        temp = N
        temp2 = N
        #print("agente ", i)
        #calcolo sommatoria ij
        while temp > 0:
            l[t][i][:] = l[t][i][:]+ lambd[t][k_temp][:]
            k_temp = k_temp+1
            temp = temp-1

        #calcolo sommatoria ji
        while temp2 > 0:
            l[t][i][:] = l[t][i][:] - lambd[t][k_temp2][:]
            k_temp2 = k_temp2+N
            temp2 = temp2-1

        x[t][i][:] = (-1 / 2) * np.linalg.inv(A[i][:]).dot(B[i] + l[t][i][:])
        #print(x[t][i][:])
        x_avg[t] = x_avg[t] + x[t][i][:]

    x_avg[t] = x_avg[t]/3

    h=0
    for i in range(0,N):
        for j in range(0,N):
            lambd[t+1][h][:] = lambd[t][h][:] + alpha*(x[t][i][:]-x[t][j][:])
            h=h+1




#calcolo valori che la funzione assume nel tempo
for t in range(0,iteration):
   for i in range(0,N):
      fval[t][i] = np.dot(x_avg[t],A[i][:]).dot(x_avg[t]) + B[i].dot(x_avg[t])
      f_best[t] = f_best[t] + fval[t][i]



#Calcolo valore teorico
for j in range(0,N):
    A_tot = A_tot + A[j][:]
    B_tot = B_tot + B[j]
x_best = (-1 / 2) * np.linalg.inv((A_tot)).dot((B_tot))
fbest = np.dot(x_best,A_tot).dot(x_best) + B_tot.dot(x_best)



plt.axhline(y=fbest, color='r', linestyle='-', label="theory value")
plt.plot(f_best, label="function value")
plt.axis([0,iteration-10, -1.502,-1.494])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
