from mpi4py import MPI
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import networkx as nx;
import sys
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

iteration = int(sys.argv[1])
n = 3
alpha = float(sys.argv[2])


A_tot = np.zeros(shape=(n, n))
B_tot = np.zeros(shape=(1))
fval = np.zeros(shape=(iteration, size))

if rank==0:
    for i in range(0,size):
        A_tot= A_tot+ np.load("A_matrices.npy")[i]
        B_tot = B_tot + np.load("B_matrices.npy")[i]
    x_opt = -np.matmul(np.linalg.inv((A_tot)),(B_tot).T)
    theory = 0.5 * np.dot(x_opt.T, A_tot).dot(x_opt) + B_tot.dot(x_opt)
    print(theory)
x_best= np.zeros(shape=(iteration,3))
f_best= np.zeros(shape=(iteration,1))
"""
if rank == 0:
    while (1):
        # Creo la matrice di adiacenza simmetrica e senza elf edges
        Adj = np.random.randint(2, size=(size, size))
        # print(Adj);

        I_n = np.eye(size, dtype=int)
        # print(I_n)
        Adj = np.bitwise_or(Adj, Adj.transpose())
        Adj = np.bitwise_or(Adj, I_n)
        Adj = Adj - I_n
        # print(Adj);

        G = nx.Graph(Adj)
        if (nx.is_connected(G)):
            print("The graph is connected \n")
            break;

    nx.draw(G)
    plt.show()
    # print(Adj)

    np.savetxt("adj_matrix", Adj)
"""
comm.Barrier()


B = np.load("B_matrices.npy")[rank]
A = np.load("A_matrices.npy")[rank]

# Costrusico vettore degli In-neighbor attraverso matrice di adiacenza
Set_ni = []
Adj = np.loadtxt("adj_matrix")[:][rank]  # preleva solo una riga
for j in range(0, size):
    if Adj[j] == 1 and j != rank:
        Set_ni.append(j)

l = np.zeros(shape=(iteration, 1, n))





B=np.load("B_matrices.npy")[rank]
A = np.load("A_matrices.npy")[rank] # preleva solo una riga




lambd = np.zeros(shape=(iteration + 1, 2 * len(Set_ni), n))
x = np.zeros(shape=(iteration, len(Set_ni) + 1, n))
prova = np.zeros(shape=(iteration, 1))

h=0
for z in Set_ni:
    lambd[0][h][:] = np.loadtxt("lambda")[z][rank]
    h = h + 1
comm.Barrier()


if rank == 0:
    error = np.zeros(shape=(iteration, 1))
    error_media = np.zeros(shape=(iteration, 1))
    x_media = np.zeros(shape=(iteration, 3))
    temp2=0
    f_media = np.zeros(shape=(iteration, 1))


for t in range(0, iteration):
    # sottraggo lambda ji
    for k in range(0, len(Set_ni)):
        l[t][0][:] = l[t][0][:] - lambd[t][k][:]
    # sommo lambda ij
    for k in range(len(Set_ni), 2 * len(Set_ni)):
        l[t][0][:] = l[t][0][:] + lambd[t][k][:]

    # minimizzazione attraverso cvx
    w = Variable(n)
    obj = Minimize(0.5 * (quad_form(w, A)) + (B+l[t][0][:]) * w)
    prob = Problem(obj)
    prob.solve()

    x[t][0][:] = np.transpose(w.value)

    # invio e ricezione stato con gli agenti vicini
    j = 1;
    for s in Set_ni:
        send = comm.isend(x[t][0][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        x[t][j][:] = req.wait()
        j = j + 1

    comm.Barrier()

    # quando tutti gli agenti hannos cambiato gli stati aggiornano le lmbda_ij e lambda_ji
    j = 1
    h = 0
    for k in range(0, len(Set_ni)):
        lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][j][:] - x[t][0][:])
        h = h + 1
        j = j + 1

    # invio lamnbda ji all'agente j
    i = 0
    for s in Set_ni:
        send = comm.isend(lambd[t + 1][i][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        lambd[t + 1][h][:] = req.wait()
        h = h + 1
        i = i + 1

    if rank != 0:
        send = comm.isend(x[t][0][:], dest=0, tag=0)
        send.wait()
    else:
        x_best[t]=x[t][0][:]
        for i in range(1,size):
            req = comm.irecv(source=i, tag=0)
            x_best[t] = x_best[t] + req.wait()
        x_best[t] = x_best[t]/size
        f_best[t]=(1/2)*(np.dot(x_best[t],A_tot).dot(x_best[t]))+ B_tot.dot(x_best[t])

        temp2 = temp2 + x_best[t]
        x_media[t] = temp2 / (t + 1)
        f_media[t] = (1/2)*(np.dot(x_media[t],A_tot).dot(x_media[t]))+ B_tot.dot(x_media[t])

        temp = theory - f_best[t]
        if temp != 0:
            if temp < 0:
                error[t] = math.log(-temp)  # calcolo errore tra valore terorico e valore della funzione
            else:
                error[t] = math.log(temp)

        temp = theory - f_media[t]
        if temp != 0:
            if temp < 0:
                error_media[t] = math.log(-temp)  # calcolo errore tra valore terorico e valore della funzione
            else:
                error_media[t] = math.log(temp)


print("agente:", rank, "X:", x[t][0][:])

if rank == 0:
    print(x_opt)
    #print(f_best[t])
    print(error)
    """plt.axhline(y=theory, color='r', linestyle='-', label="theory value")
    #print("f:",fbest)
    #print("valore teorico:", x_opt)
    plt.xlabel("iteration")
    plt.ylabel("function value")
    plt.title("alpha = 0.00168")
    plt.plot(f_best,label="function value")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
    plt.show()
"""

    plt.figure(1)
    plt.subplot(211).set_title("error function")
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(error, label="error function")
    plt.plot(error_media, label="error average function")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=0, borderaxespad=0.)

    plt.subplot(212).set_title("function value")
    plt.axhline(y=theory, color='r', linestyle='-', label="theory value")
    plt.plot(f_best, label="function value")
    plt.plot(f_media, label="average function value")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
    plt.show()